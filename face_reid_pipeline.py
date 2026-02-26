"""
Face Re-Identification Pipeline
================================
Flow:
  1. Detect faces via InsightFace buffalo_l (CUDA)
  2. Track with ByteTrack — stable track IDs across frames
  3. While tracked, buffer embeddings where face >= 80px
  4. On track death:
       a. Build pairwise cosine similarity matrix across buffered embeddings
       b. Keep only embeddings with mean pairwise sim >= 0.6 (consistent frames)
       c. If survivors < MIN_SURVIVORS (3) → discard track, no identity created
       d. Average survivors → single representative vector
       e. Query DB → match found: assign existing ID (no DB update)
                  → no match: insert new person with averaged embedding
  5. Returning persons: label drawn, embeddings never updated

Setup:
    pip install insightface opencv-python-headless numpy psycopg2-binary pgvector onnxruntime-gpu boxmot

    PostgreSQL (run once):
        CREATE EXTENSION IF NOT EXISTS vector;
        -- Tables auto-created on first run

Usage:
    python face_reid_pipeline.py --input video.mp4 --output output.mp4 --db-url postgresql://postgres:pass@localhost:5432/reid
"""

import cv2
import numpy as np
import argparse
import time
import uuid
import warnings
from typing import Optional

warnings.filterwarnings("ignore", message=".*estimate.*deprecated.*", category=FutureWarning)

import psycopg2
import psycopg2.extras
from insightface.app import FaceAnalysis
from boxmot import ByteTrack

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MIN_FACE_SIZE         = 80    # px — minimum face width AND height to buffer embedding
SIMILARITY_THRESHOLD  = 0.5   # cosine similarity to count as same person in DB
PAIRWISE_THRESHOLD    = 0.8   # minimum mean pairwise sim to keep an embedding
MIN_SURVIVORS         = 4     # minimum consistent embeddings needed to commit identity
EMBED_EVERY_N_FRAMES  = 2     # only embed every N frames per track (performance)


# ─────────────────────────────────────────────
# Pairwise embedding filtering + averaging
# ─────────────────────────────────────────────
def build_representative_embedding(embeddings: list[np.ndarray]) -> Optional[np.ndarray]:
    """
    Given a list of L2-normalised embeddings from a single track:
    1. Compute full pairwise cosine similarity matrix
    2. For each embedding, compute its mean similarity against all others
    3. Keep only embeddings whose mean >= PAIRWISE_THRESHOLD
    4. If survivors < MIN_SURVIVORS → return None (discard track)
    5. Average survivors and L2-normalise → representative vector
    """
    if len(embeddings) < MIN_SURVIVORS:
        return None

    emb_matrix = np.stack(embeddings)                        # (N, 512)
    sim_matrix = emb_matrix @ emb_matrix.T                   # (N, N) cosine sim (already normalised)

    N = len(embeddings)
    if N == 1:
        mean_sims = np.array([1.0])
    else:
        # Mean sim excluding self (diagonal)
        np.fill_diagonal(sim_matrix, 0.0)
        mean_sims = sim_matrix.sum(axis=1) / (N - 1)

    survivors = emb_matrix[mean_sims >= PAIRWISE_THRESHOLD]

    if len(survivors) < MIN_SURVIVORS:
        return None

    averaged = survivors.mean(axis=0)
    averaged = averaged / (np.linalg.norm(averaged) + 1e-8)
    return averaged


# ─────────────────────────────────────────────
# Per-track in-memory state
# ─────────────────────────────────────────────
class TrackState:
    def __init__(self):
        self._state: dict[int, dict] = {}

    def get(self, tid: int) -> dict:
        if tid not in self._state:
            self._state[tid] = {
                "person_uid":       None,    # set after DB lookup on track death
                "is_new":           True,
                "embeddings":       [],      # buffered embeddings (face >= 80px)
                "crops":            [],      # buffered face crops (np.ndarray) parallel to embeddings
                "last_embed_frame": -999,
                "bbox":             None,    # latest bbox for drawing
            }
        return self._state[tid]

    def pop_dead(self, active_ids: set[int]) -> list[dict]:
        """Return and remove states of tracks no longer active."""
        dead   = [s for tid, s in self._state.items() if tid not in active_ids]
        self._state = {tid: s for tid, s in self._state.items() if tid in active_ids}
        return dead


# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
class PersonDB:
    def __init__(self, db_url: str):
        self.conn = psycopg2.connect(db_url)
        self.conn.autocommit = True
        self._ensure_schema()

    def _ensure_schema(self):
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id          SERIAL PRIMARY KEY,
                    person_uid  TEXT UNIQUE NOT NULL,
                    first_seen  TIMESTAMP DEFAULT NOW(),
                    last_seen   TIMESTAMP DEFAULT NOW(),
                    visit_count INTEGER DEFAULT 1
                );
            """)
            # One averaged embedding per person — never updated after creation
            cur.execute("""
                CREATE TABLE IF NOT EXISTS person_embeddings (
                    id           SERIAL PRIMARY KEY,
                    person_uid   TEXT UNIQUE NOT NULL
                                 REFERENCES persons(person_uid) ON DELETE CASCADE,
                    embedding    vector(512),
                    num_frames   INTEGER DEFAULT 0,   -- how many frames went into the average
                    created_at   TIMESTAMP DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS visits (
                    id           SERIAL PRIMARY KEY,
                    person_uid   TEXT NOT NULL,
                    visited_at   TIMESTAMP DEFAULT NOW(),
                    video_source TEXT
                );
            """)
            try:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS person_embeddings_vec_idx
                    ON person_embeddings USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 50);
                """)
            except Exception:
                pass

    def _vec_str(self, emb: np.ndarray) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in emb.tolist()) + "]"

    def find_match(self, emb: np.ndarray) -> Optional[tuple[str, float]]:
        """Returns (person_uid, similarity) or None."""
        v = self._vec_str(emb)
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT person_uid,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM person_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT 1;
            """, (v, v))
            row = cur.fetchone()
        if row and float(row[1]) >= SIMILARITY_THRESHOLD:
            return row[0], float(row[1])
        return None

    def insert_person(self, emb: np.ndarray, num_frames: int, video_source: str) -> str:
        """Create new identity with averaged embedding. Never updated after this."""
        person_uid = f"Person_{uuid.uuid4().hex[:8].upper()}"
        v = self._vec_str(emb)
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO persons (person_uid) VALUES (%s);", (person_uid,))
            cur.execute("""
                INSERT INTO person_embeddings (person_uid, embedding, num_frames)
                VALUES (%s, %s::vector, %s);
            """, (person_uid, v, num_frames))
            cur.execute("""
                INSERT INTO visits (person_uid, video_source) VALUES (%s, %s);
            """, (person_uid, video_source))
        return person_uid

    def record_visit(self, person_uid: str, video_source: str):
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE persons SET last_seen = NOW(), visit_count = visit_count + 1
                WHERE person_uid = %s;
            """, (person_uid,))
            cur.execute("""
                INSERT INTO visits (person_uid, video_source) VALUES (%s, %s);
            """, (person_uid, video_source))

    def get_visit_log(self) -> list[dict]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT p.person_uid,
                       p.first_seen,
                       p.last_seen,
                       p.visit_count,
                       pe.num_frames
                FROM persons p
                LEFT JOIN person_embeddings pe ON pe.person_uid = p.person_uid
                ORDER BY p.last_seen DESC;
            """)
            return [dict(r) for r in cur.fetchall()]

    def close(self):
        self.conn.close()


# ─────────────────────────────────────────────
# Process a dead track — build embedding, query DB
# ─────────────────────────────────────────────
def process_dead_track(state: dict, db: PersonDB, video_source: str, crops_dir: str = None):
    embeddings = state["embeddings"]

    if not embeddings:
        return

    rep = build_representative_embedding(embeddings)
    if rep is None:
        print(f"  [DISCARD] track had {len(embeddings)} frames but < {MIN_SURVIVORS} consistent — discarded")
        return

    match = db.find_match(rep)
    if match:
        person_uid, similarity = match
        db.record_visit(person_uid, video_source)
        print(f"  [RETURNING] → {person_uid} (sim={similarity:.3f}, frames={len(embeddings)}, survivors={len(embeddings)})")
    else:
        person_uid = db.insert_person(rep, len(embeddings), video_source)
        print(f"  [NEW]       → {person_uid} (frames={len(embeddings)})")
        # Save best crop — pick the crop with highest det_score
        if crops_dir and state.get("crops"):
            import os
            os.makedirs(crops_dir, exist_ok=True)
            crops     = state["crops"]       # list of (det_score, np.ndarray)
            best_crop = max(crops, key=lambda x: x[0])[1]
            crop_path = os.path.join(crops_dir, f"{person_uid}.jpg")
            cv2.imwrite(crop_path, best_crop)
            print(f"  [CROP]      saved to {crop_path}")


# ─────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────
PALETTE = [
    (0, 255, 0), (255, 165, 0), (0, 200, 255), (255, 0, 128),
    (128, 0, 255), (255, 255, 0), (0, 128, 255), (0, 255, 128),
]

def uid_to_color(uid: Optional[str]) -> tuple:
    return (128, 128, 128) if uid is None else PALETTE[hash(uid) % len(PALETTE)]

def draw_track(frame, state: dict, tid: int):
    bbox = state["bbox"]
    if bbox is None:
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    person_uid = state["person_uid"]
    color = uid_to_color(person_uid)

    # Label: show person_uid if known, else show track ID while accumulating
    if person_uid:
        tag  = " [NEW]" if state["is_new"] else " [RETURNING]"
        text = f"{person_uid}{tag}"
    else:
        text = f"Track_{tid} ({len(state['embeddings'])} embs)"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def run_pipeline(input_path: str, output_path: str, db_url: str, crops_dir: str = "face_crops"):
    print("[INFO] Loading InsightFace buffalo_l on CUDA...")
    face_app = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    print("[INFO] Initialising ByteTrack...")
    tracker = ByteTrack(
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.6,
        track_buffer=60,
        match_thresh=0.8,
        frame_rate=30,
    )

    print("[INFO] Connecting to PostgreSQL...")
    db = PersonDB(db_url)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    track_state = TrackState()
    frame_idx   = 0

    print(f"[INFO] Processing {total} frames")
    print(f"[INFO] min_face={MIN_FACE_SIZE}px | pairwise_thresh={PAIRWISE_THRESHOLD} | min_survivors={MIN_SURVIVORS} | sim_thresh={SIMILARITY_THRESHOLD}")
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 1. Detect ────────────────────────────────────────────────────
        faces     = face_app.get(frame)
        det_list  = []
        face_objs = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            # Use 40px as the hard detection minimum — ByteTrack needs something to track
            if (x2 - x1) < 40 or (y2 - y1) < 40:
                continue
            conf = float(face.det_score) if hasattr(face, 'det_score') else 0.9
            det_list.append([x1, y1, x2, y2, conf, 0])
            face_objs.append(face)

        # ── 2. ByteTrack ─────────────────────────────────────────────────
        dets_np = np.array(det_list, dtype=float) if det_list else np.empty((0, 6), dtype=float)
        tracks  = tracker.update(dets_np, frame)

        active_ids = set()

        for track_row in tracks:
            x1, y1, x2, y2 = [int(v) for v in track_row[:4]]
            tid = int(track_row[4])
            active_ids.add(tid)

            state = track_state.get(tid)
            state["bbox"] = (x1, y1, x2, y2)

            # ── 3. Find closest face to this track ───────────────────────
            track_cx = (x1 + x2) / 2
            track_cy = (y1 + y2) / 2
            best_face, best_dist = None, float('inf')
            for face in face_objs:
                fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                dist = (track_cx - (fx1 + fx2) / 2) ** 2 + (track_cy - (fy1 + fy2) / 2) ** 2
                if dist < best_dist:
                    best_dist, best_face = dist, face

            # ── 4. Buffer embedding if face >= 80px (throttled) ──────────
            if (best_face is not None and
                    best_face.embedding is not None and
                    frame_idx - state["last_embed_frame"] >= EMBED_EVERY_N_FRAMES):

                fx1, fy1, fx2, fy2 = best_face.bbox.astype(int)
                face_w = fx2 - fx1
                face_h = fy2 - fy1

                if face_w >= MIN_FACE_SIZE and face_h >= MIN_FACE_SIZE:
                    emb = best_face.embedding.copy()
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                    state["embeddings"].append(emb)
                    state["last_embed_frame"] = frame_idx
                    # Buffer crop with det_score so we can pick the best one later
                    det_score = float(best_face.det_score) if hasattr(best_face, "det_score") else 0.9
                    crop = frame[max(0,fy1):fy2, max(0,fx1):fx2].copy()
                    state["crops"].append((det_score, crop))

            # ── 5. Draw ──────────────────────────────────────────────────
            draw_track(frame, state, tid)

        # ── 6. Process dead tracks ───────────────────────────────────────
        dead_states = track_state.pop_dead(active_ids)
        for dead in dead_states:
            process_dead_track(dead, db, input_path, crops_dir)

        # HUD
        cv2.putText(frame,
                    f"Frame {frame_idx}/{total}  |  Active tracks: {len(active_ids)}",
                    (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 200, 200), 1, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{frame_idx}/{total}]  {frame_idx / elapsed:.1f} fps")

    # Flush any tracks still alive at end of video
    print("[INFO] Flushing remaining live tracks...")
    remaining = track_state.pop_dead(set())
    for state in remaining:
        process_dead_track(state, db, input_path, crops_dir)

    cap.release()
    out.release()

    # ── Visit log ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VISIT LOG")
    print("=" * 70)
    log = db.get_visit_log()
    for row in log:
        print(f"  {row['person_uid']:22s}  "
              f"visits={row['visit_count']:3d}  "
              f"frames={str(row['num_frames']):>4}  "
              f"first={str(row['first_seen'])[:19]}  "
              f"last={str(row['last_seen'])[:19]}")
    print("=" * 70)
    elapsed = time.time() - t0
    print(f"\n[DONE] Output        → {output_path}")
    print(f"[DONE] Persons in DB : {len(log)}")
    print(f"[DONE] Face crops     → {crops_dir}/")
    print(f"[DONE] {frame_idx} frames in {elapsed:.1f}s  ({frame_idx / elapsed:.1f} fps avg)")

    db.close()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Re-ID — ByteTrack + pairwise embedding filter")
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--db-url",     required=True,
                        help="e.g. postgresql://postgres:pass@localhost:5432/reid")
    parser.add_argument("--threshold",  type=float, default=SIMILARITY_THRESHOLD,
                        help=f"DB match threshold (default: {SIMILARITY_THRESHOLD})")
    parser.add_argument("--min-size",   type=int,   default=MIN_FACE_SIZE,
                        help=f"Min face size to embed (default: {MIN_FACE_SIZE})")
    parser.add_argument("--crops-dir",  default="face_crops",
                        help="Directory to save best face crop per new identity (default: face_crops)")
    args = parser.parse_args()

    SIMILARITY_THRESHOLD = args.threshold
    MIN_FACE_SIZE        = args.min_size

    run_pipeline(args.input, args.output, args.db_url, crops_dir=args.crops_dir)