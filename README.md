# Face Re-Identification Pipeline

A production-oriented face re-identification system built on top of InsightFace, ByteTrack, and PostgreSQL with pgvector. Designed for choke point scenarios (doorways, corridors, entry points) where the goal is to assign a stable identity to every person who appears, and detect when they return.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works — Step by Step](#how-it-works--step-by-step)
- [Design Decisions](#design-decisions)
- [Database Schema](#database-schema)
- [Configuration Parameters](#configuration-parameters)
- [Installation](#installation)
- [PostgreSQL Setup](#postgresql-setup)
- [Usage](#usage)
- [Output](#output)
- [Tuning Guide](#tuning-guide)
- [Known Limitations](#known-limitations)

---

## Overview

The system takes a video as input and produces:

1. An annotated output video with bounding boxes and identity labels on every detected face
2. A `face_crops/` folder with one reference image per new identity (named by their ID)
3. A PostgreSQL database of all known persons and their visit history
4. A printed visit log at the end of processing

The core idea is **track-then-identify**: rather than querying the database on every single frame, we use ByteTrack to follow each person across frames, accumulate embeddings while they are visible, and only commit an identity decision when the track ends. This avoids duplicate IDs caused by noisy single-frame detections.

---

## Architecture

```
InsightFace buffalo_l (CUDA)
    ├── SCRFD face detector
    └── ArcFace R100 embedding model (512-d)

ByteTrack (boxmot)
    └── Kalman filter + two-stage IoU matching

PostgreSQL + pgvector
    ├── persons table
    ├── person_embeddings table (ivfflat cosine index)
    └── visits table
```

---

## How It Works — Step by Step

### Step 1 — Face Detection

Each frame is passed through InsightFace's `buffalo_l` model, which runs SCRFD (Sample and Computation Redistributed Face Detector) at 640×640 resolution on CUDA. SCRFD returns bounding boxes, detection confidence scores (`det_score`), and 5-point facial landmarks (left eye, right eye, nose, left mouth corner, right mouth corner).

Faces smaller than **40px** in either dimension are dropped immediately. This is the hard floor — anything smaller is not usable by ByteTrack for stable tracking. Note this is different from the 80px threshold used for embeddings.

### Step 2 — ByteTrack

ByteTrack assigns a stable integer `track_id` to each face across frames. It uses a Kalman filter to predict where each face will be in the next frame, and a two-stage matching strategy:

- **Stage 1:** High-confidence detections (det_score > 0.5) are matched to existing tracks using IoU
- **Stage 2:** The remaining unmatched low-confidence detections (det_score > 0.1) are then tried against tracks that failed stage 1

This two-stage approach is why ByteTrack handles brief occlusions well — a face that dips below the confidence threshold for a few frames (someone briefly blocking the camera) is not lost. The `track_buffer=60` setting means a track survives up to 60 frames (~2 seconds at 30fps) without a detection before it is killed.

The result is each person walking through frame gets a single, stable `track_id` for their entire visible duration even if detection occasionally fails.

### Step 3 — Embedding Buffering

For each active track, every 3 frames (`EMBED_EVERY_N_FRAMES=3`), if the associated face is **>= 80px** in both width and height, the ArcFace embedding is extracted. The embedding is a 512-dimensional float vector, L2-normalised so that cosine similarity is equivalent to dot product.

Both the embedding and the raw face crop (with the `det_score` attached) are stored in the track's in-memory state. Nothing is written to the database at this stage.

The 80px threshold exists because small faces produce unreliable embeddings. ArcFace was trained on aligned 112×112 crops — a 40px detection that gets upscaled to 112px loses too much information.

### Step 4 — Track Death and Pairwise Filtering

When ByteTrack stops returning a `track_id` (the person has left the frame or been occluded too long), the track is considered dead and `process_dead_track()` is called.

This is where the core embedding quality logic runs:

**4a. Pairwise cosine similarity matrix**

All N buffered embeddings are stacked into an (N × 512) matrix. The pairwise similarity matrix is computed as a single matrix multiplication: `sim = emb_matrix @ emb_matrix.T`, producing an (N × N) matrix where `sim[i][j]` is the cosine similarity between embedding i and embedding j.

**4b. Mean similarity per embedding**

The diagonal is zeroed out (self-similarity = 1.0 would inflate scores), and the mean similarity for each embedding against all others is computed. This gives a score per embedding reflecting how consistent it is with the rest of the track's observations.

**4c. Survivor filtering**

Any embedding whose mean similarity falls below `PAIRWISE_THRESHOLD=0.6` is discarded. These are outlier frames — the person was turning away, partially occluded, or the detector fired on a low-quality crop. The remaining embeddings are the "consistent" ones.

**4d. Discard check**

If fewer than `MIN_SURVIVORS=3` embeddings survive the filter, the entire track is discarded. No identity is created. This handles cases like:
- Someone briefly passing at the very edge of frame
- A track that only ever saw a side profile
- A false detection that ByteTrack tracked for a few frames

**4e. Averaging**

The surviving embeddings are averaged element-wise and the result is L2-normalised. This produces a single 512-d representative vector for the person. Averaging works well here because the survivors are already mutually consistent (all scored above 0.6 pairwise), so they represent the same face from slightly different angles and lighting conditions.

### Step 5 — Database Query

The averaged vector is used to query the `person_embeddings` table using pgvector's cosine distance operator (`<=>`). The query returns the closest stored embedding across all known persons.

If the similarity score (`1 - cosine_distance`) is >= `SIMILARITY_THRESHOLD=0.5`:
- The person is a **returning visitor**
- Their `visit_count` is incremented, `last_seen` is updated, and a new row is added to the `visits` table
- Their embedding is **never modified** — it stays as it was when they were first seen

If no match is found above the threshold:
- The person is **new**
- A `Person_XXXXXXXX` UID is generated (random 8-character hex)
- Their averaged embedding is stored in `person_embeddings`
- A row is added to `persons` and `visits`
- The best face crop (highest `det_score` from the buffer) is saved to `face_crops/Person_XXXXXXXX.jpg`

---

## Design Decisions

**Why no embedding updates for returning persons?**

The embedding is built from multiple consistent frames during first sighting, averaged and filtered. It is a stable representation. Updating it on return introduces the risk of overwriting a clean embedding with a noisier one from a different lighting condition or angle. For re-identification purposes, a fixed reference is more reliable than a drifting one.

**Why track first, identify at death rather than identify per-frame?**

Querying the database on every frame for every face causes two problems: it is slow, and more critically, any frame where the face happens to be at a bad angle generates a garbage embedding that either creates a duplicate identity or fails to match the correct one. By accumulating and filtering first, the DB only ever sees high-quality averaged vectors.

**Why ArcFace specifically?**

ArcFace uses additive angular margin loss during training, which produces embeddings where intra-class distances (same person, different angles/lighting) are smaller than inter-class distances. It is the standard choice for face re-identification and consistently outperforms alternatives on benchmarks like LFW, CFP-FP, and AgeDB.

**Why ivfflat index?**

For small databases (< 100k persons) ivfflat with `lists=50` gives fast approximate nearest-neighbour search. If the database grows beyond 100k, switch to `hnsw` index type for better recall at scale.

---

## Database Schema

```sql
-- Identity registry
CREATE TABLE persons (
    id          SERIAL PRIMARY KEY,
    person_uid  TEXT UNIQUE NOT NULL,   -- e.g. "Person_FF2A139D"
    first_seen  TIMESTAMP DEFAULT NOW(),
    last_seen   TIMESTAMP DEFAULT NOW(),
    visit_count INTEGER DEFAULT 1
);

-- One averaged embedding per person, fixed at creation
CREATE TABLE person_embeddings (
    id           SERIAL PRIMARY KEY,
    person_uid   TEXT UNIQUE NOT NULL REFERENCES persons(person_uid) ON DELETE CASCADE,
    embedding    vector(512),           -- ArcFace 512-d, L2 normalised, averaged
    num_frames   INTEGER DEFAULT 0,     -- number of frames that went into the average
    created_at   TIMESTAMP DEFAULT NOW()
);

-- Every visit event
CREATE TABLE visits (
    id           SERIAL PRIMARY KEY,
    person_uid   TEXT NOT NULL,
    visited_at   TIMESTAMP DEFAULT NOW(),
    video_source TEXT                   -- path to the source video
);

-- Approximate nearest neighbour index
CREATE INDEX person_embeddings_vec_idx
ON person_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);
```

---

## Configuration Parameters

All constants are at the top of `face_reid_pipeline.py`:

| Parameter | Default | Description |
|---|---|---|
| `MIN_FACE_SIZE` | `80` | Minimum face width AND height in pixels to extract an embedding. Faces below this are tracked but not embedded. |
| `SIMILARITY_THRESHOLD` | `0.5` | Cosine similarity threshold for DB matching. Above this = same person. Raise to reduce false positives, lower to reduce false negatives. |
| `PAIRWISE_THRESHOLD` | `0.6` | Minimum mean pairwise cosine similarity for an embedding to survive the consistency filter. Embeddings below this are outliers and discarded. |
| `MIN_SURVIVORS` | `3` | Minimum number of consistent embeddings required to commit an identity. Tracks with fewer survivors are silently discarded. |
| `EMBED_EVERY_N_FRAMES` | `3` | Throttle — only extract an embedding every N frames per track. Reduces GPU load without losing coverage. |

ByteTrack parameters (in `run_pipeline`):

| Parameter | Value | Description |
|---|---|---|
| `track_high_thresh` | `0.5` | Minimum det_score for stage-1 matching |
| `track_low_thresh` | `0.1` | Minimum det_score for stage-2 (byte trick) |
| `new_track_thresh` | `0.6` | Minimum det_score to initialise a new track |
| `track_buffer` | `60` | Frames to keep a lost track alive via Kalman prediction |
| `match_thresh` | `0.8` | IoU threshold for Hungarian matching |

---

## Installation

```bash
# Create and activate virtualenv
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install insightface opencv-python-headless numpy psycopg2-binary pgvector onnxruntime-gpu boxmot

# InsightFace will auto-download the buffalo_l model on first run (~500MB)
# It is saved to ~/.insightface/models/buffalo_l/
```

GPU requirements:
- CUDA-capable GPU (any modern NVIDIA GPU)
- CUDA toolkit installed and on PATH
- `onnxruntime-gpu` must match your CUDA version — check the onnxruntime release page if you get CUDA errors

---

## PostgreSQL Setup

```bash
# Install PostgreSQL and pgvector extension
sudo apt install postgresql postgresql-contrib -y
sudo apt install postgresql-16-pgvector -y   # replace 16 with your postgres version

# Start the service
sudo service postgresql start

# Create database and set password
sudo -u postgres psql
```

Inside psql:

```sql
ALTER USER postgres PASSWORD 'yourpassword';
CREATE DATABASE reid;
\q
```

Enable the vector extension (the pipeline also does this automatically on startup):

```sql
\c reid
CREATE EXTENSION IF NOT EXISTS vector;
```

If you need to reset all data and start fresh:

```sql
DROP TABLE IF EXISTS visits;
DROP TABLE IF EXISTS person_embeddings;
DROP TABLE IF EXISTS persons;
-- Tables will be recreated on next pipeline run
```

---

## Usage

**Basic run:**
```bash
python face_reid_pipeline.py \
  --input /path/to/video.mp4 \
  --output /path/to/output.mp4 \
  --db-url "postgresql://postgres:yourpassword@localhost:5432/reid"
```

**All options:**
```bash
python face_reid_pipeline.py \
  --input video.mp4 \
  --output output_annotated.mp4 \
  --db-url "postgresql://postgres:yourpassword@localhost:5432/reid" \
  --threshold 0.5 \
  --min-size 80 \
  --crops-dir face_crops
```

**Converting ChokePoint dataset frames to video first:**
```bash
python frames_to_video.py \
  --input /path/to/P2E_S5_C1.1 \
  --output P2E_S5_C1.1.mp4 \
  --fps 15
```

---

## Output

**Annotated video:** bounding boxes drawn on every tracked face.
- Grey box with `Track_N (X embs)` = track active, still accumulating embeddings, identity not yet resolved
- Coloured box with `Person_XXXXXXXX [NEW]` = identity resolved, first sighting
- Coloured box with `Person_XXXXXXXX [RETURNING]` = identity resolved, returning visitor
- Each person UID gets a consistent colour derived from a hash of their UID

**face_crops/ directory:** one `.jpg` per new identity, named `Person_XXXXXXXX.jpg`. This is the single highest-confidence crop from the track's lifetime, useful for manually verifying the pipeline is assigning IDs correctly.

**Visit log** printed to stdout at the end:
```
======================================================================
VISIT LOG
======================================================================
  Person_FF2A139D         visits=  3  frames=  14  first=2024-01-15 10:23:01  last=2024-01-15 10:31:47
  Person_D641DBE0         visits=  1  frames=   8  first=2024-01-15 10:25:33  last=2024-01-15 10:25:33
======================================================================
```

---

## Tuning Guide

**Getting too many duplicate IDs (same person, multiple IDs):**
- Lower `SIMILARITY_THRESHOLD` (try 0.45)
- Lower `PAIRWISE_THRESHOLD` (try 0.5) so more frames survive the filter and the average is more stable
- Lower `MIN_SURVIVORS` (try 2) so short tracks don't get discarded

**Getting too many false matches (different people sharing an ID):**
- Raise `SIMILARITY_THRESHOLD` (try 0.55 or 0.6)
- Raise `PAIRWISE_THRESHOLD` (try 0.65) to make the average stricter
- Raise `MIN_SURVIVORS` (try 5) to require more evidence before committing

**Pipeline running slowly:**
- Increase `EMBED_EVERY_N_FRAMES` (try 5 or 10) to embed less often
- Reduce `det_size` in `face_app.prepare()` from 640 to 320 for faster detection at cost of small-face accuracy
- Make sure `onnxruntime-gpu` is actually using CUDA — check for `[INFO] Loading InsightFace buffalo_l on CUDA...` and no fallback warnings

**Too many tracks being discarded:**
- Lower `MIN_FACE_SIZE` (try 60) to start embedding earlier while the face is approaching
- Lower `track_buffer` if people move quickly through the scene so dead tracks are processed sooner

---

## Known Limitations

- **Identity is resolved at track death** — during a live track the bounding box shows `Track_N` not a person UID. For a real-time system you would need an early DB lookup strategy.
- **Single embedding per person** — the averaged vector is fixed at first sighting. If someone's appearance changes drastically (e.g. hat on vs hat off, heavy lighting change) across sessions, the similarity might dip below threshold and they get a second identity.
- **No frontality enforcement** — partial profiles can still make it through if they happen to be >= 80px. The pairwise filter helps reject these as outliers but does not guarantee it.
- **pgvector ivfflat requires minimum rows before index is useful** — with fewer than ~100 persons the sequential scan is actually faster; the index pays off at scale.