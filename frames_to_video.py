"""
ChokePoint Frames → Video
=========================
Reads sequentially named image frames from a ChokePoint dataset folder
and compiles them into an MP4 video.

Usage:
    python frames_to_video.py --input /path/to/P2E_S5_C1.1 --output output.mp4
    python frames_to_video.py --input /path/to/P2E_S5_C1.1 --output output.mp4 --fps 15
"""

import cv2
import os
import argparse
from pathlib import Path


SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def frames_to_video(input_dir: str, output_path: str, fps: float = 15.0):
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    # Collect all image files and sort them — sorting by name gives correct order
    # for ChokePoint naming convention (P2E_S5_C1.1_000001.jpg etc.)
    frames = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTS
    ])

    if not frames:
        raise ValueError(f"No image files found in: {input_dir}")

    print(f"[INFO] Found {len(frames)} frames in '{input_dir}'")
    print(f"[INFO] First frame : {frames[0].name}")
    print(f"[INFO] Last frame  : {frames[-1].name}")
    print(f"[INFO] FPS         : {fps}")

    # Read first frame to get dimensions
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise ValueError(f"Could not read first frame: {frames[0]}")

    height, width = first.shape[:2]
    print(f"[INFO] Resolution  : {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    for i, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"  [WARN] Skipping unreadable frame: {frame_path.name}")
            continue

        # Sanity check — skip frames with unexpected dimensions
        if img.shape[:2] != (height, width):
            print(f"  [WARN] Skipping frame with mismatched size: {frame_path.name}")
            continue

        writer.write(img)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(frames)}] written...")

    writer.release()
    print(f"\n[DONE] Video saved to: {output_path}")
    print(f"[DONE] Duration: {len(frames)/fps:.1f}s  ({len(frames)} frames @ {fps} fps)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChokePoint frames → MP4 video")
    parser.add_argument("--input",  required=True, help="Folder containing image frames")
    parser.add_argument("--output", required=True, help="Output .mp4 path")
    parser.add_argument("--fps",    type=float, default=15.0,
                        help="Frames per second for output video (default: 15)")
    args = parser.parse_args()

    frames_to_video(args.input, args.output, args.fps)