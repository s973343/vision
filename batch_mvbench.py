import argparse
import json
import os
import sys
import time

import config
import main
import audio_processing
import video_processing


VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".m4v"}


def build_video_index(video_dir):
    index = {}
    duplicates = {}
    for root, _, files in os.walk(video_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            path = os.path.join(root, name)
            for key in (name, os.path.splitext(name)[0]):
                if key in index:
                    duplicates.setdefault(key, []).append(path)
                else:
                    index[key] = path
    return index, duplicates


def load_json_items(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "annotations", "items", "examples"):
            val = data.get(key)
            if isinstance(val, list):
                return val
    return []


def resolve_video_path(entry, video_dir, index):
    raw = entry.get("video") or entry.get("video_path") or entry.get("video_id")
    if not raw:
        return None
    raw = os.path.normpath(raw)
    if os.path.isabs(raw) and os.path.exists(raw):
        return raw
    candidate = os.path.join(video_dir, raw)
    if os.path.exists(candidate):
        return candidate
    base = os.path.basename(raw)
    if base in index:
        return index[base]
    stem = os.path.splitext(base)[0]
    return index.get(stem)


def build_query_text(entry):
    question = (entry.get("question") or entry.get("query") or "").strip()
    candidates = entry.get("candidates") or entry.get("options") or []
    if isinstance(candidates, list) and candidates:
        opts = "; ".join(str(x) for x in candidates)
        if question:
            return f"{question} Options: {opts}"
        return f"Options: {opts}"
    return question


def set_pipeline_context(video_path, query_text, frame_dir, audio_out):
    # Update config
    config.VIDEO_INPUT = video_path
    config.USER_DESCRIPTION = query_text or config.USER_DESCRIPTION
    config.FRAME_DIR = frame_dir
    config.AUDIO_OUTPUT = audio_out

    # Update imported module globals (they use "from config import ...")
    main.VIDEO_INPUT = video_path
    main.USER_DESCRIPTION = query_text or main.USER_DESCRIPTION
    audio_processing.VIDEO_INPUT = video_path
    audio_processing.AUDIO_OUTPUT = audio_out
    video_processing.VIDEO_INPUT = video_path
    video_processing.FRAME_DIR = frame_dir


def run_batch(root_dir, json_dir, video_dir, frames_root, audio_root, limit, task):
    if not os.path.isdir(json_dir):
        print(f"Error: JSON dir not found: {json_dir}")
        return 1
    if not os.path.isdir(video_dir):
        print(f"Error: Video dir not found: {video_dir}")
        return 1

    print(f"Indexing videos in: {video_dir}")
    index, duplicates = build_video_index(video_dir)
    if duplicates:
        print(f"Warning: {len(duplicates)} duplicate video keys found. Using first match.")

    json_files = []
    for name in os.listdir(json_dir):
        if name.lower().endswith(".json"):
            if task and task != name and task != os.path.splitext(name)[0]:
                continue
            json_files.append(os.path.join(json_dir, name))
    json_files.sort()

    if not json_files:
        print("No JSON files found to process.")
        return 1

    total = 0
    missing = 0
    failures = 0

    for json_path in json_files:
        task_name = os.path.splitext(os.path.basename(json_path))[0]
        items = load_json_items(json_path)
        if not items:
            print(f"Skipping {task_name}: no items")
            continue

        print(f"\n=== Task: {task_name} ({len(items)} items) ===")
        for entry in items:
            if limit and total >= limit:
                print(f"Reached limit {limit}, stopping.")
                print(f"Done. processed={total} missing={missing} failures={failures}")
                return 0

            video_path = resolve_video_path(entry, video_dir, index)
            if not video_path:
                missing += 1
                continue

            query_text = build_query_text(entry)
            video_stem = os.path.splitext(os.path.basename(video_path))[0]
            frame_dir = os.path.join(frames_root, task_name, video_stem)
            audio_out = os.path.join(audio_root, f"{task_name}_{video_stem}.mp3")
            os.makedirs(frame_dir, exist_ok=True)
            os.makedirs(audio_root, exist_ok=True)

            set_pipeline_context(video_path, query_text, frame_dir, audio_out)

            total += 1
            print(f"\n[{total}] {task_name} | {video_stem}")
            start = time.time()
            try:
                main.run_ingestion_pipeline()
            except Exception as e:
                failures += 1
                print(f"Error: {e}")
                continue
            finally:
                elapsed = time.time() - start
                print(f"Elapsed: {elapsed:.2f}s")

    print(f"\nDone. processed={total} missing={missing} failures={failures}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Batch MVBench ingestion runner")
    parser.add_argument("--root", default="data/MVBench", help="MVBench root folder")
    parser.add_argument("--json-dir", default=None, help="Override JSON dir")
    parser.add_argument("--video-dir", default=None, help="Override video dir")
    parser.add_argument("--frames-root", default="data/frames/mvbench", help="Root for extracted frames")
    parser.add_argument("--audio-root", default="data/audio/mvbench", help="Root for extracted audio")
    parser.add_argument("--limit", type=int, default=0, help="Max total items to process (0 = no limit)")
    parser.add_argument("--task", default="", help="Process only one JSON task (name or filename)")
    return parser.parse_args()


def main_cli():
    args = parse_args()
    root_dir = args.root
    json_dir = args.json_dir or os.path.join(root_dir, "json")
    video_dir = args.video_dir or os.path.join(root_dir, "video")
    return run_batch(root_dir, json_dir, video_dir, args.frames_root, args.audio_root, args.limit, args.task)


if __name__ == "__main__":
    sys.exit(main_cli())
