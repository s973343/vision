import argparse
import json
import os
import re
import sys
import time

import clip
import torch

from rag_query import query_video_rag


VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".m4v"}


def build_video_index(video_dir):
    index = {}
    for root, _, files in os.walk(video_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            path = os.path.join(root, name)
            index[name] = path
            index[os.path.splitext(name)[0]] = path
    return index


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
    if video_dir:
        candidate = os.path.join(video_dir, raw)
        if os.path.exists(candidate):
            return candidate
    base = os.path.basename(raw)
    return index.get(base) or index.get(os.path.splitext(base)[0])


def normalize_answer(text):
    if text is None:
        return ""
    return str(text).strip().lower()


def extract_first_number(text):
    if not text:
        return None
    m = re.search(r"\b(\d+)\b", text)
    return m.group(1) if m else None


def pick_prediction(raw_response, candidates):
    # Take first line before evidence/citations if present
    if not raw_response:
        return ""
    head = raw_response.split("\n\n", 1)[0].strip()
    lower_head = head.lower()

    # If candidates are present, try to match one directly
    if isinstance(candidates, list) and candidates:
        for c in candidates:
            c_str = normalize_answer(c)
            if c_str and c_str in lower_head:
                return c_str
        # fallback: number match
        num = extract_first_number(lower_head)
        if num:
            for c in candidates:
                if normalize_answer(c) == num:
                    return num

    # Fallback: first number, else full head
    num = extract_first_number(lower_head)
    if num:
        return num
    return normalize_answer(head)


def tokenize_text(text):
    return re.findall(r"[a-z0-9]+", normalize_answer(text))


def bleu1(pred, ref):
    pred_toks = tokenize_text(pred)
    ref_toks = tokenize_text(ref)
    if not pred_toks:
        return 0.0
    ref_counts = {}
    for t in ref_toks:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    match = 0
    for t in pred_toks:
        if ref_counts.get(t, 0) > 0:
            match += 1
            ref_counts[t] -= 1
    precision = match / len(pred_toks)
    return precision


def rouge_l(pred, ref):
    pred_toks = tokenize_text(pred)
    ref_toks = tokenize_text(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    # LCS dynamic programming
    dp = [[0] * (len(ref_toks) + 1) for _ in range(len(pred_toks) + 1)]
    for i in range(1, len(pred_toks) + 1):
        for j in range(1, len(ref_toks) + 1):
            if pred_toks[i - 1] == ref_toks[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]
    recall = lcs / len(ref_toks)
    return recall


def text_cosine_similarity(model, text_a, text_b):
    if not text_a or not text_b:
        return 0.0
    tokens = clip.tokenize([text_a, text_b], truncate=True).to("cpu")
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return float(torch.sum(feats[0] * feats[1]).item())


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate JSON QA against ChromaDB + RAG")
    parser.add_argument("--json", default="", help="Path to JSON file")
    parser.add_argument("--json-dir", default="", help="Folder with multiple JSON files")
    parser.add_argument("--task", default="", help="Process only one JSON (name or filename)")
    parser.add_argument("--video-dir", default="", help="Folder with videos")
    parser.add_argument("--root", default="", help="Dataset root (used if --video-dir not provided)")
    parser.add_argument("--frames-root", default="", help="Root for frames (optional, for image attachments)")
    parser.add_argument("--attach-images", action="store_true", help="Attach frames to the VLM prompt")
    parser.add_argument("--limit", type=int, default=0, help="Max items to evaluate (0 = no limit)")
    parser.add_argument("--out", default="eval_results.jsonl", help="Output JSONL path")
    return parser.parse_args()


def main():
    args = parse_args()

    json_paths = []
    if args.json:
        if not os.path.exists(args.json):
            print(f"Error: JSON not found: {args.json}")
            return 1
        json_paths = [args.json]
    elif args.json_dir:
        if not os.path.isdir(args.json_dir):
            print(f"Error: JSON dir not found: {args.json_dir}")
            return 1
        for name in os.listdir(args.json_dir):
            if not name.lower().endswith(".json"):
                continue
            if args.task and args.task not in (name, os.path.splitext(name)[0]):
                continue
            json_paths.append(os.path.join(args.json_dir, name))
        json_paths.sort()
    else:
        print("Error: Provide --json or --json-dir")
        return 1

    video_dir = args.video_dir
    if not video_dir and args.root:
        video_dir = os.path.join(args.root, "video")

    index = build_video_index(video_dir) if video_dir and os.path.isdir(video_dir) else {}
    model, _ = clip.load("ViT-B/16", device="cpu")

    total = 0
    correct = 0
    missing = 0
    failures = 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out_f:
        for json_path in json_paths:
            items = load_json_items(json_path)
            if not items:
                print(f"Skipping {json_path}: no items")
                continue

            task_name = os.path.splitext(os.path.basename(json_path))[0]
            for entry in items:
                if args.limit and total >= args.limit:
                    break

                video_path = resolve_video_path(entry, video_dir, index)
                if not video_path:
                    missing += 1
                    continue

                video_file = os.path.basename(video_path)
                video_stem = os.path.splitext(video_file)[0]
                frame_dir = ""
                if args.frames_root:
                    frame_dir = os.path.join(args.frames_root, task_name, video_stem)

                question = (entry.get("question") or entry.get("query") or "").strip()
                answer = normalize_answer(entry.get("answer"))
                candidates = entry.get("candidates") or entry.get("options") or []

                total += 1
                start = time.time()
                try:
                    raw = query_video_rag(
                        question,
                        video_filename=video_file,
                        frame_dir=frame_dir or None,
                        attach_images=args.attach_images,
                    )
                except Exception as e:
                    failures += 1
                    out_f.write(json.dumps({
                        "task": task_name,
                        "video": video_file,
                        "question": question,
                        "answer": answer,
                        "prediction": "",
                        "correct": False,
                        "error": str(e)
                    }) + "\n")
                    continue
                elapsed = time.time() - start

            pred = pick_prediction(raw, candidates)
            is_correct = pred == answer
            if is_correct:
                correct += 1
            cos_sim = text_cosine_similarity(model, pred, answer)
            bleu = bleu1(pred, answer)
            rouge = rouge_l(pred, answer)

            out_f.write(json.dumps({
                "task": task_name,
                "video": video_file,
                "question": question,
                "answer": answer,
                "prediction": pred,
                "correct": is_correct,
                "cosine_sim": round(cos_sim, 4),
                "bleu_1": round(bleu, 4),
                "rouge_l": round(rouge, 4),
                "elapsed_s": round(elapsed, 2)
            }) + "\n")

    acc = (correct / total) * 100 if total else 0.0
    print(f"Done. total={total} correct={correct} missing={missing} failures={failures} acc={acc:.2f}%")
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
