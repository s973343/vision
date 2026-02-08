import os
import cv2
import torch
import clip
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ============================================================
# CONFIG
# ============================================================
VIDEO_PATH = r".\data\Trishul_480P.mp4"
OUTPUT_DIR = "video_segments"

CLIP_DURATION = 2.0          # seconds per clip
CHANGE_THRESHOLD = 0.25      # semantic change sensitivity
MIN_SEGMENT_DURATION = 4.0   # seconds

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD MODELS
# ============================================================
print("[INFO] Loading models...")

clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

blip_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)

# ============================================================
# STEP 1: COMPUTE ACTIVITY SEGMENTS (LABEL-FREE)
# ============================================================
print("[INFO] Computing activity segments...")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps

frames_per_clip = int(CLIP_DURATION * fps)

clip_embeddings = []
clip_times = []

clip_idx = 0
while True:
    frames = []
    for _ in range(frames_per_clip):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    if not frames:
        break

    mid_frame = frames[len(frames) // 2]
    image = Image.fromarray(cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB))
    image_input = clip_preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = clip_model.encode_image(image_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    clip_embeddings.append(emb.cpu().numpy()[0])

    start = clip_idx * CLIP_DURATION
    end = min(start + CLIP_DURATION, video_duration)
    clip_times.append((start, end))

    clip_idx += 1

cap.release()
clip_embeddings = np.vstack(clip_embeddings)

def cosine_distance(a, b):
    return 1.0 - np.dot(a, b)

change_scores = [
    cosine_distance(clip_embeddings[i], clip_embeddings[i - 1])
    for i in range(1, len(clip_embeddings))
]

change_scores = np.array(change_scores)
change_scores = change_scores / change_scores.max()

change_points = np.where(change_scores > CHANGE_THRESHOLD)[0] + 1

segments = []
start_idx = 0

for idx in change_points:
    s = clip_times[start_idx][0]
    e = clip_times[idx - 1][1]
    if e - s >= MIN_SEGMENT_DURATION:
        segments.append((s, e))
    start_idx = idx

segments.append((clip_times[start_idx][0], clip_times[-1][1]))

print(f"[INFO] Found {len(segments)} activity segments.")

# ============================================================
# STEP 2: AUTO-LABEL SEGMENTS
# ============================================================
print("[INFO] Auto-labeling segments...")

def generate_label(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = blip_processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=20)
    return blip_processor.decode(out[0], skip_special_tokens=True)

cap = cv2.VideoCapture(VIDEO_PATH)
labeled_segments = []

for (start, end) in segments:
    mid_time = (start + end) / 2
    cap.set(cv2.CAP_PROP_POS_MSEC, mid_time * 1000)
    ret, frame = cap.read()
    if not ret:
        label = "unknown activity"
    else:
        label = generate_label(frame)

    labeled_segments.append((start, end, label))

cap.release()

# ============================================================
# STEP 3: SAVE VIDEO SEGMENTS
# ============================================================
print("[INFO] Saving video segments...")

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

for i, (start, end, label) in enumerate(labeled_segments, 1):
    out_path = os.path.join(OUTPUT_DIR, f"segment_{i:03d}.mp4")

    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    start_frame = int(start * fps)
    end_frame = int(end * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current = start_frame
    while current < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        current += 1

    writer.release()

    print(f"Segment {i}:")
    print(f"  Time     : {start:.2f} – {end:.2f} s")
    print(f"  Activity : {label}")
    print(f"  Saved    : {out_path}\n")

cap.release()

print("[DONE] Video segmentation, labeling, and saving complete.")
