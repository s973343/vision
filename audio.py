import os
import whisper
import librosa
import soundfile as sf

# =============================
# CONFIG
# =============================
AUDIO_PATH = r".\data\Trishul_480P.mp4"
OUTPUT_DIR = "speech_segments"
WHISPER_MODEL = "base"     # tiny | base | small | medium
MIN_SEGMENT_DURATION = 2.0  # seconds (merge shorter segments)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# Load Whisper model
# =============================
print("[INFO] Loading Whisper model...")
model = whisper.load_model(WHISPER_MODEL)

# =============================
# Transcribe (speech-aware)
# =============================
print("[INFO] Transcribing audio...")
result = model.transcribe(AUDIO_PATH,language="hi",      # audio language (Hindi)
        task="translate",   # OUTPUT IN ENGLISH
        fp16=False          # CPU-safe)
    )
# =============================
# Load audio safely (MP4-compatible)
# =============================
audio, sr = librosa.load(AUDIO_PATH, sr=None, mono=True)

# =============================
# Merge short Whisper segments
# =============================
merged_segments = []

buffer_start = None
buffer_end = None
buffer_text = ""

for seg in result["segments"]:
    start = seg["start"]
    end = seg["end"]
    text = seg["text"].strip()

    if buffer_start is None:
        buffer_start = start
        buffer_end = end
        buffer_text = text
    else:
        buffer_end = end
        buffer_text += " " + text

    # Flush buffer if long enough
    if (buffer_end - buffer_start) >= MIN_SEGMENT_DURATION:
        merged_segments.append({
            "start": buffer_start,
            "end": buffer_end,
            "text": buffer_text.strip()
        })
        buffer_start = None
        buffer_end = None
        buffer_text = ""

# Flush remaining buffer
if buffer_start is not None:
    merged_segments.append({
        "start": buffer_start,
        "end": buffer_end,
        "text": buffer_text.strip()
    })

# =============================
# Save speech-aware segments
# =============================
print(f"[INFO] Saving {len(merged_segments)} speech segments...\n")

for i, seg in enumerate(merged_segments, 1):
    start = seg["start"]
    end = seg["end"]
    text = seg["text"]

    start_sample = int(start * sr)
    end_sample = int(end * sr)

    segment_audio = audio[start_sample:end_sample]

    out_path = os.path.join(OUTPUT_DIR, f"segment_{i:03d}.wav")
    sf.write(out_path, segment_audio, sr)

    print(f"Segment {i}:")
    print(f"  Time     : {start:.2f} – {end:.2f} s")
    print(f"  Duration : {end - start:.2f} s")
    print(f"  Text     : {text}")
    print(f"  Saved    : {out_path}\n")

print("[DONE] Speech-aware segmentation complete.")
