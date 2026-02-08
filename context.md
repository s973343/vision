# Project Context

## Overview
This workspace contains a video understanding + RAG ingestion pipeline. The pipeline extracts audio and keyframes from a video, generates captions and causal reasoning, fuses text+image embeddings, and stores them in ChromaDB for later retrieval.

## Main Pipeline (main.py)
1. Extract audio and transcribe (Whisper).
2. Extract up to 100 keyframes (OpenCV).
3. For each frame: generate short/long captions (Nebius VLM), derive causal text, fuse embeddings (CLIP), and store in ChromaDB.

## Key Files
- main.py: Orchestrates the ingestion pipeline.
- audio_processing.py: Audio extraction + Whisper transcription.
- video_processing.py: Keyframe extraction with OpenCV.
- image_captioning.py: Nebius VLM captioning (short/long).
- causal_analysis.py: Causal reasoning using Nebius LLM.
- knowledge_base.py: CLIP embeddings + ChromaDB storage.
- config.py: Paths, model names, API keys, user description.
- rag_query.py: (active tab) likely handles retrieval queries.

## External Services
- Nebius API (OpenAI-compatible base URL) for VLM/LLM.
- ChromaDB for vector storage.
- OpenAI Whisper for transcription.
- OpenAI CLIP for embeddings.

## Notes
- Config uses .env for NEBIUS_API_KEY and GROQ_API_KEY.
- Video input default: data/Trishul_480P.mp4
