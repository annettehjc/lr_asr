#!/usr/bin/env python3
"""
Evaluates SSDD performance using Semantic-Constrained Mapping.
Maps ASR output to the nearest digit embedding (0-9).
"""

import os
import re
import argparse
import pandas as pd
import numpy as np
import torch
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from jiwer import wer, cer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Config
SCRIPT_DIR = Path(__file__).parent.absolute()
ASR_MODEL_NAME = "jkhyjkhy/wav2vec2-large-xlsr-sw-ASR"
SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TARGET_SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default Paths
DEFAULT_DATA_PATH = SCRIPT_DIR / "Spoken-Swahili-Digit-Dataset-master"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_ssdd_semantic"

# Vocabulary
DIGIT_MAP = {
    '0': 'sifuri', '1': 'moja', '2': 'mbili', '3': 'tatu', '4': 'nne',
    '5': 'tano', '6': 'sita', '7': 'saba', '8': 'nane', '9': 'tisa'
}
DIGIT_WORDS = list(DIGIT_MAP.values())

def normalize_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[.,;:!?\"\'-]', ' ', text)
    text = text.lower()
    return ' '.join(text.split())

def load_models():
    print(f"Loading ASR: {ASR_MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_NAME).to(DEVICE)
    model.eval()
    
    print(f"Loading Semantic Model: {SEMANTIC_MODEL_NAME}")
    semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    
    return processor, model, semantic_model

def get_ssdd_files(root_path):
    recordings_dir = Path(root_path) / "recordings-github"
    if not recordings_dir.exists():
        raise FileNotFoundError(f"Check data path: {recordings_dir}")
        
    data = []
    for speaker_dir in recordings_dir.iterdir():
        if not speaker_dir.is_dir(): continue
        
        # Parse speaker info
        parts = speaker_dir.name.split('_')
        # Format usually: digit_speakerID_gender
        speaker_id, gender = (parts[1], parts[2]) if len(parts) >= 3 else (speaker_dir.name, "unknown")
        
        for wav_file in speaker_dir.glob("*.wav*"):
            digit_label = wav_file.name.split('_')[0]
            if digit_label in DIGIT_MAP:
                data.append({
                    'audio_id': wav_file.name,
                    'speaker_id': speaker_id,
                    'gender': gender,
                    'digit_label': digit_label,
                    'reference': DIGIT_MAP[digit_label],
                    'audio_path': wav_file
                })
    return pd.DataFrame(data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = get_ssdd_files(args.data_path)
    if args.max_samples:
        df = df.sample(min(args.max_samples, len(df)), random_state=42)
    
    processor, asr_model, semantic_model = load_models()
    
    # Embed targets once
    print("Pre-computing digit embeddings...")
    digit_embeddings = semantic_model.encode(DIGIT_WORDS)
    
    results = []
    # Using tqdm for progress tracking
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        try:
            # Load Audio
            speech, _ = librosa.load(row['audio_path'], sr=TARGET_SAMPLE_RATE)
            inputs = processor(speech, return_tensors="pt", sampling_rate=TARGET_SAMPLE_RATE).input_values.to(DEVICE)
            
            # ASR Inference
            with torch.no_grad():
                logits = asr_model(inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            hyp_raw = processor.decode(pred_ids[0])
            hyp_norm = normalize_text(hyp_raw)
            
            # Semantic Matching Strategy
            best_match = ""
            confidence = 0.0
            
            if hyp_norm:
                hyp_embedding = semantic_model.encode([hyp_norm])
                sims = cosine_similarity(hyp_embedding, digit_embeddings)[0]
                best_idx = np.argmax(sims)
                best_match = DIGIT_WORDS[best_idx]
                confidence = sims[best_idx]
            
            ref = normalize_text(row['reference'])
            
            results.append({
                **row,
                'hyp_raw': hyp_raw,
                'hyp_semantic': best_match,
                'wer_raw': wer(ref, hyp_norm) if hyp_norm else 1.0,
                'wer_semantic': wer(ref, best_match) if best_match else 1.0,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Skipping {row['audio_id']}: {e}")
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_dir / "detailed_results_semantic.csv", index=False)
    
    # Summary
    print("\n--- Evaluation Summary (SSDD) ---")
    print(f"Samples: {len(res_df)}")
    print(f"Raw WER: {res_df['wer_raw'].mean():.4f}")
    print(f"Semantic Corrected WER: {res_df['wer_semantic'].mean():.4f}")
    
    # Gender Split
    print("\n[By Gender]")
    gender_stats = res_df.groupby('gender')['wer_semantic'].mean()
    for g, w in gender_stats.items():
        print(f"  {g}: {w:.4f}")
        
    print(f"\nFinal CSV: {output_dir / 'detailed_results_semantic.csv'}")

if __name__ == "__main__":
    main()
