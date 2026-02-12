#!/usr/bin/env python3
"""
SSDD Evaluation with Semantic-Constrained Mapping.
Matches Greedy ASR output to the most semantically similar digit word.
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
# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
ASR_MODEL_NAME = "jkhyjkhy/wav2vec2-large-xlsr-sw-ASR"
SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TARGET_SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default Paths
DEFAULT_DATA_PATH = SCRIPT_DIR / "Spoken-Swahili-Digit-Dataset-master"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_ssdd_semantic"

# Swahili Digit Vocabulary
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
    print(f"🚀 Loading ASR model: {ASR_MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_NAME).to(DEVICE)
    model.eval()
    
    print(f"🧠 Loading Semantic model: {SEMANTIC_MODEL_NAME}")
    semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    
    return processor, model, semantic_model

def discover_ssdd_files(root_path):
    recordings_dir = Path(root_path) / "recordings-github"
    data = []
    if not recordings_dir.exists():
        raise FileNotFoundError(f"Directory not found: {recordings_dir}")
        
    for speaker_dir in recordings_dir.iterdir():
        if not speaker_dir.is_dir(): continue
        parts = speaker_dir.name.split('_')
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
    
    df = discover_ssdd_files(args.data_path)
    if args.max_samples:
        df = df.sample(min(args.max_samples, len(df)), random_state=42)
    
    processor, asr_model, semantic_model = load_models()
    
    # Pre-compute embeddings for the 10 digit words
    print(f"🪄 Pre-computing digit embeddings...")
    digit_embeddings = semantic_model.encode(DIGIT_WORDS)
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating SSDD (Semantic-Constrained)"):
        try:
            # 1. ASR Transcription (Greedy)
            speech, _ = librosa.load(row['audio_path'], sr=TARGET_SAMPLE_RATE)
            inputs = processor(speech, return_tensors="pt", sampling_rate=TARGET_SAMPLE_RATE).input_values.to(DEVICE)
            with torch.no_grad():
                logits = asr_model(inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            hyp_raw = processor.decode(pred_ids[0])
            hyp_norm = normalize_text(hyp_raw)
            
            # 2. Semantic Matching
            # If hyp is empty or just noise, we still try to match or handle it
            best_match = ""
            similarity_to_best = 0.0
            
            if hyp_norm:
                hyp_embedding = semantic_model.encode([hyp_norm])
                similarities = cosine_similarity(hyp_embedding, digit_embeddings)[0]
                best_idx = np.argmax(similarities)
                best_match = DIGIT_WORDS[best_idx]
                similarity_to_best = similarities[best_idx]
            else:
                # If model predicted nothing, we count it as error
                best_match = ""
            
            ref = normalize_text(row['reference'])
            
            results.append({
                **row,
                'hyp_raw': hyp_raw,
                'hyp_semantic': best_match,
                'wer_raw': wer(ref, hyp_norm) if hyp_norm else 1.0,
                'wer_semantic': wer(ref, best_match) if best_match else 1.0,
                'cer_semantic': cer(ref, best_match) if best_match else 1.0,
                'semantic_confidence': similarity_to_best
            })
        except Exception as e:
            print(f"Error on {row['audio_id']}: {e}")
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_dir / "detailed_results_semantic.csv", index=False)
    
    # Global Summary Comparison
    print("\n" + "="*60)
    print("🎯 SEMANTIC-CONSTRAINED EVALUATION SUMMARY (SSDD)")
    print("="*60)
    print(f"Metric | Raw (Greedy) | Semantic-Constrained")
    print("-" * 50)
    print(f"Mean WER | {res_df['wer_raw'].mean():.4f} | {res_df['wer_semantic'].mean():.4f}")
    print(f"Mean CER | - | {res_df['cer_semantic'].mean():.4f}")
    print(f"Mean Confidence | - | {res_df['semantic_confidence'].mean():.4f}")
    print("="*60)
    
    # Gender Analysis for Semantic Corrected
    gender_sum = res_df.groupby('gender')['wer_semantic'].mean().to_dict()
    print("\n👥 Semantic-Corrected WER by Gender:")
    for g, w in gender_sum.items():
        print(f"  {g}: {w:.4f}")
        
    print(f"\n💾 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
