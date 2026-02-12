#!/usr/bin/env python3
"""
SSDD Evaluation Script with Language Model (LM) Integration.
Compares Greedy Decoding vs. LM-Beam Search Decoding.
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
from pyctcdecode import build_ctcdecoder

# Configuration
ASR_MODEL_NAME = "jkhyjkhy/wav2vec2-large-xlsr-sw-ASR"
LM_PATH = "/Users/youngheejeong/Desktop/2025 summer/Automatic speech/ASR project/Eval_dataset/language_model/5gram.bin"
UNIGRAMS_PATH = "/Users/youngheejeong/Desktop/2025 summer/Automatic speech/ASR project/Eval_dataset/language_model/unigrams.txt"
TARGET_SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Swahili Digit Mapping
DIGIT_MAP = {
    '0': 'sifuri', '1': 'moja', '2': 'mbili', '3': 'tatu', '4': 'nne',
    '5': 'tano', '6': 'sita', '7': 'saba', '8': 'nane', '9': 'tisa'
}

def normalize_text(text):
    text = re.sub(r'[.,;:!?\"\'-]', ' ', text)
    text = text.lower()
    return ' '.join(text.split())

def load_models_and_decoder():
    print(f"🚀 Loading ASR model: {ASR_MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_NAME).to(DEVICE)
    model.eval()

    # Build Decoder
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    # The model has 52 output units, but tokenizer has 54. 
    # Use only the first 52 labels which match the CTC head.
    labels = [x[0].replace("|", " ") for x in sorted_vocab[:52]]
    
    print(f"📊 Model Vocab Size: 52 | Decoder Labels: {len(labels)}")
    print(f"🧠 Building LM Decoder from: {LM_PATH}")
    with open(UNIGRAMS_PATH, "r") as f:
        unigrams = [line.strip() for line in f.readlines()]
    
    decoder = build_ctcdecoder(
        labels,
        kenlm_model_path=LM_PATH,
        unigrams=unigrams,
    )
    return processor, model, decoder

def discover_ssdd_files(root_path):
    recordings_dir = Path(root_path) / "recordings-github"
    data = []
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
    parser.add_argument("--data_path", default="Eval_dataset/Spoken-Swahili-Digit-Dataset-master")
    parser.add_argument("--output_dir", default="Eval_dataset/results_ssdd_lm")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = discover_ssdd_files(args.data_path)
    if args.max_samples:
        df = df.sample(min(args.max_samples, len(df)), random_state=42)
    
    processor, model, decoder = load_models_and_decoder()
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating SSDD with LM"):
        try:
            speech, _ = librosa.load(row['audio_path'], sr=TARGET_SAMPLE_RATE)
            inputs = processor(speech, return_tensors="pt", sampling_rate=TARGET_SAMPLE_RATE).input_values.to(DEVICE)
            
            with torch.no_grad():
                logits = model(inputs).logits[0].cpu().numpy()
            
            # 1. Greedy Decoding
            pred_ids = np.argmax(logits, axis=-1)
            hyp_greedy = processor.decode(pred_ids)
            
            # 2. LM-Beam Search Decoding
            hyp_lm = decoder.decode(logits)
            
            ref = normalize_text(row['reference'])
            h_greedy = normalize_text(hyp_greedy)
            h_lm = normalize_text(hyp_lm)
            
            results.append({
                'audio_id': row['audio_id'],
                'digit': row['digit_label'],
                'gender': row['gender'],
                'reference': ref,
                'greedy_hyp': h_greedy,
                'lm_hyp': h_lm,
                'wer_greedy': wer(ref, h_greedy) if h_greedy else 1.0,
                'cer_greedy': cer(ref, h_greedy) if h_greedy else 1.0,
                'wer_lm': wer(ref, h_lm) if h_lm else 1.0,
                'cer_lm': cer(ref, h_lm) if h_lm else 1.0
            })
        except Exception as e:
            print(f"Error on {row['audio_id']}: {e}")
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_dir / "comparison_results_ssdd.csv", index=False)
    
    # Final Comparison Summary
    print("\n" + "="*50)
    print("🏆 DECODING STRATEGY COMPARISON (SSDD)")
    print("="*50)
    print(f"Metric | Greedy | LM-Beam")
    print("-" * 30)
    print(f"Mean WER | {res_df['wer_greedy'].mean():.4f} | {res_df['wer_lm'].mean():.4f}")
    print(f"Mean CER | {res_df['cer_greedy'].mean():.4f} | {res_df['cer_lm'].mean():.4f}")
    print("="*50)
    
    # Save statistics
    stats = {
        'greedy_wer': res_df['wer_greedy'].mean(),
        'lm_wer': res_df['wer_lm'].mean(),
        'greedy_cer': res_df['cer_greedy'].mean(),
        'lm_cer': res_df['cer_lm'].mean()
    }
    pd.DataFrame([stats]).to_csv(output_dir / "summary_statistics.csv", index=False)
    print(f"\n💾 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
