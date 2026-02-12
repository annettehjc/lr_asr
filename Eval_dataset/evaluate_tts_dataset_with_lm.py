#!/usr/bin/env python3
"""
TTS Evaluation Script with Language Model (LM) Integration.
Compares Greedy Decoding vs. LM-Beam Search Decoding on full sentences.
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

# Data Paths
SCRIPT_DIR = Path(__file__).parent.absolute()
DATASET_ROOT = SCRIPT_DIR / "A kiswahili Dataset for Development of Text-To-Speech System/Kiswahili Dataset"
METADATA_PATH = DATASET_ROOT / "Text/metadata.xlsx"
AUDIO_DIR = DATASET_ROOT / "wavs"

def normalize_text(text):
    if not isinstance(text, str): return ""
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
    # Match the model's 52 output units
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--output_dir", default="Eval_dataset/results_tts_lm")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Metadata
    print(f"📂 Loading metadata from: {METADATA_PATH}")
    df_raw = pd.read_excel(METADATA_PATH)
    
    # The excel row is one big string: "ID|transcribed|normalized"
    parsed_data = []
    for _, row in df_raw.iterrows():
        line = str(row.iloc[0])
        parts = line.split('|')
        if len(parts) >= 3:
            parsed_data.append({
                'audio_id': parts[0],
                'normalized_text': parts[2]
            })
    
    df_meta = pd.DataFrame(parsed_data)
    
    if args.max_samples:
        df_meta = df_meta.head(args.max_samples)
    
    processor, model, decoder = load_models_and_decoder()
    
    results = []
    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc="Evaluating TTS with LM"):
        try:
            audio_id = str(row['audio_id'])
            audio_path = AUDIO_DIR / f"{audio_id}.wav"
            
            if not audio_path.exists():
                print(f"⚠️ Audio not found: {audio_path}")
                continue
                
            speech, _ = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE)
            inputs = processor(speech, return_tensors="pt", sampling_rate=TARGET_SAMPLE_RATE).input_values.to(DEVICE)
            
            with torch.no_grad():
                logits = model(inputs).logits[0].cpu().numpy()
            
            # 1. Greedy Decoding
            pred_ids = np.argmax(logits, axis=-1)
            hyp_greedy = processor.decode(pred_ids)
            
            # 2. LM-Beam Search Decoding
            hyp_lm = decoder.decode(logits)
            
            ref = normalize_text(row['normalized_text'])
            h_greedy = normalize_text(hyp_greedy)
            h_lm = normalize_text(hyp_lm)
            
            results.append({
                'audio_id': audio_id,
                'reference': ref,
                'greedy_hyp': h_greedy,
                'lm_hyp': h_lm,
                'wer_greedy': wer(ref, h_greedy) if h_greedy else 1.0,
                'cer_greedy': cer(ref, h_greedy) if h_greedy else 1.0,
                'wer_lm': wer(ref, h_lm) if h_lm else 1.0,
                'cer_lm': cer(ref, h_lm) if h_lm else 1.0
            })
        except Exception as e:
            print(f"Error on {audio_id}: {e}")
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(output_dir / "comparison_results_tts.csv", index=False)
    
    # Global Summary
    print("\n" + "="*50)
    print("🏆 DECODING STRATEGY COMPARISON (TTS)")
    print("="*50)
    print(f"Metric | Greedy | LM-Beam")
    print("-" * 30)
    print(f"Mean WER | {res_df['wer_greedy'].mean():.4f} | {res_df['wer_lm'].mean():.4f}")
    print(f"Mean CER | {res_df['cer_greedy'].mean():.4f} | {res_df['cer_lm'].mean():.4f}")
    print("="*50)
    
    print(f"\n💾 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
