#!/usr/bin/env python3
"""
Evaluates ASR performance on the Kiswahili TTS dataset using Wav2Vec2.
Compares Greedy Decoding vs. KenLM-based Beam Search.
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

# Config
ASR_MODEL_NAME = "jkhyjkhy/wav2vec2-large-xlsr-sw-ASR"
LM_PATH = Path("Eval_dataset/language_model/5gram.bin")
UNIGRAMS_PATH = Path("Eval_dataset/language_model/unigrams.txt")
TARGET_SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data paths
SCRIPT_DIR = Path(__file__).parent.absolute()
# Allow overriding dataset root via env var or arg, but default to relative path
DATASET_ROOT = SCRIPT_DIR / "A kiswahili Dataset for Development of Text-To-Speech System/Kiswahili Dataset"
METADATA_PATH = DATASET_ROOT / "Text/metadata.xlsx"
AUDIO_DIR = DATASET_ROOT / "wavs"

def normalize_text(text):
    if not isinstance(text, str): return ""
    # Remove punctuation and lowercase
    text = re.sub(r'[.,;:!?\"\'-]', ' ', text)
    text = text.lower()
    return ' '.join(text.split())

def load_models():
    print(f"Loading model: {ASR_MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL_NAME).to(DEVICE)
    model.eval()

    # Setup Decoder
    vocab = processor.tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    # Wav2Vec2-XLSR-53 has 52 output units usually, need to match CTC head
    labels = [x[0].replace("|", " ") for x in sorted_vocab[:52]]
    
    print(f"Building decoder with LM: {LM_PATH.name}")
    with open(UNIGRAMS_PATH, "r") as f:
        unigrams = [line.strip() for line in f.readlines()]
    
    decoder = build_ctcdecoder(
        labels,
        kenlm_model_path=str(LM_PATH),
        unigrams=unigrams,
    )
    return processor, model, decoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for quick testing")
    parser.add_argument("--output_dir", default="Eval_dataset/results_tts_lm")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metadata loading
    print(f"Reading metadata from {METADATA_PATH}")
    df_raw = pd.read_excel(METADATA_PATH)
    
    # Metadata format is "ID|transcribed|normalized"
    parsed_data = []
    for _, row in df_raw.iterrows():
        line = str(row.iloc[0])
        parts = line.split('|')
        if len(parts) >= 3:
            parsed_data.append({
                'audio_id': parts[0],
                'normalized_text': parts[2]
            })
    
    df = pd.DataFrame(parsed_data)
    if args.max_samples:
        df = df.head(args.max_samples)
    
    processor, model, decoder = load_models()
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            audio_id = str(row['audio_id'])
            audio_path = AUDIO_DIR / f"{audio_id}.wav"
            
            if not audio_path.exists():
                print(f"Missing file: {audio_path}")
                continue
                
            speech, _ = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE)
            inputs = processor(speech, return_tensors="pt", sampling_rate=TARGET_SAMPLE_RATE).input_values.to(DEVICE)
            
            with torch.no_grad():
                logits = model(inputs).logits[0].cpu().numpy()
            
            # Greedy
            pred_ids = np.argmax(logits, axis=-1)
            hyp_greedy = processor.decode(pred_ids)
            
            # Beam Search
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
            print(f"Failed on {audio_id}: {e}")
            
    res_df = pd.DataFrame(results)
    out_path = output_dir / "comparison_results_tts.csv"
    res_df.to_csv(out_path, index=False)
    
    # Summary
    print("\n--- Results Summary (TTS) ---")
    print(f"Samples processed: {len(res_df)}")
    print(f"Greedy WER: {res_df['wer_greedy'].mean():.4f} | CER: {res_df['cer_greedy'].mean():.4f}")
    print(f"LM-Beam WER: {res_df['wer_lm'].mean():.4f} | CER: {res_df['cer_lm'].mean():.4f}")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()
