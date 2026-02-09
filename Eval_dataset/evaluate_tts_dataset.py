#!/usr/bin/env python3
"""
Evaluation script for Kiswahili TTS dataset using Wav2Vec2 ASR model.
Calculates WER, CER, and Semantic Error Rate.
"""

import os
import pandas as pd
import numpy as np
import torch
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from jiwer import wer, cer

# ============================================================================
# Configuration
# ============================================================================

# Paths - use absolute paths based on script location, change it at your disposal!
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_ROOT = SCRIPT_DIR / "A kiswahili Dataset for Development of Text-To-Speech System/Kiswahili Dataset"
METADATA_PATH = DATASET_ROOT / "Text/metadata.xlsx"
AUDIO_DIR = DATASET_ROOT / "wavs"
OUTPUT_DIR = SCRIPT_DIR / "results"

# Model configuration
MODEL_NAME = "jkhyjkhy/wav2vec2-large-xlsr-sw-ASR"
SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Audio settings
TARGET_SAMPLE_RATE = 16000

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Step 1: Parse Metadata
# ============================================================================

def parse_metadata(metadata_path):
    """
    Parse the metadata.xlsx file to extract audio IDs and normalized transcripts.
    
    Format: ID|transcribed_text|normalized_text
    """
    print("📖 Parsing metadata...")
    
    # Read Excel file
    df = pd.read_excel(metadata_path, header=None)
    
    # The data is in a single column with pipe-delimited format
    parsed_data = []
    
    for idx, row in df.iterrows():
        # Get the cell content
        cell_content = str(row[0])
        
        # Split by pipe
        parts = cell_content.split('|')
        
        if len(parts) >= 3:
            audio_id = parts[0].strip()
            transcribed = parts[1].strip()
            normalized = parts[2].strip()
            
            parsed_data.append({
                'audio_id': audio_id,
                'transcribed_text': transcribed,
                'normalized_text': normalized,
                'audio_path': AUDIO_DIR / f"{audio_id}.wav"
            })
    
    result_df = pd.DataFrame(parsed_data)
    
    # Filter out entries where audio file doesn't exist
    result_df = result_df[result_df['audio_path'].apply(lambda x: x.exists())]
    
    print(f"✅ Parsed {len(result_df)} valid audio-transcript pairs")
    
    return result_df

# ============================================================================
# Step 2: Load Models
# ============================================================================

def load_asr_model():
    """Load Wav2Vec2 ASR model from HuggingFace."""
    print(f"🤖 Loading ASR model: {MODEL_NAME}")
    
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    
    print("✅ ASR model loaded")
    return processor, model

def load_semantic_model():
    """Load multilingual sentence transformer for semantic similarity."""
    print(f"🧠 Loading semantic model: {SEMANTIC_MODEL_NAME}")
    
    semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    
    print("✅ Semantic model loaded")
    return semantic_model

# ============================================================================
# Step 3: ASR Inference
# ============================================================================

def transcribe_audio(audio_path, processor, model):
    """
    Transcribe a single audio file using Wav2Vec2.
    
    Args:
        audio_path: Path to audio file
        processor: Wav2Vec2Processor
        model: Wav2Vec2ForCTC model
    
    Returns:
        Transcribed text
    """
    # Load and resample audio
    speech, _ = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE)
    
    # Process audio
    inputs = processor(
        speech,
        return_tensors="pt",
        padding=True,
        sampling_rate=TARGET_SAMPLE_RATE
    )
    
    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Greedy decoding
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]
    
    return transcription

# ============================================================================
# Step 4: Text Normalization & Metrics
# ============================================================================

def normalize_text(text):
    """
    Normalize text for fair ASR evaluation.
    
    - Remove punctuation (.,;:!?\"'-)
    - Convert to lowercase
    
    This is necessary because Wav2Vec2 models typically:
    - Do not predict punctuation
    - Output only lowercase text
    """
    import re
    # Remove punctuation
    text = re.sub(r'[.,;:!?"''-]', ' ', text)  # Replace with space
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def calculate_wer_cer(reference, hypothesis, normalize=True):
    """
    Calculate Word Error Rate and Character Error Rate.
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        normalize: Whether to apply text normalization (default: True)
    
    Returns:
        Tuple of (wer_score, cer_score)
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    
    wer_score = wer(reference, hypothesis)
    cer_score = cer(reference, hypothesis)
    return wer_score, cer_score

def calculate_semantic_error_rate(reference, hypothesis, semantic_model):
    """
    Calculate Semantic Error Rate using sentence embeddings.
    
    SER = 1 - cosine_similarity
    """
    # Generate embeddings
    ref_embedding = semantic_model.encode([reference])
    hyp_embedding = semantic_model.encode([hypothesis])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(ref_embedding, hyp_embedding)[0][0]
    
    # Semantic error rate
    ser = 1 - similarity
    
    return ser, similarity

# ============================================================================
# Step 5: Main Evaluation Loop
# ============================================================================

def evaluate_dataset(metadata_df, processor, asr_model, semantic_model, max_samples=None):
    """
    Run evaluation on the entire dataset.
    
    Args:
        metadata_df: DataFrame with audio paths and transcripts
        processor: Wav2Vec2Processor
        asr_model: Wav2Vec2ForCTC model
        semantic_model: SentenceTransformer model
        max_samples: Optional limit on number of samples to evaluate
    
    Returns:
        DataFrame with results
    """
    print(f"\n🚀 Starting evaluation on {len(metadata_df)} samples...")
    
    if max_samples:
        metadata_df = metadata_df.head(max_samples)
        print(f"   (Limited to {max_samples} samples for testing)")
    
    results = []
    
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Evaluating"):
        try:
            # Transcribe audio
            hypothesis = transcribe_audio(row['audio_path'], processor, asr_model)
            reference = row['normalized_text']
            
            # Calculate metrics
            wer_score, cer_score = calculate_wer_cer(reference, hypothesis)
            ser_score, similarity = calculate_semantic_error_rate(reference, hypothesis, semantic_model)
            
            results.append({
                'audio_id': row['audio_id'],
                'reference': reference,
                'hypothesis': hypothesis,
                'wer': wer_score,
                'cer': cer_score,
                'ser': ser_score,
                'semantic_similarity': similarity
            })
            
        except Exception as e:
            print(f"\n⚠️  Error processing {row['audio_id']}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    return results_df

# ============================================================================
# Step 6: Generate Report
# ============================================================================

def generate_report(results_df, output_dir):
    """Generate evaluation report and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_path = output_dir / "detailed_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Detailed results saved to: {results_path}")
    
    # Calculate summary statistics
    summary = {
        'total_samples': len(results_df),
        'mean_wer': results_df['wer'].mean(),
        'median_wer': results_df['wer'].median(),
        'std_wer': results_df['wer'].std(),
        'mean_cer': results_df['cer'].mean(),
        'median_cer': results_df['cer'].median(),
        'std_cer': results_df['cer'].std(),
        'mean_ser': results_df['ser'].mean(),
        'median_ser': results_df['ser'].median(),
        'std_ser': results_df['ser'].std(),
        'mean_semantic_similarity': results_df['semantic_similarity'].mean()
    }
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / "summary_statistics.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Samples:              {summary['total_samples']}")
    print(f"\nWord Error Rate (WER):")
    print(f"  Mean:                     {summary['mean_wer']:.4f} ({summary['mean_wer']*100:.2f}%)")
    print(f"  Median:                   {summary['median_wer']:.4f}")
    print(f"  Std Dev:                  {summary['std_wer']:.4f}")
    print(f"\nCharacter Error Rate (CER):")
    print(f"  Mean:                     {summary['mean_cer']:.4f} ({summary['mean_cer']*100:.2f}%)")
    print(f"  Median:                   {summary['median_cer']:.4f}")
    print(f"  Std Dev:                  {summary['std_cer']:.4f}")
    print(f"\nSemantic Error Rate (SER):")
    print(f"  Mean:                     {summary['mean_ser']:.4f}")
    print(f"  Median:                   {summary['median_ser']:.4f}")
    print(f"  Std Dev:                  {summary['std_ser']:.4f}")
    print(f"  Mean Similarity:          {summary['mean_semantic_similarity']:.4f}")
    print("="*60)
    
    # Find best and worst cases
    print("\n🏆 Best WER samples (top 5):")
    best_samples = results_df.nsmallest(5, 'wer')[['audio_id', 'wer', 'reference', 'hypothesis']]
    for idx, row in best_samples.iterrows():
        print(f"\n  {row['audio_id']} (WER: {row['wer']:.4f})")
        print(f"    Ref: {row['reference'][:80]}...")
        print(f"    Hyp: {row['hypothesis'][:80]}...")
    
    print("\n⚠️  Worst WER samples (top 5):")
    worst_samples = results_df.nlargest(5, 'wer')[['audio_id', 'wer', 'reference', 'hypothesis']]
    for idx, row in worst_samples.iterrows():
        print(f"\n  {row['audio_id']} (WER: {row['wer']:.4f})")
        print(f"    Ref: {row['reference'][:80]}...")
        print(f"    Hyp: {row['hypothesis'][:80]}...")
    
    print(f"\n💾 Summary statistics saved to: {summary_path}")

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main evaluation pipeline."""
    print("="*60)
    print("🎯 Kiswahili TTS Dataset Evaluation")
    print("="*60)
    
    # Step 1: Parse metadata
    metadata_df = parse_metadata(METADATA_PATH)
    
    # Step 2: Load models
    processor, asr_model = load_asr_model()
    semantic_model = load_semantic_model()
    
    # Step 3: Run evaluation
    # For testing, you can limit samples: max_samples=10
    results_df = evaluate_dataset(metadata_df, processor, asr_model, semantic_model, max_samples=None)
    
    # Step 4: Generate report
    generate_report(results_df, OUTPUT_DIR)
    
    print("\n✅ Evaluation complete!")

if __name__ == "__main__":
    main()
