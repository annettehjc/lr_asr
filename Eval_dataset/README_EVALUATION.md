# Quick Start Guide

## Running the Evaluation

The evaluation script is ready to use. Run it with:

```bash
# Activate virtual environment
source venv/bin/activate

# Run evaluation (full dataset)
python Eval_dataset/evaluate_tts_dataset.py

# Or run on limited samples for testing (edit script, set max_samples=10)
```

## First Run Notes

**The first run will take longer** because it needs to:
1. Download Wav2Vec2 model from HuggingFace (~1-2GB)
2. Download sentence transformer model (~100MB)
3. Process all 7,000+ audio files

Subsequent runs will be faster as models are cached.

## Output

Results will be saved to `Eval_dataset/results/`:
- `detailed_results.csv` - Per-sample metrics
- `summary_statistics.csv` - Overall statistics

## Metrics Explained

- **WER (Word Error Rate)**: Percentage of word-level errors
- **CER (Character Error Rate)**: Percentage of character-level errors  
- **SER (Semantic Error Rate)**: 1 - cosine similarity of sentence embeddings
  - Lower is better (0 = perfect semantic match)
  - Measures meaning preservation, not just word accuracy
