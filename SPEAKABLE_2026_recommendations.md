# Aligning Swahili ASR Experiment with SPEAKABLE 2026 Core Strands

This document outlines recommendations for developing our Swahili wav2vec2 fine-tuning experiment in accordance with the three core strands of SPEAKABLE 2026.

## Current Experiment Summary

- **Model**: `facebook/wav2vec2-large-xlsr-53` fine-tuned for Swahili ASR
- **Dataset**: 120,100 samples (~138 hours), 90/10 train/test split
- **Training**: Full fine-tuning, batch size 8, LR 1e-4, 1 epoch
- **Result**: WER ~24.53%
- **HuggingFace**: `jkhyjkhy/wav2vec2-large-xlsr-sw-ASR`

---

## 1. Efficient Adaptation

> *"Parameter-efficient methods, multilingual transfer, knowledge distillation, and edge-constrained inference for low-resource speech tasks."*

### Current Approach
- Full fine-tuning of all model parameters
- Only feature extractor is frozen

### Recommended Improvements

| Method | Description | Benefits |
|--------|-------------|----------|
| **LoRA/QLoRA** | Train only low-rank adapter matrices | 90%+ reduction in trainable parameters |
| **Adapter Tuning** | Insert small adapter modules in transformer blocks | Modular, reusable components |
| **Knowledge Distillation** | Transfer knowledge from large to smaller model | Enable edge deployment |
| **Multilingual Transfer** | Leverage pre-training on related Bantu languages | Improved low-resource performance |

### Implementation Example (LoRA)

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~0.5% of original parameters
```

### Metrics to Report
- Number of trainable parameters vs. total parameters
- Training time comparison
- Memory (VRAM) usage
- Inference latency on edge devices

---

## 2. Meaningful Evaluation

> *"Task-appropriate metrics, calibration analysis, and slice-aware reporting by accent, dialect, and channel."*

### Current Approach
- Only Word Error Rate (WER) is measured
- No demographic or recording condition analysis

### Recommended Improvements

| Metric | Description | Relevance to Swahili |
|--------|-------------|----------------------|
| **CER** | Character Error Rate | Captures morphological accuracy |
| **MER** | Match Error Rate | Word boundary precision |
| **WIL** | Word Information Lost | Information-theoretic measure |
| **Calibration** | Confidence vs. accuracy alignment | Reliability assessment |

### Slice-Aware Reporting

Analyze performance across different speaker populations and recording conditions:

```python
import pandas as pd
from jiwer import wer

def evaluate_by_slice(predictions, references, metadata, slice_column):
    """Compute WER for each demographic/condition slice."""
    df = pd.DataFrame({
        'pred': predictions,
        'ref': references,
        'slice': metadata[slice_column]
    })

    results = {}
    for slice_value in df['slice'].unique():
        subset = df[df['slice'] == slice_value]
        slice_wer = wer(
            list(subset['ref']),
            list(subset['pred'])
        )
        results[slice_value] = {
            'wer': slice_wer,
            'n_samples': len(subset)
        }

    return results

# Example slices to analyze:
# - Gender: male / female
# - Age group: young / adult / elderly
# - Dialect/Region: coastal / inland / urban
# - Recording channel: studio / field / phone
```

### Calibration Analysis

```python
import numpy as np

def compute_calibration(confidences, accuracies, n_bins=10):
    """Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_accuracy = accuracies[mask].mean()
            bin_confidence = confidences[mask].mean()
            ece += mask.sum() * abs(bin_accuracy - bin_confidence)

    return ece / len(confidences)
```

### Recommended Evaluation Table Format

| Slice | N Samples | WER (%) | CER (%) | 95% CI |
|-------|-----------|---------|---------|--------|
| Overall | 12,010 | 24.53 | - | - |
| Male | - | - | - | - |
| Female | - | - | - | - |
| Coastal dialect | - | - | - | - |
| Inland dialect | - | - | - | - |

---

## 3. Responsible Practice

> *"Bias analysis as standard scientific work, transparent data documentation, and privacy guardrails."*

### Current Approach
- Limited data documentation
- No bias analysis
- No explicit privacy considerations

### Recommended Improvements

#### Data Card

Create a comprehensive data card documenting:

```markdown
## Dataset: Swahili ASR Corpus

### Overview
- **Size**: 120,100 audio samples (~138 hours)
- **Language**: Swahili (sw)
- **Source**: [Document original source - Common Voice, etc.]

### Speaker Demographics
- Gender distribution: X% male, Y% female
- Age range: X-Y years
- Dialect regions represented: [List regions]
- Number of unique speakers: N

### Recording Conditions
- Sampling rate: 16kHz
- Recording environments: [studio/field/etc.]
- Audio quality: [Description]

### Data Collection
- Collection methodology: [Description]
- Consent process: [Description]
- Time period: [Dates]

### Known Limitations
- [List any known biases or gaps]
- [Underrepresented demographics]

### Privacy Considerations
- Speaker anonymization: [Yes/No, method]
- Sensitive content filtering: [Description]
```

#### Model Card

Add to HuggingFace repository:

```markdown
## Model: wav2vec2-large-xlsr-sw-ASR

### Intended Use
- Primary: Swahili automatic speech recognition
- Users: Researchers, developers building Swahili applications

### Out-of-Scope Use
- Not intended for: [surveillance, speaker identification, etc.]
- May not perform well on: [specific dialects, noisy conditions, etc.]

### Bias Analysis
| Demographic | WER | Relative Performance |
|-------------|-----|---------------------|
| [Group A] | X% | Baseline |
| [Group B] | Y% | +Z% relative |

### Limitations
- [List known limitations]
- [Performance gaps across demographics]

### Ethical Considerations
- [Privacy measures taken]
- [Potential misuse concerns]
```

#### Bias Analysis Framework

```python
def compute_bias_metrics(slice_results):
    """Compute fairness metrics across demographic slices."""
    wers = [v['wer'] for v in slice_results.values()]

    metrics = {
        'max_wer_gap': max(wers) - min(wers),
        'wer_ratio': max(wers) / min(wers) if min(wers) > 0 else float('inf'),
        'std_wer': np.std(wers),
        'worst_performing_slice': max(slice_results, key=lambda k: slice_results[k]['wer'])
    }

    return metrics
```

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
- [ ] Add CER metric to evaluation
- [ ] Create basic Data Card
- [ ] Create basic Model Card for HuggingFace

### Phase 2: Efficient Adaptation (2-3 weeks)
- [ ] Implement LoRA fine-tuning
- [ ] Compare parameter efficiency vs. full fine-tuning
- [ ] Measure inference latency

### Phase 3: Comprehensive Evaluation (2-3 weeks)
- [ ] Collect/annotate metadata for demographic slices
- [ ] Implement slice-aware evaluation pipeline
- [ ] Compute calibration metrics

### Phase 4: Responsible AI (1-2 weeks)
- [ ] Conduct bias analysis across available slices
- [ ] Complete comprehensive Data Card
- [ ] Complete comprehensive Model Card
- [ ] Document limitations and ethical considerations

---

## References

- SPEAKABLE 2026 Workshop: https://speakable-2026.github.io/
- PEFT Library: https://github.com/huggingface/peft
- HuggingFace Model Cards: https://huggingface.co/docs/hub/model-cards
- Data Cards: https://arxiv.org/abs/2204.01075
