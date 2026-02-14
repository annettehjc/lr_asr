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

---

## 3. Responsible Practice

> *"Bias analysis as standard scientific work, transparent data documentation, and privacy guardrails."*

### Current Approach
- Limited data documentation
- No bias analysis
- No explicit privacy considerations


## References

- SPEAKABLE 2026 Workshop: https://speakable-2026.github.io/
- PEFT Library: https://github.com/huggingface/peft
- HuggingFace Model Cards: https://huggingface.co/docs/hub/model-cards
- Data Cards: https://arxiv.org/abs/2204.01075
