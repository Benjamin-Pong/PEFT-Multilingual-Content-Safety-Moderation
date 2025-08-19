# Overview

This project is the first comparative study of alternative parameter-efficient finetuning methods to Low Rank Adaptation (LoRA), notably Adaptive Low-Rank Adaptation (AdaLoRA) and Weight-Decomposition Low-Rank Adaptation (DoRA), for downstream multilingual content moderation tasks which is a multi-step classification task. Preliminary experimental
results on the PolyGuard datasets (Kumar et al.,2025) demonstrate the potential for these alternative parameter-efficient fine-tuning methods in outperforming LoRA and full-finetuning for a subset of content moderation tasks. This study also reveals the challenges of decoder-only models for classification tasks by incorporating In-Context-Learning and Constraint Decoding techniques. More rigorous experiments need to be done to further support the claim that AdaLORA and DORA can both surppass LoRA and full-finetuning for cross-lingual content moderation tasks.

# Dataset

PolyGuard dataset can be retrieved from Huggingface

Train split: ToxicityPrompts/PolyGuardMix
Test split: ToxicityPrompts/PolyGuardPrompts

Cite: Kumar, Priyanshu, et al. "Polyguard: A multilingual safety moderation tool for 17 languages." arXiv preprint arXiv:2504.04377 (2025).


