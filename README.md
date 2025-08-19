# Overview

This project is the first comparative study of alternative parameter-efficient finetuning methods to Low Rank Adaptation (LoRA), notably Adaptive Low-Rank Adaptation (AdaLoRA) and Weight-Decomposition Low-Rank Adaptation (DoRA), for downstream multilingual content moderation tasks which is a multi-step classification task. Preliminary proof-of-concept-experimental results on the PolyGuard datasets (Kumar et al.,2025) demonstrate the potential for these alternative parameter-efficient fine-tuning methods in outperforming LoRA and full-finetuning for a subset of content moderation tasks. This study also reveals the challenges of decoder-only models for classification tasks by incorporating In-Context-Learning and Constraint Decoding techniques. More rigorous experiments need to be done to further support the claim that AdaLORA and DORA can both surppass LoRA and full-finetuning for cross-lingual content moderation tasks.

# Dataset

PolyGuard dataset can be retrieved from Huggingface

Train split: ToxicityPrompts/PolyGuardMix
Test split: ToxicityPrompts/PolyGuardPrompts

Reference: Kumar, Priyanshu, et al. "Polyguard: A multilingual safety moderation tool for 17 languages." arXiv preprint arXiv:2504.04377 (2025).

# Methods

## Fine-tuning

Experiments include full-parameter fine-tuning, DoRA and AdaLoRA fine-tuning using the PolyGuard datasets on Qwen2.5-7B-Instruct. They contain the model set-up and also the hyperparameters that were employed. Performance of these individual models were compared against polyGuard (which was finetuned using LoRA).

Individual scripts for these different modes of fine-tuning modes were adapted from Unsloth and can be found in the repository as follow:
1. Full-parameter fine-tuning: /qwen_full_finetuning.py
2. AdaLoRA fine-tuning: /qwen_adalora_finetuning.py
3. DoRA finetuning: /qwen_dora_finetuning.py

## In-context Learning


## Constrained Decoding using Pydantic tools

Due to the verbosity of LLMs, they may not be the best use case for classification tasks. LLMs tend to predict a lot more text eventhough it has been instructed to predict a single label/word. To circumvent this issue, I implemented a class using Pydantic to constrain the output of the LLM at inference time, such that it is able to produce key label predictions for prompt harms, the response harm, prompt violation label(s) and response violation label(s).


# Evaluation

Evaluation script can be found at /evaluation.py

Evaluation is done on a subset of the PolyGuardPrompts dataset due to computational constraints.  2706 randomly sampled instances (with a random seed of 42) were used for evaluation across all four models. 

Since this task comprises a binary classification task and a multilabel classification task, two metrics are used, following Kumar et al., 2025.

As prompt harm labels and response harm labels are treated as binary classification tasks, F1 score is
used for the positive label (unsafe for harmfulness
and yes for response refusal).

On the other hand, determining prompt and response violations is a multilabel classification task, Jaccard Similary was used to compare the predicted violations with ground truth violations. The Jaccard Similarity quantifies how similar two datasets are by comparing their shared elements to their combined elements. It is calculated by dividing the number of common items by the total number of unique items in both sets. The result is a value between 0 (completely dissimilar) and 1 (exactly the same). The aggregate Jaccard similarity is computed as the average of all similarity scores.




# Results


# Limitations and Future Work

