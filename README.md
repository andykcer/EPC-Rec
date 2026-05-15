# EPC-Rec

**EPC-Rec** is a course recommendation framework designed to mitigate popularity bias while improving recommendation accuracy. It integrates **Epistemic-Guided Sampling**, **Popularity-Aware Mixture Attention**, and a **Calibrated Objective** to enhance learner-course matching and improve long-tail course recommendation.

## Overview

Online learning platforms provide learners with access to a large number of courses, but the rapid growth of available resources also makes course discovery increasingly challenging. Existing recommendation methods often optimize ranking accuracy based on observed interactions, which may over-emphasize popular courses and under-represent long-tail courses.

To address this issue, EPC-Rec aims to improve course recommendation by considering both learner-course relevance and popularity-aware exposure. Specifically, it incorporates knowledge-background signals and popularity-aware modeling to better match learners with suitable courses.

## Main Components

EPC-Rec consists of three key components:

1. **Epistemic-Guided Sampling**

   This module constructs informative negative samples based on epistemic uncertainty. Instead of randomly selecting negative courses, it identifies harder and more meaningful negative samples to improve the model's discriminative ability.

2. **Popularity-Aware Mixture Attention**

   This module captures learner-course matching patterns across different course popularity groups. By introducing popularity-aware attention, EPC-Rec reduces excessive reliance on highly popular courses and improves representation learning for long-tail courses.

3. **Calibrated Objective**

   This objective regularizes the recommendation process by encouraging a better balance between ranking accuracy and popularity bias mitigation. It helps improve long-tail exposure while maintaining competitive recommendation performance.

## Features

- Course recommendation with popularity-aware modeling
- Epistemic-guided negative sampling
- Long-tail course recommendation enhancement
- Popularity bias mitigation
- Support for top-K evaluation
- Evaluation with accuracy and long-tail exposure metrics

## Evaluation Metrics

The model can be evaluated using both accuracy-oriented and popularity-aware metrics, including:

- `Recall@K`
- `NDCG@K`
- `MRR@K`
- `ARP@K`
- `HRcold@K`
- `LTE@K`

where `ARP@K` measures the average popularity of recommended courses, `HRcold@K` evaluates the hit ratio on cold courses, and `LTE@K` measures long-tail exposure in the recommendation list.

## Project Structure

```text
EPC-Rec/
├── data/                 # Dataset files
├── models/               # Model implementation
├── utils/                # Utility functions
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── config.py             # Hyperparameter configuration
└── README.md             # Project documentation
