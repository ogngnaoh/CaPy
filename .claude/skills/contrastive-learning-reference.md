---
name: contrastive-learning-reference
description: InfoNCE, temperature, CLIP-style training reference. Load when working on losses or training.
---

# Contrastive Learning Reference

## InfoNCE (NT-Xent)
S[i,j] = cos(z_a[i], z_b[j]) / temperature
Loss = CrossEntropy(S, [0,1,...,N-1]) averaged both directions.
Perfect matching: loss = log(N). This is the MINIMUM, not zero.

## Temperature
Below 0.05: unstable gradients. Above 0.5: too smooth, weak signal.
CLIP uses learnable temperature, initialized at 0.07.
Always clamp: temp.clamp(min=0.01, max=10.0).

## Numerical Stability
Subtract max logits before softmax: logits -= logits.max(dim=-1, keepdim=True).values

## Multi-Modal CLIP Pattern
K modalities produce K*(K-1)/2 pairwise losses.
CaPy: 3 modalities = 3 pairwise losses, weighted by lambda.

## Batch Size
More negatives = better contrastive signal. Minimum useful: 64. Sweet spot for ~2K dataset: 128-256.
