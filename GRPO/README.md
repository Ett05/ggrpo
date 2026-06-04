# GRPO from Scratch

Implementing Group Relative Policy Optimization (GRPO) from scratch in PyTorch to fine-tune Qwen2.5-1.5B on code generation tasks.

## What this is

[1-2 sentences: what GRPO is and what you built]

## How it works

### Training loop
[Explain in plain English: generate G outputs, score them, normalize rewards, compute loss, backprop]

### Reward function
[Explain subprocess sandboxing, pass/fail, edge cases handled]

### Loss computation
[Explain per-token log prob * advantage, KL penalty, clipping]

## Results

[Insert reward curve plot]

Average reward improved from X to Y over 50 epochs on a 10-problem coding dataset.

## What's missing / limitations

- Small dataset (10 toy problems)
- [anything else honest]

## Future work

- Replace naive `get_per_token_logps` with a fused Triton kernel
- Scale to larger coding datasets (HumanEval, MBPP)
- Multi-GPU training with DeepSpeed

## Requirements


## Usage

## File Structure 
GRPO/
dataset.py    # problems and prompts
reward.py     # sandboxed subprocess runner
train.py      # main training loop