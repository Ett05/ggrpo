import copy
import torch
from typing import Callable, Any, Optional
from ggrpo.kernels import get_per_token_logps
from ggrpo.history import GRPOHistory

def train(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: list[dict[str, Any]],
    reward_fn: Callable[[list[Any]], list[float]],
    ref_model: Optional[torch.nn.Module] = None,
    num_epochs: int = 10,
    group_size: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 0.9,
    lr: float = 1e-5,
    beta: float = 0.04,
    system_prompt: str = "",
    device: str = "cuda"
) -> GRPOHistory:
    """
    Fine-tunes a causal language model using Group Relative Policy Optimization (GRPO) 
    powered by fused Triton kernels for high memory efficiency.

    Args:
        model: PyTorch causal language model to fine-tune.
        tokenizer: Model tokenizer.
        dataset: List of dataset dicts containing 'prompt' and 'test' keys.
        reward_fn: Function taking list of [generated_text, test_case] and returning float rewards.
        ref_model: Optional reference model for KL divergence constraint (cloned from model if None).
        num_epochs: Number of training epochs.
        group_size: Number of candidate completions sampled per prompt (G).
        max_new_tokens: Max tokens to generate per completion.
        temperature: Sampling temperature.
        lr: Optimizer learning rate.
        beta: KL divergence penalty coefficient.
        system_prompt: System prompt prepended to prompts.
        device: PyTorch device ('cuda').

    Returns:
        GRPOHistory object containing epoch metrics with .plot() method.
    """
    if ref_model is None:
        ref_model = copy.deepcopy(model)
        
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    history = GRPOHistory()

    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_losses = []

        for item in dataset:
            prompt = system_prompt + item["prompt"]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=group_size
                )
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            outputs = outputs.clone()
            prompt_length = inputs["input_ids"].shape[1]
            generated = outputs[:, prompt_length:]

            # Evaluate rewards
            answers = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]
            test_cases = [[answers[i], item["test"]] for i in range(len(answers))]
            
            raw_rewards = reward_fn(test_cases)
            rewards = torch.tensor(raw_rewards, dtype=torch.float32, device=device)
            
            # Compute group-relative advantages
            average = torch.mean(rewards)
            std = torch.std(rewards)
            advantages = (rewards - average) / (std + 1e-8)

            # Compute policy logps using fused Triton kernel
            forward_pass = model(outputs)
            logits = forward_pass.logits[:, :-1, :].contiguous()
            input_ids = outputs[:, 1:].contiguous()
            
            per_token_logps = get_per_token_logps(logits, input_ids)
            per_token_logps = per_token_logps[:, prompt_length - 1:]

            # Compute reference model logps using fused Triton kernel
            with torch.no_grad():
                ref_logits = ref_model(outputs).logits[:, :-1, :].contiguous()
                ref_per_token_logps = get_per_token_logps(ref_logits, input_ids)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

            # Compute GRPO loss and KL divergence
            per_token_losses = -per_token_logps * advantages.unsqueeze(1)
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            
            total_loss = (per_token_losses + beta * per_token_kl).mean()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_rewards.extend(raw_rewards)
            epoch_losses.append(total_loss.item())

        avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        
        history.epoch_rewards.append(avg_reward)
        history.epoch_losses.append(avg_loss)
        
        print(f"Epoch {epoch + 1}/{num_epochs} - Avg Reward: {avg_reward:.3f} - Loss: {avg_loss:.4f}")

    return history
