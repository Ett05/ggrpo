import torch
import re
from dataset import problems, num_return_sequences, system
from reward import reward_function
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype=torch.bfloat16).to("cuda")
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.bfloat16).to("cuda")
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False
def extract_code(text):
    # try markdown code block first
    match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fall back to finding def ... to end of text
    match = re.search(r'(def \w+.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""
def get_per_token_logps(logits, input_ids):
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)
num_epochs = 50
n = 50
learning_rate = 1e-5
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)#training loop
all_epoch_rewards = []
clip_param = 0.2
old_logps = []
old_logps_per_sequence = []
for epoch in range(num_epochs):
    epoch_rewards = []
    for problem in problems:
        test_cases = []
        prompt = system + problem["prompt"]
        # print(prompt)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.9, num_return_sequences=num_return_sequences)
        torch.cuda.empty_cache()
        # decode only the generated part, not the prompt
        outputs = outputs.clone()
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        answers = [extract_code(tokenizer.decode(g, skip_special_tokens=True)) for g in generated]
        if epoch % n == 0: 
          print(answers[0])
          torch.save(model.state_dict(), 'checkpoint.pt')
        # print(prompt, answers)
        test_cases = [[answers[i], problem["test"]] for i in range(len(answers))]

        rewards = reward_function(test_cases)
        rewards = torch.tensor(rewards, dtype=torch.float32, device="cuda")
        average = torch.mean(rewards)
        advantages = rewards - average
        #loss loop
        losses = []
        prompt_length = inputs["input_ids"].shape[1]
        forward_pass = model(outputs)
        logits = forward_pass.logits[:, :-1, :]
        input_ids = outputs[:, 1:]
        per_token_logps = get_per_token_logps(logits, input_ids)
        per_token_logps = per_token_logps[:, prompt_length-1:]
        old_logps_per_sequence.append(per_token_logps.detach())
        # if len(old_logps) > i:  # after first epoch
        #     ratio = torch.exp(per_token_logps - old_logps[i])
        #     clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        #     per_token_losses = -torch.min(ratio * (rewards[i] - average), clipped_ratio * (rewards[i] - average))
        # else:
        #     per_token_losses = per_token_logps * -1 * (rewards[i] - average)

        with torch.no_grad():
            ref_logits = ref_model(outputs).logits[:, :-1, :]
            ref_per_token_logps = get_per_token_logps(ref_logits, input_ids)
            ref_per_token_logps = ref_per_token_logps[:, prompt_length-1:]
        
        per_token_losses = -per_token_logps * advantages.unsqueeze(1)


        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        beta = 0.04
        # per_token_losses = per_token_losses - beta * per_token_kl
        # losses.append((per_token_losses.sum()))
        total_loss = (per_token_losses + beta * per_token_kl).mean()


        # print(f"problem: {problem['prompt'][:30]}... rewards: {rewards.tolist()}, loss: {sum(losses).item():.4f}")
        optimizer.zero_grad()
        total_loss.backward()
        old_logps = old_logps_per_sequence
        old_logps_per_sequence = []    
        optimizer.step()
        epoch_rewards.extend(rewards.tolist())
    print(f"epoch {epoch}, avg reward: {sum(epoch_rewards)/len(epoch_rewards):.3f}")
    all_epoch_rewards.append(sum(epoch_rewards)/len(epoch_rewards))
