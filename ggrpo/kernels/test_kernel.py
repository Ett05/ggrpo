import torch
import triton
import triton.language as tl
import time

@triton.jit
def get_per_token_logps_forward_kernel(
    logits_ptr,
    input_ids_ptr,
    per_token_logps_ptr,
    vocab_size,
    lse_max_ptr,
    lse_sum_ptr,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    row_start = row_id * vocab_size

    # Load target token ID for this row
    target_token_id = tl.load(input_ids_ptr + row_id)
    # Load corresponding logit value
    target_token_logit = tl.load(logits_ptr + row_start + target_token_id)

    running_max = float('-inf')
    running_sum = 0.0

    # Online softmax loop
    for i in range(0, vocab_size, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        logits_block = tl.load(logits_ptr + row_start + cols, mask=mask, other=float('-inf'))
        
        m_local = tl.max(logits_block, axis=0)
        d_local = tl.sum(tl.exp(logits_block - m_local), axis=0)
        
        new_max = tl.maximum(running_max, m_local)
        # Rescale the running sum and local sum to the new maximum
        running_sum = running_sum * tl.exp(running_max - new_max) + d_local * tl.exp(m_local - new_max)
        running_max = new_max

    # Compute final log-probability
    log_prob = (target_token_logit - running_max) - tl.log(running_sum)
    # Store output log-prob
    tl.store(per_token_logps_ptr + row_id, log_prob)
    
    # Store maximums and sums for the backward pass (to avoid recomputing them)
    tl.store(lse_max_ptr + row_id, running_max)
    tl.store(lse_sum_ptr + row_id, running_sum)

@triton.jit
def get_per_token_logps_backward_kernel(
    d_per_token_logps_ptr, # dy
    logits_ptr,            # x
    input_ids_ptr,         # target token indices
    lse_max_ptr,           # saved m
    lse_sum_ptr,           # saved d
    d_logits_ptr,          # Output: dx (gradients)
    vocab_size,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    row_start = row_id * vocab_size
    
    # Load row-level constants from HBM
    d_per_token_logps = tl.load(d_per_token_logps_ptr + row_id)
    target_token_id = tl.load(input_ids_ptr + row_id)
    lse_max = tl.load(lse_max_ptr + row_id)
    lse_sum = tl.load(lse_sum_ptr + row_id)
    
    for i in range(0, vocab_size, BLOCK_SIZE):
        cols = i + tl.arange(0, BLOCK_SIZE)
        mask = cols < vocab_size
        
        # Load logits block from HBM
        logits_block = tl.load(logits_ptr + row_start + cols, mask=mask, other=float('-inf'))
        
        # Recompute softmax probabilities on-the-fly in Registers
        p_block = tl.exp(logits_block - lse_max) / lse_sum
        
        # Create target mask (1.0 at target index, 0.0 elsewhere)
        is_target = tl.where(cols == target_token_id, 1.0, 0.0)
        
        # Compute gradient for this block
        dx_block = d_per_token_logps * (is_target - p_block)
        
        # Store gradient block back to HBM
        tl.store(d_logits_ptr + row_start + cols, dx_block, mask=mask)

class FusedGetPerTokenLogps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, input_ids):
        # Handle 3D logits of shape [batch, seq_len, vocab_size]
        orig_shape = logits.shape
        logits_flat = logits.view(-1, orig_shape[-1])
        input_ids_flat = input_ids.view(-1)
        
        num_rows, vocab_size = logits_flat.shape
        
        # Allocate outputs in HBM
        per_token_logps = torch.empty(num_rows, dtype=logits.dtype, device=logits.device)
        lse_max = torch.empty(num_rows, dtype=torch.float32, device=logits.device)
        lse_sum = torch.empty(num_rows, dtype=torch.float32, device=logits.device)
        
        # Launcher configuration
        BLOCK_SIZE = 1024
        grid = (num_rows,)
        
        get_per_token_logps_forward_kernel[grid](
            logits_flat,
            input_ids_flat,
            per_token_logps,
            vocab_size,
            lse_max,
            lse_sum,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Save tensors for the backward pass
        ctx.save_for_backward(logits_flat, input_ids_flat, lse_max, lse_sum)
        ctx.orig_shape = orig_shape
        
        return per_token_logps.view(orig_shape[0], orig_shape[1])

    @staticmethod
    def backward(ctx, grad_output):
        logits_flat, input_ids_flat, lse_max, lse_sum = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        
        # Flatten incoming gradient [batch, seq_len] -> [num_rows]
        grad_output_flat = grad_output.reshape(-1)
        
        num_rows, vocab_size = logits_flat.shape
        d_logits = torch.empty_like(logits_flat)
        
        BLOCK_SIZE = 1024
        grid = (num_rows,)
        
        get_per_token_logps_backward_kernel[grid](
            grad_output_flat,
            logits_flat,
            input_ids_flat,
            lse_max,
            lse_sum,
            d_logits,
            vocab_size,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        # Return gradients matching forward args: (logits, input_ids)
        # input_ids doesn't have gradients, so we return None for it.
        return d_logits.view(*orig_shape), None

# PyTorch eager reference implementation for validation
def get_per_token_logps_ref(logits, input_ids):
    log_probs = logits.log_softmax(dim=-1)
    return torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

if __name__ == "__main__":
    device = "cuda"
    
    # Test 1: Autograd gradcheck (Gold standard test for custom gradients)
    print("Test 1: Running torch.autograd.gradcheck...")
    # Keep sizes small for numerical gradcheck since it builds a full Jacobian matrix
    logits_small = torch.randn(2, 4, 32, dtype=torch.float64, device=device, requires_grad=True)
    input_ids_small = torch.randint(0, 32, (2, 4), device=device)
    
    # gradcheck returns True if analytical and numerical gradients match
    test_grad = torch.autograd.gradcheck(
        FusedGetPerTokenLogps.apply, 
        (logits_small, input_ids_small), 
        eps=1e-6, 
        atol=1e-4
    )
    print(f"-> Gradcheck validation: {test_grad}\n")

    # Test 2: Correctness and Benchmark (Forward & Backward)
    batch_size = 4
    seq_len = 1024
    vocab_size = 32002
    
    print(f"Test 2: Benchmarking size [B={batch_size}, S={seq_len}, V={vocab_size}]...")
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Reference forward & backward
    torch.cuda.synchronize()
    start = time.time()
    ref_out = get_per_token_logps_ref(logits, input_ids)
    ref_loss = ref_out.sum()
    ref_loss.backward()
    ref_grad = logits.grad.clone()
    torch.cuda.synchronize()
    ref_time = time.time() - start
    
    # Reset grad
    logits.grad.zero_()
    
    # Triton forward & backward
    torch.cuda.synchronize()
    start = time.time()
    triton_out = FusedGetPerTokenLogps.apply(logits, input_ids)
    triton_loss = triton_out.sum()
    triton_loss.backward()
    triton_grad = logits.grad.clone()
    torch.cuda.synchronize()
    triton_time = time.time() - start

    # Correctness Assertions
    forward_correct = torch.allclose(ref_out, triton_out, atol=1e-5, rtol=1e-5)
    backward_correct = torch.allclose(ref_grad, triton_grad, atol=1e-5, rtol=1e-5)
    
    print(f"-> Forward correctness: {forward_correct}")
    print(f"-> Backward correctness: {backward_correct}")
    print(f"-> PyTorch execution time: {ref_time * 1000:.2f} ms")
    print(f"-> Triton execution time: {triton_time * 1000:.2f} ms")
    print(f"-> Speedup factor: {ref_time / triton_time:.2f}x")
