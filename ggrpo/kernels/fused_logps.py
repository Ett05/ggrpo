import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    HAS_TRITON = False

if HAS_TRITON:
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

        target_token_id = tl.load(input_ids_ptr + row_id)
        target_token_logit = tl.load(logits_ptr + row_start + target_token_id)

        running_max = float('-inf')
        running_sum = 0.0

        for i in range(0, vocab_size, BLOCK_SIZE):
            cols = i + tl.arange(0, BLOCK_SIZE)
            mask = cols < vocab_size
            logits_block = tl.load(logits_ptr + row_start + cols, mask=mask, other=float('-inf'))
            
            m_local = tl.max(logits_block, axis=0)
            d_local = tl.sum(tl.exp(logits_block - m_local), axis=0)
            
            new_max = tl.maximum(running_max, m_local)
            running_sum = running_sum * tl.exp(running_max - new_max) + d_local * tl.exp(m_local - new_max)
            running_max = new_max

        log_prob = (target_token_logit - running_max) - tl.log(running_sum)
        tl.store(per_token_logps_ptr + row_id, log_prob)
        
        tl.store(lse_max_ptr + row_id, running_max)
        tl.store(lse_sum_ptr + row_id, running_sum)

    @triton.jit
    def get_per_token_logps_backward_kernel(
        d_per_token_logps_ptr,
        logits_ptr,
        input_ids_ptr,
        lse_max_ptr,
        lse_sum_ptr,
        d_logits_ptr,
        vocab_size,
        BLOCK_SIZE: tl.constexpr
    ):
        row_id = tl.program_id(0)
        row_start = row_id * vocab_size
        
        d_per_token_logps = tl.load(d_per_token_logps_ptr + row_id)
        target_token_id = tl.load(input_ids_ptr + row_id)
        lse_max = tl.load(lse_max_ptr + row_id)
        lse_sum = tl.load(lse_sum_ptr + row_id)
        
        for i in range(0, vocab_size, BLOCK_SIZE):
            cols = i + tl.arange(0, BLOCK_SIZE)
            mask = cols < vocab_size
            
            logits_block = tl.load(logits_ptr + row_start + cols, mask=mask, other=float('-inf'))
            p_block = tl.exp(logits_block - lse_max) / lse_sum
            is_target = tl.where(cols == target_token_id, 1.0, 0.0)
            dx_block = d_per_token_logps * (is_target - p_block)
            
            tl.store(d_logits_ptr + row_start + cols, dx_block, mask=mask)

class FusedGetPerTokenLogps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, input_ids):
        logits = logits.contiguous()
        input_ids = input_ids.contiguous()
        orig_shape = logits.shape
        logits_flat = logits.view(-1, orig_shape[-1])
        input_ids_flat = input_ids.view(-1)
        
        if logits.device.type != "cuda" or not HAS_TRITON:
            log_probs = logits_flat.log_softmax(dim=-1)
            per_token_logps = torch.gather(log_probs, dim=-1, index=input_ids_flat.unsqueeze(-1)).squeeze(-1)
            ctx.save_for_backward(logits_flat, input_ids_flat)
            ctx.orig_shape = orig_shape
            ctx.is_cpu_fallback = True
            return per_token_logps.view(orig_shape[0], orig_shape[1])

        ctx.is_cpu_fallback = False
        num_rows, vocab_size = logits_flat.shape
        
        per_token_logps = torch.empty(num_rows, dtype=logits.dtype, device=logits.device)
        lse_max = torch.empty(num_rows, dtype=torch.float32, device=logits.device)
        lse_sum = torch.empty(num_rows, dtype=torch.float32, device=logits.device)
        
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
        
        ctx.save_for_backward(logits_flat, input_ids_flat, lse_max, lse_sum)
        ctx.orig_shape = orig_shape
        
        return per_token_logps.view(orig_shape[0], orig_shape[1])

    @staticmethod
    def backward(ctx, grad_output):
        orig_shape = ctx.orig_shape
        grad_output = grad_output.contiguous()
        grad_output_flat = grad_output.view(-1)
        
        if getattr(ctx, "is_cpu_fallback", False):
            logits_flat, input_ids_flat = ctx.saved_tensors
            probs = logits_flat.softmax(dim=-1)
            is_target = torch.zeros_like(probs).scatter_(1, input_ids_flat.unsqueeze(-1), 1.0)
            d_logits_flat = grad_output_flat.unsqueeze(-1) * (is_target - probs)
            return d_logits_flat.view(*orig_shape), None

        logits_flat, input_ids_flat, lse_max, lse_sum = ctx.saved_tensors
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
        
        return d_logits.view(*orig_shape), None

def get_per_token_logps(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Computes per-token log probabilities using a fused Triton kernel.
    Memory efficient (does not materialize the full logits/log-softmax tensor in VRAM).
    """
    return FusedGetPerTokenLogps.apply(logits, input_ids)

