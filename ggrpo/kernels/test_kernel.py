import torch
import time
from ggrpo.kernels.fused_logps import FusedGetPerTokenLogps, get_per_token_logps

# PyTorch eager reference implementation for validation
def get_per_token_logps_ref(logits, input_ids):
    log_probs = logits.log_softmax(dim=-1)
    return torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running tests on device: {device}")
    
    # Test 1: Autograd gradcheck
    print("Test 1: Running torch.autograd.gradcheck...")
    logits_small = torch.randn(2, 4, 32, dtype=torch.float64, device=device, requires_grad=True)
    input_ids_small = torch.randint(0, 32, (2, 4), device=device)
    
    test_grad = torch.autograd.gradcheck(
        FusedGetPerTokenLogps.apply, 
        (logits_small, input_ids_small), 
        eps=1e-6, 
        atol=1e-4
    )
    print(f"-> Gradcheck validation: {test_grad}\n")

    # Test 2: Correctness and Benchmark (Forward & Backward)
    if device == "cuda":
        batch_size = 4
        seq_len = 1024
        vocab_size = 32002
        
        print(f"Test 2: Benchmarking size [B={batch_size}, S={seq_len}, V={vocab_size}]...")
        logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        torch.cuda.synchronize()
        start = time.time()
        ref_out = get_per_token_logps_ref(logits, input_ids)
        ref_loss = ref_out.sum()
        ref_loss.backward()
        ref_grad = logits.grad.clone()
        torch.cuda.synchronize()
        ref_time = time.time() - start
        
        logits.grad.zero_()
        
        torch.cuda.synchronize()
        start = time.time()
        triton_out = FusedGetPerTokenLogps.apply(logits, input_ids)
        triton_loss = triton_out.sum()
        triton_loss.backward()
        triton_grad = logits.grad.clone()
        torch.cuda.synchronize()
        triton_time = time.time() - start

        forward_correct = torch.allclose(ref_out, triton_out, atol=1e-5, rtol=1e-5)
        backward_correct = torch.allclose(ref_grad, triton_grad, atol=1e-5, rtol=1e-5)
        
        print(f"-> Forward correctness: {forward_correct}")
        print(f"-> Backward correctness: {backward_correct}")
        print(f"-> PyTorch execution time: {ref_time * 1000:.2f} ms")
        print(f"-> Triton execution time: {triton_time * 1000:.2f} ms")
        print(f"-> Speedup factor: {ref_time / triton_time:.2f}x")


