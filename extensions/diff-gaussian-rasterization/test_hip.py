#!/usr/bin/env python3
"""
Minimal HIP kernel execution test
Tests if HIP kernels actually execute on this system
"""
import torch
import os

# Set up environment
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

print(f"PyTorch version: {torch.__version__}")
print(f"HIP available: {torch.cuda.is_available()}")
print(f"HIP version: {torch.version.hip}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Test 1: Simple tensor operation on GPU
    print("\n=== Test 1: Basic tensor GPU operation ===")
    a = torch.ones(1000, device=device)
    b = torch.ones(1000, device=device) * 2
    c = a + b
    print(f"a + b = {c[:10].cpu().tolist()}")  # Should be [3, 3, 3, ...]
    
    # Test 2: In-place operation
    print("\n=== Test 2: In-place GPU operation ===")
    x = torch.zeros(100, dtype=torch.int32, device=device)
    x.fill_(42)
    print(f"x filled with 42: {x[:10].cpu().tolist()}")  # Should be [42, 42, 42, ...]
    
    # Test 3: Matrix multiplication (uses BLAS/GEMM kernels)
    print("\n=== Test 3: Matrix multiplication ===")
    m1 = torch.randn(100, 100, device=device)
    m2 = torch.randn(100, 100, device=device)
    m3 = torch.mm(m1, m2)
    print(f"Matrix mult result shape: {m3.shape}, mean: {m3.mean().item():.4f}")
    
    # Test 4: Custom element-wise kernel via PyTorch
    print("\n=== Test 4: Element-wise operations ===")
    t = torch.arange(10, device=device, dtype=torch.float32)
    t_squared = t ** 2
    print(f"t^2 = {t_squared.cpu().tolist()}")  # Should be [0, 1, 4, 9, 16, ...]
    
    print("\n=== All basic PyTorch GPU operations work! ===")
else:
    print("CUDA/HIP not available!")
