"""Dump generated PTO kernel source for a small failing serial-kernel case."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import tilelang
from tilelang import language as T  # noqa
import torch
import example_group_norm as G

tilelang.disable_cache()
torch.manual_seed(0)

# Small case that uses the SERIAL kernel (s_num>=2) and PTO backend.
# all_zeros-like finite input, groups=1 -> should output beta (finite), PTO gives nan.
shape = [2, 8, 32, 32]  # S=1024 -> block_S=512, s_num=2 -> SERIAL (pto) path
num_groups = 2
dtype_str = "float32"
torch_dtype = torch.float32

x = torch.randn(shape, dtype=torch_dtype, device="npu")
C = shape[1]
gamma = torch.randn(C, dtype=torch_dtype, device="npu")
beta = torch.randn(C, dtype=torch_dtype, device="npu")

# Reproduce host dispatch math to confirm serial path
N = shape[0]
S = 1
for i in range(2, len(shape)):
    S *= shape[i]
cpg = C // num_groups
block_S = G._find_block_S(S, cpg, dtype_str)
s_num = (S + block_S - 1) // block_S
print(f"[cfg] N={N} C={C} groups={num_groups} cpg={cpg} S={S} block_S={block_S} s_num={s_num}")

# Build the serial kernel directly and dump source
cpg_padded = max(((cpg + 15) // 16) * 16, 16)
S_padded = s_num * block_S
kern = G.group_norm_kernel_serial(N, num_groups, cpg_padded, S_padded, block_S, s_num, 1e-5, cpg, S, dtype_str)
src = kern.get_kernel_source()
out = os.path.join(os.path.dirname(__file__), "_pto_serial.cpp")
with open(out, "w") as f:
    f.write(src)
print(f"[dump] wrote {len(src)} bytes to {out}")

# Run and compare
y = G.group_norm(x, gamma, beta, num_groups, 1e-5)
torch.npu.synchronize()
yg = G.golden_group_norm(x, gamma, beta, num_groups, 1e-5)
yk = y.float().cpu()
ygc = yg.float().cpu()
print("[out] kernel has nan:", torch.isnan(yk).any().item(), " golden has nan:", torch.isnan(ygc).any().item())
print("[out] kernel[0,0,0,:6]=", yk.reshape(N,C,-1)[0,0,:6].tolist())
print("[out] golden[0,0,0,:6]=", ygc.reshape(N,C,-1)[0,0,:6].tolist())
d = (yk-ygc).abs()
print("[out] max abs diff:", d.max().item(), " mean:", d.mean().item())
