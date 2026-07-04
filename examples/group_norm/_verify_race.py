"""
验证假设：PTO 的 fill (PIPE_S) 和后续 vector 操作 (PIPE_V) 之间存在流水线竞争。

对比两个版本：
1. fill → add（无 barrier，可能有竞争）
2. fill → barrier → add（有 barrier，无竞争）
"""
import tilelang
from tilelang import language as T
import torch

tilelang.disable_cache()
torch.manual_seed(42)

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CPG, BS = 4, 512


def build_no_barrier(target):
    """fill → add（无 barrier）"""
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def k(cpg, block_S):
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"),
                 out: T.Tensor((cpg, block_S), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    acc = T.alloc_ub([cpg, block_S], "float32")
                    with T.Scope("V"):
                        T.copy(x, xb)
                        T.tile.fill(acc, 0.0)
                        T.barrier_all()
                        T.tile.add(acc, acc, xb)
                        T.barrier_all()
                        T.copy(acc, out)
        return main
    return k


def build_with_barrier(target):
    """fill → barrier → add（有 barrier）"""
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def k(cpg, block_S):
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"),
                 out: T.Tensor((cpg, block_S), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    acc = T.alloc_ub([cpg, block_S], "float32")
                    with T.Scope("V"):
                        T.copy(x, xb)
                        T.tile.fill(acc, 0.0)
                        T.barrier_all()
                        T.tile.add(acc, acc, xb)
                        T.barrier_all()
                        T.copy(acc, out)
        return main
    return k


def build_fill_then_copy_then_add(target):
    """fill → GM copy → barrier → add（模拟原始 kernel 的模式）"""
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def k(cpg, block_S):
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"),
                 out: T.Tensor((cpg, block_S), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    acc = T.alloc_ub([cpg, block_S], "float32")
                    with T.Scope("V"):
                        T.tile.fill(acc, 0.0)
                        T.copy(x, xb)
                        T.barrier_all()
                        T.tile.add(acc, acc, xb)
                        T.barrier_all()
                        T.copy(acc, out)
        return main
    return k


x = torch.randn(CPG, BS, dtype=torch.float32, device="npu")
golden = x.float().cpu()

print("=" * 60)
print("验证 fill + add 流水线竞争")
print("=" * 60)

print("\n--- Test A: fill → GM_copy → barrier → add (原始 kernel 模式) ---")
for tgt in ["ascendc", "pto"]:
    try:
        kern = build_fill_then_copy_then_add(tgt)(CPG, BS)
        out = kern(x)
        torch.npu.synchronize()
        result = out.float().cpu()
        err = (result - golden).abs().max().item()
        status = "PASS" if err < 1e-4 else "FAIL"
        print(f"  [{tgt:8s}] {status} max_err={err:.2e}")
    except Exception as e:
        print(f"  [{tgt:8s}] ERROR: {str(e)[:150]}")

print("\n--- Test B: GM_copy → fill → barrier → add ---")
for tgt in ["ascendc", "pto"]:
    try:
        kern = build_no_barrier(tgt)(CPG, BS)
        out = kern(x)
        torch.npu.synchronize()
        result = out.float().cpu()
        err = (result - golden).abs().max().item()
        status = "PASS" if err < 1e-4 else "FAIL"
        print(f"  [{tgt:8s}] {status} max_err={err:.2e}")
    except Exception as e:
        print(f"  [{tgt:8s}] ERROR: {str(e)[:150]}")
