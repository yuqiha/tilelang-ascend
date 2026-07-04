"""
阶段3: 精确隔离 - reduce + 标量运算链
每个子测试独立创建 kernel，避免 NPU 状态污染
"""
import tilelang
from tilelang import language as T
import torch
import sys

tilelang.disable_cache()
torch.manual_seed(42)

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CPG, BS = 4, 512
test_name = sys.argv[1] if len(sys.argv) > 1 else "all"


def run_test(name, builder_fn, x, golden, threshold=1e-2):
    print(f"\n--- {name} ---")
    for tgt in ["ascendc", "pto"]:
        try:
            kern = builder_fn(tgt)(CPG, BS)
            out = kern(x)
            torch.npu.synchronize()
            result = out.float().cpu()
            if result.dim() == 0:
                result_val = result.item()
                err = abs(result_val - golden)
                status = "PASS" if err < threshold else "FAIL"
                print(f"  [{tgt:8s}] {status} val={result_val:.6f} golden={golden:.6f} err={err:.2e}")
            else:
                err = (result - golden).abs().max().item()
                status = "PASS" if err < threshold else "FAIL"
                print(f"  [{tgt:8s}] {status} max_err={err:.2e}")
                print(f"           result={result.reshape(-1)[:4].tolist()}")
                print(f"           golden={golden.reshape(-1)[:4].tolist()}")
        except Exception as e:
            print(f"  [{tgt:8s}] ERROR: {str(e)[:150]}")


# 3a: 两个并行 reduce_sum 链
def build_3a(target):
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def k(cpg, block_S):
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"),
                 out: T.Tensor((2,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    sum_a = T.alloc_ub([cpg, block_S], "float32")
                    sum_sq = T.alloc_ub([cpg, block_S], "float32")
                    row = T.alloc_ub([cpg], "float32")
                    row_sq = T.alloc_ub([cpg], "float32")
                    total = T.alloc_ub([1], "float32")
                    total_sq = T.alloc_ub([1], "float32")
                    ob = T.alloc_ub([2], "float32")
                    with T.Scope("V"):
                        T.copy(x, xb)
                        T.tile.fill(sum_a, 0.0)
                        T.tile.fill(sum_sq, 0.0)
                        T.barrier_all()
                        T.copy(xb, sum_a)
                        T.tile.mul(xb, xb, xb)
                        T.copy(xb, sum_sq)
                        T.barrier_all()
                        T.reduce_sum(sum_a, row, dim=-1)
                        T.reduce_sum(sum_sq, row_sq, dim=-1)
                        T.reduce_sum(row, total, dim=-1)
                        T.reduce_sum(row_sq, total_sq, dim=-1)
                        T.barrier_all()
                        T.copy(total, ob[0:1])
                        T.copy(total_sq, ob[1:2])
                        T.copy(ob, out)
        return main
    return k


# 3b: reduce + 标量除法
def build_3b(target):
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def k(cpg, block_S):
        cnt_val = cpg * block_S
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"),
                 out: T.Tensor((1,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    sum_a = T.alloc_ub([cpg, block_S], "float32")
                    row = T.alloc_ub([cpg], "float32")
                    total = T.alloc_ub([1], "float32")
                    with T.Scope("V"):
                        T.copy(x, xb)
                        T.tile.fill(sum_a, 0.0)
                        T.tile.add(sum_a, sum_a, xb)
                        T.barrier_all()
                        T.reduce_sum(sum_a, row, dim=-1)
                        T.reduce_sum(row, total, dim=-1)
                        cnt = T.cast(cnt_val, "float32")
                        T.tile.div(total, total, cnt)
                        T.barrier_all()
                        T.copy(total, out)
        return main
    return k


# 3c: reduce + 标量运算链 (mean, var, std)
def build_3c(target):
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def k(cpg, block_S):
        cnt_val = cpg * block_S
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"),
                 out: T.Tensor((3,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    sum_a = T.alloc_ub([cpg, block_S], "float32")
                    sum_sq = T.alloc_ub([cpg, block_S], "float32")
                    row = T.alloc_ub([cpg], "float32")
                    row_sq = T.alloc_ub([cpg], "float32")
                    total = T.alloc_ub([1], "float32")
                    total_sq = T.alloc_ub([1], "float32")
                    mean_sq_val = T.alloc_ub([1], "float32")
                    var_val = T.alloc_ub([1], "float32")
                    std_val = T.alloc_ub([1], "float32")
                    ob = T.alloc_ub([3], "float32")
                    with T.Scope("V"):
                        T.copy(x, xb)
                        T.tile.fill(sum_a, 0.0)
                        T.tile.fill(sum_sq, 0.0)
                        T.tile.add(sum_a, sum_a, xb)
                        T.tile.mul(xb, xb, xb)
                        T.tile.add(sum_sq, sum_sq, xb)
                        T.barrier_all()
                        T.reduce_sum(sum_a, row, dim=-1)
                        T.reduce_sum(sum_sq, row_sq, dim=-1)
                        T.reduce_sum(row, total, dim=-1)
                        T.reduce_sum(row_sq, total_sq, dim=-1)
                        cnt = T.cast(cnt_val, "float32")
                        T.tile.div(total, total, cnt)
                        T.tile.div(total_sq, total_sq, cnt)
                        T.tile.mul(mean_sq_val, total, total)
                        T.tile.sub(var_val, total_sq, mean_sq_val)
                        eps_v = T.cast(1e-5, "float32")
                        T.tile.add(var_val, var_val, eps_v)
                        T.tile.sqrt(std_val, var_val)
                        T.barrier_all()
                        T.copy(total, ob[0:1])
                        T.copy(total_sq, ob[1:2])
                        T.copy(std_val, ob[2:3])
                        T.copy(ob, out)
        return main
    return k


# 3d: fill + broadcast
def build_3d(target):
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def k(cpg, block_S):
        @T.prim_func
        def main(scalar_in: T.Tensor((1,), "float32"),
                 out: T.Tensor((cpg, block_S), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    sb = T.alloc_ub([1], "float32")
                    col = T.alloc_ub([cpg, 1], "float32")
                    bc = T.alloc_ub([cpg, block_S], "float32")
                    with T.Scope("V"):
                        T.copy(scalar_in, sb)
                        T.barrier_all()
                        T.tile.fill(col, sb)
                        T.tile.broadcast(bc, col)
                        T.barrier_all()
                        T.copy(bc, out)
        return main
    return k


x = torch.randn(CPG, BS, dtype=torch.float32, device="npu")
xf = x.float().cpu()
cnt = CPG * BS

if test_name in ("all", "3a"):
    g_sum = xf.sum().item()
    g_sum_sq = (xf * xf).sum().item()
    golden_3a = torch.tensor([g_sum, g_sum_sq])
    run_test("3a: 两个并行 reduce_sum 链", build_3a, x, golden_3a, threshold=1.0)

if test_name in ("all", "3b"):
    g_mean = xf.sum().item() / cnt
    run_test("3b: reduce + 标量除法 (mean)", build_3b, x, torch.tensor([g_mean]), threshold=1e-2)

if test_name in ("all", "3c"):
    g_mean = xf.sum().item() / cnt
    g_total_sq = (xf * xf).sum().item() / cnt
    g_var = g_total_sq - g_mean * g_mean + 1e-5
    g_std = g_var ** 0.5
    golden_3c = torch.tensor([g_mean, g_total_sq, g_std])
    run_test("3c: reduce + 标量运算链 (mean/var/std)", build_3c, x, golden_3c, threshold=1e-2)

if test_name in ("all", "3d"):
    scalar_val = torch.tensor([3.14], dtype=torch.float32, device="npu")
    golden_3d = torch.full((CPG, BS), 3.14)
    run_test("3d: fill(col, scalar) + broadcast", build_3d, scalar_val, golden_3d, threshold=1e-4)
