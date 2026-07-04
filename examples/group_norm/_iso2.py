"""Isolate the full pass-1 scalar reduction chain: sum_a->sum_row->total, var/std.
Output mean(total), total_sq, std to compare pto vs ascendc vs hand golden."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import tilelang
from tilelang import language as T
import torch

tilelang.disable_cache()
torch.manual_seed(0)

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CPG, BS = 4, 512
S_ORIG = BS  # single tile, s_num=1 equivalent for this isolation


def build_chain(target):
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def k(cpg, block_S, S_orig, eps=1e-5):
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"),
                 out: T.Tensor((3,), "float32")):  # [mean, total_sq, std]
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    sum_a = T.alloc_ub([cpg, block_S], "float32")
                    sum_sq_a = T.alloc_ub([cpg, block_S], "float32")
                    sum_row = T.alloc_ub([cpg], "float32")
                    sum_sq_row = T.alloc_ub([cpg], "float32")
                    total = T.alloc_ub([1], "float32")
                    total_sq = T.alloc_ub([1], "float32")
                    mean_sq_val = T.alloc_ub([1], "float32")
                    var_val = T.alloc_ub([1], "float32")
                    std_val = T.alloc_ub([1], "float32")
                    ob = T.alloc_ub([3], "float32")
                    with T.Scope("V"):
                        T.tile.fill(sum_a, 0.0)
                        T.tile.fill(sum_sq_a, 0.0)
                        T.copy(x[0:cpg, 0:block_S], xb)
                        T.barrier_all()
                        T.tile.add(sum_a, sum_a, xb)
                        T.tile.mul(xb, xb, xb)
                        T.tile.add(sum_sq_a, sum_sq_a, xb)
                        T.barrier_all()
                        T.reduce_sum(sum_a, sum_row, dim=-1)
                        T.reduce_sum(sum_sq_a, sum_sq_row, dim=-1)
                        T.reduce_sum(sum_row, total, dim=-1)
                        T.reduce_sum(sum_sq_row, total_sq, dim=-1)
                        cnt = T.cast(cpg * S_orig, "float32")
                        T.tile.div(total, total, cnt)
                        T.tile.div(total_sq, total_sq, cnt)
                        T.tile.mul(mean_sq_val, total, total)
                        T.tile.sub(var_val, total_sq, mean_sq_val)
                        eps_v = T.cast(eps, "float32")
                        T.tile.add(var_val, var_val, eps_v)
                        T.tile.sqrt(std_val, var_val)
                        T.barrier_all()
                        T.copy(total, ob[0:1])
                        T.copy(total_sq, ob[1:2])
                        T.copy(std_val, ob[2:3])
                        T.barrier_all()
                        T.copy(ob, out)
        return main
    return k


x = torch.randn(CPG, BS, dtype=torch.float32, device="npu")
xf = x.float().cpu()
cnt = CPG * S_ORIG
g_mean = xf.sum().item() / cnt
g_total_sq = (xf * xf).sum().item() / cnt
g_var = g_total_sq - g_mean * g_mean + 1e-5
g_std = g_var ** 0.5
print(f"golden: mean={g_mean:.6f} total_sq={g_total_sq:.6f} std={g_std:.6f}")

for tgt in ["ascendc", "pto"]:
    try:
        kern = build_chain(tgt)(CPG, BS, S_ORIG)
        y = kern(x)
        torch.npu.synchronize()
        yk = y.float().cpu().tolist()
        print(f"[{tgt}] mean={yk[0]:.6f} total_sq={yk[1]:.6f} std={yk[2]:.6f}")
    except Exception as e:
        print(f"[{tgt}] ERROR {str(e)[:200]}")
