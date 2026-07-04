"""Isolated single-op tests: reduce_sum, broadcast, copy — compare pto vs ascendc vs hand golden."""
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


def build_reduce(target):
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def k(cpg, block_S):
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"), out: T.Tensor((cpg,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    ob = T.alloc_ub([cpg], "float32")
                    with T.Scope("V"):
                        T.copy(x[0:cpg, 0:block_S], xb)
                        T.barrier_all()
                        T.reduce_sum(xb, ob, dim=-1)
                        T.barrier_all()
                        T.copy(ob, out)
        return main
    return k


def build_bcast(target):
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def k(cpg, block_S):
        @T.prim_func
        def main(col: T.Tensor((cpg, 1), "float32"), out: T.Tensor((cpg, block_S), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    cb = T.alloc_ub([cpg, 1], "float32")
                    ob = T.alloc_ub([cpg, block_S], "float32")
                    with T.Scope("V"):
                        T.copy(col[0:cpg, 0:1], cb)
                        T.barrier_all()
                        T.tile.broadcast(ob, cb)
                        T.barrier_all()
                        T.copy(ob, out)
        return main
    return k


def run(name, builder, xin, golden):
    print(f"\n=== {name} ===")
    for tgt in ["ascendc", "pto"]:
        try:
            kern = builder(tgt)(CPG, BS)
            y = kern(xin)
            torch.npu.synchronize()
            yk = y.float().cpu()
            d = (yk - golden).abs()
            print(f"[{tgt}] nan={torch.isnan(yk).any().item()} maxdiff={d.max().item():.4e} out(sample)={yk.reshape(-1)[:4].tolist()}")
        except Exception as e:
            print(f"[{tgt}] ERROR {str(e)[:150]}")
    print(f"golden(sample)={golden.reshape(-1)[:4].tolist()}")


# reduce
x = torch.randn(CPG, BS, dtype=torch.float32, device="npu")
g_red = x.float().cpu().sum(dim=-1)
run("reduce_sum [cpg,BS]->[cpg]", build_reduce, x, g_red)

# broadcast
col = torch.randn(CPG, 1, dtype=torch.float32, device="npu")
g_bc = col.float().cpu().expand(CPG, BS).contiguous()
run("broadcast [cpg,1]->[cpg,BS]", build_bcast, col, g_bc)
