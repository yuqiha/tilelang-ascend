"""
精确定位：分别输出 sum_a 和 sum_sq_a 的 reduce 结果
验证 sum_sq_a 的 fill → accumulate → reduce 链是否正确
"""
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
CAST_LOW2HIGH = "CAST_NONE"

CPG, BS = 4, 512


def build_sum_sq_test(target):
    """测试 sum_sq_a 的 fill → accumulate → reduce 链"""
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def k(cpg, block_S, s_num, S_orig):
        tile_elem = cpg * block_S
        @T.prim_func
        def main(x: T.Tensor((cpg, S_orig), "float32"),
                 out: T.Tensor((4,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    sum_a = T.alloc_ub([cpg, block_S], "float32")
                    sum_sq_a = T.alloc_ub([cpg, block_S], "float32")
                    sum_row = T.alloc_ub([cpg], "float32")
                    sum_sq_row = T.alloc_ub([cpg], "float32")
                    data_buf = T.alloc_ub([2, cpg, block_S], "float32")
                    total = T.alloc_ub([1], "float32")
                    total_sq = T.alloc_ub([1], "float32")
                    data_cal = T.alloc_ub([cpg, block_S], "float32")
                    ob = T.alloc_ub([4], "float32")

                    with T.Scope("V"):
                        T.tile.fill(sum_a, 0.0)
                        T.tile.fill(sum_sq_a, 0.0)
                        T.copy(x[0:cpg, 0:block_S], data_buf[0, :, :])
                        T.barrier_all()

                        for si in T.serial(s_num):
                            cur = si % 2
                            nxt = (si + 1) % 2
                            if si < s_num - 1:
                                s_off = (si + 1) * block_S
                                T.copy(x[0:cpg, s_off:s_off + block_S], data_buf[nxt, :, :])
                            T.copy(data_buf[cur, :, :], data_cal)
                            T.tile.add(sum_a, sum_a, data_cal)
                            T.tile.mul(data_cal, data_cal, data_cal)
                            T.tile.add(sum_sq_a, sum_sq_a, data_cal)
                            T.barrier_all()

                        T.reduce_sum(sum_a, sum_row, dim=-1)
                        T.reduce_sum(sum_sq_a, sum_sq_row, dim=-1)
                        T.reduce_sum(sum_row, total, dim=-1)
                        T.reduce_sum(sum_sq_row, total_sq, dim=-1)

                        T.barrier_all()
                        # 输出: sum_row[0], sum_sq_row[0], total, total_sq
                        T.copy(sum_row[0:1], ob[0:1])
                        T.copy(sum_sq_row[0:1], ob[1:2])
                        T.copy(total, ob[2:3])
                        T.copy(total_sq, ob[3:4])
                        T.barrier_all()
                        T.copy(ob, out)
        return main
    return k


S = 1024
s_num = S // BS  # 2
x = torch.randn(CPG, S, dtype=torch.float32, device="npu")
xf = x.float().cpu()

g_sum = xf.sum().item()
g_sum_sq = (xf * xf).sum().item()
g_sum_row0 = xf[0].sum().item()
g_sum_sq_row0 = (xf[0] * xf[0]).sum().item()

print(f"[golden] sum_row[0]={g_sum_row0:.6f}")
print(f"[golden] sum_sq_row[0]={g_sum_sq_row0:.6f}")
print(f"[golden] total={g_sum:.6f}")
print(f"[golden] total_sq={g_sum_sq:.6f}")

for tgt in ["ascendc", "pto"]:
    try:
        kern = build_sum_sq_test(tgt)(CPG, BS, s_num, S)
        out = kern(x)
        torch.npu.synchronize()
        yk = out.float().cpu().tolist()
        print(f"\n[{tgt}]")
        print(f"  sum_row[0]   = {yk[0]:.6f}  (golden={g_sum_row0:.6f}, err={abs(yk[0]-g_sum_row0):.2e})")
        print(f"  sum_sq_row[0]= {yk[1]:.6f}  (golden={g_sum_sq_row0:.6f}, err={abs(yk[1]-g_sum_sq_row0):.2e})")
        print(f"  total        = {yk[2]:.6f}  (golden={g_sum:.6f}, err={abs(yk[2]-g_sum):.2e})")
        print(f"  total_sq     = {yk[3]:.6f}  (golden={g_sum_sq:.6f}, err={abs(yk[3]-g_sum_sq):.2e})")
    except Exception as e:
        print(f"\n[{tgt}] ERROR: {str(e)[:200]}")
