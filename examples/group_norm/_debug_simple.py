"""简化版精度隔离：只输出单个 block 的 mean/std 对比 PTO vs AscendC"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import tilelang
from tilelang import language as T
import torch

tilelang.disable_cache()
torch.manual_seed(42)

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

N, C, S = 2, 8, 1024
num_groups = 2
cpg = C // num_groups  # 4
cpg_padded = max(((cpg + 15) // 16) * 16, 16)  # 16
block_S = 512
s_num = S // block_S  # 2
S_padded = s_num * block_S  # 1024
dtype_str = "float32"
torch_dtype = torch.float32
eps = 1e-5

print(f"[cfg] N={N} C={C} groups={num_groups} cpg={cpg} S={S} block_S={block_S} s_num={s_num}")


def build_stage1_simple(target):
    """简化版：只处理第一个 block，输出 mean/var/std"""
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def kernel(cpg, block_S, s_num, S_orig, eps, dtype):
        @T.prim_func
        def main(
            x: T.Tensor((cpg, S_orig), dtype),
            out: T.Tensor((3,), "float32"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    sum_a = T.alloc_ub([cpg, block_S], "float32")
                    sum_sq_a = T.alloc_ub([cpg, block_S], "float32")
                    sum_row = T.alloc_ub([cpg], "float32")
                    sum_sq_row = T.alloc_ub([cpg], "float32")
                    data_buf_p1 = T.alloc_ub([2, cpg, block_S], dtype)
                    total = T.alloc_ub([1], "float32")
                    total_sq = T.alloc_ub([1], "float32")
                    mean_sq_val = T.alloc_ub([1], "float32")
                    var_val = T.alloc_ub([1], "float32")
                    std_val = T.alloc_ub([1], "float32")
                    data_cal = T.alloc_ub([cpg, block_S], "float32")
                    ob = T.alloc_ub([3], "float32")

                    with T.Scope("V"):
                        T.tile.fill(sum_a, 0.0)
                        T.tile.fill(sum_sq_a, 0.0)
                        T.copy(x[0:cpg, 0:block_S], data_buf_p1[0, :, :])
                        T.barrier_all()

                        for si in T.serial(s_num):
                            cur = si % 2
                            nxt = (si + 1) % 2
                            if si < s_num - 1:
                                s_off_nxt = (si + 1) * block_S
                                T.copy(
                                    x[0:cpg, s_off_nxt: s_off_nxt + block_S],
                                    data_buf_p1[nxt, :, :],
                                )
                            T.copy(data_buf_p1[cur, :, :], data_cal)
                            T.tile.add(sum_a, sum_a, data_cal)
                            T.tile.mul(data_cal, data_cal, data_cal)
                            T.tile.add(sum_sq_a, sum_sq_a, data_cal)
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
    return kernel


x = torch.randn(cpg, S, dtype=torch_dtype, device="npu")
xf = x.float().cpu()
cnt = cpg * S
g_sum = xf.sum().item()
g_mean = g_sum / cnt
g_sum_sq = (xf * xf).sum().item()
g_total_sq = g_sum_sq / cnt
g_var = g_total_sq - g_mean * g_mean + eps
g_std = g_var ** 0.5

print(f"\n=== Pass 1: mean/var/std ===")
print(f"[golden] mean={g_mean:.6f} total_sq={g_total_sq:.6f} std={g_std:.6f}")

for tgt in ["ascendc", "pto"]:
    try:
        kern = build_stage1_simple(tgt)(cpg, block_S, s_num, S, eps, dtype_str)
        out = kern(x)
        torch.npu.synchronize()
        yk = out.float().cpu().tolist()
        print(f"[{tgt:8s}] mean={yk[0]:.6f} total_sq={yk[1]:.6f} std={yk[2]:.6f}")
        print(f"         diff: mean={abs(yk[0]-g_mean):.2e} total_sq={abs(yk[1]-g_total_sq):.2e} std={abs(yk[2]-g_std):.2e}")
    except Exception as e:
        print(f"[{tgt:8s}] ERROR {str(e)[:200]}")
