"""
精确诊断：在 TMULS 后立即保存 total_sq 副本，验证后续操作是否破坏了它。

Test A: TMULS → fill(total_sq) → output (已验证正确)
Test B: TMULS → copy total_sq to backup → TMUL → fill(backup) → output
Test C: TMULS → TMUL → TSUB → fill(var_val) → output
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

N, C, S = 2, 8, 1024
num_groups = 2
cpg = C // num_groups
cpg_padded = max(((cpg + 15) // 16) * 16, 16)
block_S = 512
s_num = S // block_S
S_padded = s_num * block_S
dtype_str = "float32"
torch_dtype = torch.float32
eps = 1e-5

mode = sys.argv[1] if len(sys.argv) > 1 else "B"
print(f"[cfg] cpg={cpg} S={S} block_S={block_S} s_num={s_num} mode={mode}")


def make_kernel(target, test_mode):
    @tilelang.jit(out_idx=[3], pass_configs=pass_configs, target=target)
    def kernel(N, G, cpg_padded, S_padded, block_S, s_num, eps, cpg, S_orig, dtype):
        block_num = N * G

        @T.prim_func
        def main(
            x: T.Tensor((N, G, cpg, S_orig), dtype),
            gamma: T.Tensor((G, cpg), dtype),
            beta: T.Tensor((G, cpg), dtype),
            y: T.Tensor((N, G, cpg, S_padded), dtype),
        ):
            with T.Kernel(block_num, is_npu=True) as (cid, vid):
                n = cid // G
                g = cid % G
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
                    out_a = T.alloc_ub([cpg, block_S], "float32")
                    out_b = T.alloc_ub([cpg, block_S], "float32")
                    backup = T.alloc_ub([1], "float32")

                    with T.Scope("V"):
                        T.tile.fill(sum_a, 0.0)
                        T.tile.fill(sum_sq_a, 0.0)
                        T.copy(x[n, g, 0:cpg, 0:block_S], data_buf_p1[0, :, :])
                        T.barrier_all()

                        for si in T.serial(s_num):
                            cur = si % 2
                            nxt = (si + 1) % 2
                            if si < s_num - 1:
                                s_off_nxt = (si + 1) * block_S
                                T.copy(
                                    x[n, g, 0:cpg, s_off_nxt: s_off_nxt + block_S],
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

                        if test_mode == "B":
                            # 立即备份 total_sq
                            T.copy(total_sq, backup)
                            # 执行 TMUL
                            T.tile.mul(mean_sq_val, total, total)
                            # 输出 backup 和 mean_sq_val
                            T.tile.fill(out_a, backup)
                            T.tile.fill(out_b, mean_sq_val)
                        elif test_mode == "C":
                            # 完整链: TMUL → TSUB → TADDS → TSQRT
                            T.tile.mul(mean_sq_val, total, total)
                            T.tile.sub(var_val, total_sq, mean_sq_val)
                            eps_v = T.cast(eps, "float32")
                            T.tile.add(var_val, var_val, eps_v)
                            T.tile.sqrt(std_val, var_val)
                            T.tile.fill(out_a, var_val)
                            T.tile.fill(out_b, std_val)
                        elif test_mode == "D":
                            # 只用 T.copy 代替 T.tile.sub
                            T.tile.mul(mean_sq_val, total, total)
                            T.copy(total_sq, backup)
                            T.tile.sub(var_val, backup, mean_sq_val)
                            eps_v = T.cast(eps, "float32")
                            T.tile.add(var_val, var_val, eps_v)
                            T.tile.sqrt(std_val, var_val)
                            T.tile.fill(out_a, var_val)
                            T.tile.fill(out_b, std_val)

                        T.barrier_all()
                        T.copy(out_a, y[n, g, 0:cpg, 0:block_S])
                        T.copy(out_b, y[n, g, 0:cpg, block_S:2*block_S])
        return main
    return kernel


x = torch.randn(N, num_groups, cpg, S, dtype=torch_dtype, device="npu")
gamma = torch.randn(C, dtype=torch_dtype, device="npu")
beta = torch.randn(C, dtype=torch_dtype, device="npu")
x4 = x.reshape(N, num_groups, cpg, S)
g2 = gamma.reshape(num_groups, cpg)
b2 = beta.reshape(num_groups, cpg)

xf = x.float().cpu().reshape(N, num_groups, cpg, S)
block = xf[0, 0]
g_mean = block.mean().item()
g_total_sq = (block * block).mean().item()
g_mean_sq = g_mean * g_mean
g_var = g_total_sq - g_mean_sq + eps
g_std = g_var ** 0.5

if mode == "B":
    print(f"[golden] backup(total_sq)={g_total_sq:.6f}  mean_sq_val={g_mean_sq:.6f}")
elif mode == "C":
    print(f"[golden] var_val={g_var:.6f}  std_val={g_std:.6f}")
elif mode == "D":
    print(f"[golden] var_val={g_var:.6f}  std_val={g_std:.6f} (with copy backup)")

for tgt in ["ascendc", "pto"]:
    try:
        kern = make_kernel(tgt, mode)(N, num_groups, cpg_padded, S_padded, block_S, s_num, eps, cpg, S, dtype_str)
        y = kern(x4, g2, b2)
        torch.npu.synchronize()
        yk = y.float().cpu()
        a_val = yk[0, 0, 0, 0].item()
        b_val = yk[0, 0, 0, block_S].item()
        if mode == "B":
            print(f"[{tgt:8s}] backup(total_sq)={a_val:.6f} (golden={g_total_sq:.6f})")
            print(f"           mean_sq_val   ={b_val:.6f} (golden={g_mean_sq:.6f})")
        elif mode in ("C", "D"):
            print(f"[{tgt:8s}] var_val={a_val:.6f} (golden={g_var:.6f})")
            print(f"           std_val={b_val:.6f} (golden={g_std:.6f})")
    except Exception as e:
        print(f"[{tgt:8s}] ERROR: {str(e)[:200]}")
