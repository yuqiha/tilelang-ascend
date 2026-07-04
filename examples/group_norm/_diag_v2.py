"""
精确定位：在完整 kernel 基础上，额外输出 total_sq 和 std_val。
使用 4D 输出，避免 crash。

输出布局 (block 0,0):
  row 0, col 0: total (mean * count)
  row 0, col 1: total_sq (sum_sq / count)
  row 0, col 2: var_val
  row 0, col 3: std_val
  row 1, col 0: mean_bc 采样
  row 1, col 1: std_bc 采样
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

print(f"[cfg] cpg={cpg} S={S} block_S={block_S} s_num={s_num}")


def make_diag_kernel(target):
    @tilelang.jit(out_idx=[3], pass_configs=pass_configs, target=target)
    def kernel(N, G, cpg_padded, S_padded, block_S, s_num, eps, cpg, S_orig, dtype):
        block_num = N * G
        tile_elem = cpg * block_S

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
                    mean_col = T.alloc_ub([cpg, 1], "float32")
                    std_col = T.alloc_ub([cpg, 1], "float32")
                    data_cal = T.alloc_ub([cpg, block_S], "float32")
                    mean_bc = T.alloc_ub([cpg, block_S], "float32")
                    std_bc = T.alloc_ub([cpg, block_S], "float32")
                    out_buf = T.alloc_ub([cpg, block_S], "float32")

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
                        T.tile.mul(mean_sq_val, total, total)
                        T.tile.sub(var_val, total_sq, mean_sq_val)
                        eps_v = T.cast(eps, "float32")
                        T.tile.add(var_val, var_val, eps_v)
                        T.tile.sqrt(std_val, var_val)

                        T.tile.fill(mean_col, total)
                        T.tile.broadcast(mean_bc, mean_col)
                        T.tile.fill(std_col, std_val)
                        T.tile.broadcast(std_bc, std_col)

                        # 用 fill 将标量值写入 out_buf 的各行
                        # row 0 = total (mean)
                        T.tile.fill(out_buf[0:1, 0:block_S], total)
                        # row 1 = total_sq (E[x²])
                        T.tile.fill(out_buf[1:2, 0:block_S], total_sq)
                        # row 2 = var_val
                        T.tile.fill(out_buf[2:3, 0:block_S], var_val)
                        # row 3 = std_val
                        T.tile.fill(out_buf[3:4, 0:block_S], std_val)
                        T.barrier_all()
                        T.copy(out_buf, y[n, g, 0:cpg, 0:block_S])
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
g_var = block.var(unbiased=False).item()
g_std = (g_var + eps) ** 0.5
g_total = g_mean
g_total_sq = (block * block).mean().item()
g_var_val = g_total_sq - g_mean * g_mean + eps

print(f"[golden] total(=mean)={g_mean:.6f}")
print(f"[golden] total_sq(=E[x²])={g_total_sq:.6f}")
print(f"[golden] var_val={g_var_val:.6f}")
print(f"[golden] std_val={g_std:.6f}")

for tgt in ["ascendc", "pto"]:
    try:
        kern = make_diag_kernel(tgt)(N, num_groups, cpg_padded, S_padded, block_S, s_num, eps, cpg, S, dtype_str)
        y = kern(x4, g2, b2)
        torch.npu.synchronize()
        yk = y.float().cpu()
        # row 0 = total, row 1 = total_sq, row 2 = var_val, row 3 = std_val
        total_val = yk[0, 0, 0, 0].item()
        total_sq_val = yk[0, 0, 1, 0].item()
        var_val_out = yk[0, 0, 2, 0].item()
        std_val_out = yk[0, 0, 3, 0].item()
        print(f"\n[{tgt}]")
        print(f"  total     = {total_val:.6f}  (golden={g_mean:.6f})")
        print(f"  total_sq  = {total_sq_val:.6f}  (golden={g_total_sq:.6f})")
        print(f"  var_val   = {var_val_out:.6f}  (golden={g_var_val:.6f})")
        print(f"  std_val   = {std_val_out:.6f}  (golden={g_std:.6f})")
    except Exception as e:
        print(f"\n[{tgt}] ERROR: {str(e)[:200]}")
