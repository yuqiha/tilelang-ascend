"""
诊断脚本：输出 PTO 和 AscendC 的中间值，定位精度偏差的确切位置。
输出：mean, std, gamma_bc[0,0], beta_bc[0,0], 以及 normalize 后的前4个值
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
CAST_HIGH2LOW = "CAST_RINT"


def make_diag(target):
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def kernel(N, G, cpg_padded, S_padded, block_S, s_num, eps, cpg, S_orig, dtype):
        block_num = N * G
        tile_elem = cpg * block_S
        use_fp32 = dtype in ["float16", "bfloat16"]
        cal_dtype = "float32" if use_fp32 else dtype

        @T.prim_func
        def main(
            x: T.Tensor((N, G, cpg, S_orig), dtype),
            gamma: T.Tensor((G, cpg), dtype),
            beta: T.Tensor((G, cpg), dtype),
            out: T.Tensor((N, G, 8), "float32"),
        ):
            with T.Kernel(block_num, is_npu=True) as (cid, vid):
                n = cid // G
                g = cid % G
                if vid == 0:
                    sum_a = T.alloc_ub([cpg, block_S], cal_dtype)
                    sum_sq_a = T.alloc_ub([cpg, block_S], cal_dtype)
                    sum_row = T.alloc_ub([cpg], cal_dtype)
                    sum_sq_row = T.alloc_ub([cpg], cal_dtype)
                    data_buf_p1 = T.alloc_ub([2, cpg, block_S], dtype)
                    total = T.alloc_ub([1], cal_dtype)
                    total_sq = T.alloc_ub([1], cal_dtype)
                    mean_sq_val = T.alloc_ub([1], cal_dtype)
                    var_val = T.alloc_ub([1], cal_dtype)
                    std_val = T.alloc_ub([1], cal_dtype)
                    mean_col = T.alloc_ub([cpg, 1], cal_dtype)
                    std_col = T.alloc_ub([cpg, 1], cal_dtype)
                    data_cal = T.alloc_ub([cpg, block_S], cal_dtype)
                    mean_bc = T.alloc_ub([cpg, block_S], cal_dtype)
                    std_bc = T.alloc_ub([cpg, block_S], cal_dtype)
                    gamma_raw = T.alloc_ub([cpg_padded, 1], dtype)
                    beta_raw = T.alloc_ub([cpg_padded, 1], dtype)
                    gamma_cal = T.alloc_ub([cpg_padded, 1], cal_dtype)
                    beta_cal = T.alloc_ub([cpg_padded, 1], cal_dtype)
                    gamma_bc_full = T.alloc_ub([cpg_padded, block_S], cal_dtype)
                    beta_bc_full = T.alloc_ub([cpg_padded, block_S], cal_dtype)
                    gamma_bc = T.alloc_ub([cpg, block_S], cal_dtype)
                    beta_bc = T.alloc_ub([cpg, block_S], cal_dtype)
                    data_buf_p2 = T.alloc_ub([2, cpg, block_S], dtype)
                    data_cal_p2 = T.alloc_ub([cpg, block_S], cal_dtype)
                    ob = T.alloc_ub([8], "float32")

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
                                T.copy(x[n, g, 0:cpg, s_off_nxt: s_off_nxt + block_S], data_buf_p1[nxt, :, :])
                            if use_fp32:
                                T.tile.cast(data_cal, data_buf_p1[cur, :, :], CAST_LOW2HIGH, tile_elem)
                            else:
                                T.copy(data_buf_p1[cur, :, :], data_cal)
                            T.tile.add(sum_a, sum_a, data_cal)
                            T.tile.mul(data_cal, data_cal, data_cal)
                            T.tile.add(sum_sq_a, sum_sq_a, data_cal)
                            T.barrier_all()
                        T.reduce_sum(sum_a, sum_row, dim=-1)
                        T.reduce_sum(sum_sq_a, sum_sq_row, dim=-1)
                        T.reduce_sum(sum_row, total, dim=-1)
                        T.reduce_sum(sum_sq_row, total_sq, dim=-1)
                        cnt = T.cast(cpg * S_orig, cal_dtype)
                        T.tile.div(total, total, cnt)
                        T.tile.div(total_sq, total_sq, cnt)
                        T.tile.mul(mean_sq_val, total, total)
                        T.tile.sub(var_val, total_sq, mean_sq_val)
                        eps_v = T.cast(eps, cal_dtype)
                        T.tile.add(var_val, var_val, eps_v)
                        T.tile.sqrt(std_val, var_val)
                        T.tile.fill(mean_col, total)
                        T.tile.broadcast(mean_bc, mean_col)
                        T.tile.fill(std_col, std_val)
                        T.tile.broadcast(std_bc, std_col)
                        T.copy(gamma[g, 0:cpg], gamma_raw, pad_value=0.0)
                        T.copy(beta[g, 0:cpg], beta_raw, pad_value=0.0)
                        T.barrier_all()
                        if use_fp32:
                            T.tile.cast(gamma_cal, gamma_raw, CAST_LOW2HIGH, cpg_padded)
                            T.tile.cast(beta_cal, beta_raw, CAST_LOW2HIGH, cpg_padded)
                        else:
                            T.copy(gamma_raw, gamma_cal)
                            T.copy(beta_raw, beta_cal)
                        T.tile.broadcast(gamma_bc_full, gamma_cal)
                        T.tile.broadcast(beta_bc_full, beta_cal)
                        T.copy(gamma_bc_full[0:cpg, 0:block_S], gamma_bc)
                        T.copy(beta_bc_full[0:cpg, 0:block_S], beta_bc)
                        T.barrier_all()

                        # 输出中间值: mean, std, gamma_bc[0,0], beta_bc[0,0]
                        T.barrier_all()
                        T.copy(total, ob[0:1])
                        T.copy(std_val, ob[1:2])
                        T.copy(gamma_bc[0, 0:1], ob[2:3])
                        T.copy(beta_bc[0, 0:1], ob[3:4])
                        T.copy(mean_bc[0, 0:1], ob[4:5])
                        T.copy(std_bc[0, 0:1], ob[5:6])
                        T.copy(gamma_bc[1, 0:1], ob[6:7])
                        T.copy(beta_bc[1, 0:1], ob[7:8])
                        T.barrier_all()
                        T.copy(ob, out[n, g, 0:8])
        return main
    return kernel


shape = [2, 8, 32, 32]
num_groups = 2
dtype_str = "float32"
torch_dtype = torch.float32
eps = 1e-5
N, C = shape[0], shape[1]
S = shape[2] * shape[3]
cpg = C // num_groups
cpg_padded = max(((cpg + 15) // 16) * 16, 16)

def _find_block_S(S, cpg, dtype_str):
    UB_BUDGET = 192 * 1024
    cal_bytes = 4
    dtype_bytes = 4
    c = max(cpg, 1)
    per_block = c * (6 * cal_bytes + 6 * dtype_bytes)
    max_block_S = (UB_BUDGET // per_block // 16) * 16
    max_block_S = min(max_block_S, 512)
    for bs in range(max_block_S, 0, -16):
        if S % bs == 0:
            return bs
    return max(16, max_block_S)

block_S = _find_block_S(S, cpg, dtype_str)
s_num = (S + block_S - 1) // block_S
S_padded = s_num * block_S
print(f"[cfg] N={N} C={C} groups={num_groups} cpg={cpg} S={S} block_S={block_S} s_num={s_num}")

x = torch.randn(shape, dtype=torch_dtype, device="npu")
gamma = torch.randn(C, dtype=torch_dtype, device="npu")
beta = torch.randn(C, dtype=torch_dtype, device="npu")

print(f"\n[host] gamma={gamma.float().cpu().tolist()}")
print(f"[host] beta={beta.float().cpu().tolist()}")

x4 = x.reshape(N, num_groups, cpg, S)
g2 = gamma.reshape(num_groups, cpg)
b2 = beta.reshape(num_groups, cpg)

# golden 中间值
xf = x.float().cpu().reshape(N, num_groups, cpg, S)
gf = gamma.float().cpu().reshape(num_groups, cpg)
bf = beta.float().cpu().reshape(num_groups, cpg)
for n_i in range(N):
    for g_i in range(num_groups):
        block = xf[n_i, g_i]
        m = block.mean().item()
        v = block.var(unbiased=False).item()
        s = (v + eps) ** 0.5
        print(f"[golden] block({n_i},{g_i}): mean={m:.6f} std={s:.6f} gamma[0]={gf[g_i,0]:.6f} beta[0]={bf[g_i,0]:.6f}")

print()
for tgt in ["ascendc", "pto"]:
    kern = make_diag(tgt)(N, num_groups, cpg_padded, S_padded, block_S, s_num, eps, cpg, S, dtype_str)
    out = kern(x4, g2, b2)
    torch.npu.synchronize()
    yk = out.float().cpu()
    print(f"[{tgt}]")
    for n_i in range(N):
        for g_i in range(num_groups):
            v = yk[n_i, g_i].tolist()
            print(f"  block({n_i},{g_i}): mean={v[0]:.6f} std={v[1]:.6f} gamma_bc[0,0]={v[2]:.6f} beta_bc[0,0]={v[3]:.6f} mean_bc[0,0]={v[4]:.6f} std_bc[0,0]={v[5]:.6f} gamma_bc[1,0]={v[6]:.6f} beta_bc[1,0]={v[7]:.6f}")
