"""Progressive precision isolation: identify which stage of the serial kernel
diverges between PTO and AscendC backends.

Stage 1: Pass 1 accumulation (sum_a, sum_sq_a) -> reduce -> mean/var/std
Stage 2: Pass 2 normalization (sub/div/mul/add) -> output
"""
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


def build_stage1(target):
    """Only compute mean and std (Pass 1), output them for comparison."""
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def kernel(N, G, cpg_padded, S_padded, block_S, s_num, eps, cpg, S_orig, dtype):
        block_num = N * G
        tile_elem = cpg * block_S

        @T.prim_func
        def main(
            x: T.Tensor((N, G, cpg, S_orig), dtype),
            out: T.Tensor((N, G, 3), "float32"),
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
                    ob = T.alloc_ub([3], "float32")

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

                        T.barrier_all()
                        T.copy(total, ob[0:1])
                        T.copy(total_sq, ob[1:2])
                        T.copy(std_val, ob[2:3])
                        T.barrier_all()
                        T.copy(ob, out[n, g, 0:3])
        return main
    return kernel


def build_full_pto_vs_ascendc(target):
    """Full serial kernel for comparison."""
    CAST_LOW2HIGH = "CAST_NONE"
    CAST_HIGH2LOW = "CAST_RINT"

    @tilelang.jit(out_idx=[3], pass_configs=pass_configs, target=target)
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
            y: T.Tensor((N, G, cpg, S_padded), dtype),
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
                    out_buf_p2 = T.alloc_ub([2, cpg, block_S], dtype)

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
                        T.set_flag("mte3", "mte2", 0)
                        T.set_flag("mte3", "mte2", 1)
                        T.wait_flag("mte3", "mte2", 0)
                        T.copy(x[n, g, 0:cpg, 0:block_S], data_buf_p2[0, :, :])
                        T.set_flag("mte2", "v", 0)
                        for si in T.serial(s_num):
                            cur = si % 2
                            nxt = (si + 1) % 2
                            if si < s_num - 1:
                                s_off_nxt = (si + 1) * block_S
                                T.wait_flag("mte3", "mte2", nxt)
                                T.copy(x[n, g, 0:cpg, s_off_nxt: s_off_nxt + block_S], data_buf_p2[nxt, :, :])
                                T.set_flag("mte2", "v", nxt)
                            T.wait_flag("mte2", "v", cur)
                            if use_fp32:
                                T.tile.cast(data_cal_p2, data_buf_p2[cur, :, :], CAST_LOW2HIGH, tile_elem)
                            else:
                                T.copy(data_buf_p2[cur, :, :], data_cal_p2)
                            T.tile.sub(data_cal_p2, data_cal_p2, mean_bc)
                            T.tile.div(data_cal_p2, data_cal_p2, std_bc)
                            T.tile.mul(data_cal_p2, data_cal_p2, gamma_bc)
                            T.tile.add(data_cal_p2, data_cal_p2, beta_bc)
                            if use_fp32:
                                T.tile.cast(out_buf_p2[cur, :, :], data_cal_p2, CAST_HIGH2LOW, tile_elem)
                            else:
                                T.copy(data_cal_p2, out_buf_p2[cur, :, :])
                            T.set_flag("v", "mte3", cur)
                            T.wait_flag("v", "mte3", cur)
                            s_off_cur = si * block_S
                            T.copy(out_buf_p2[cur, :, :], y[n, g, 0:cpg, s_off_cur: s_off_cur + block_S])
                            T.set_flag("mte3", "mte2", cur)
                        T.wait_flag("mte3", "mte2", 0)
                        T.wait_flag("mte3", "mte2", 1)
        return main
    return kernel


x = torch.randn(N, num_groups, cpg, S, dtype=torch_dtype, device="npu")
gamma = torch.randn(num_groups, cpg, dtype=torch_dtype, device="npu")
beta = torch.randn(num_groups, cpg, dtype=torch_dtype, device="npu")

xf = x.float().cpu()
gf = gamma.float().cpu()
bf = beta.float().cpu()
cnt = cpg * S
g_sum = xf.sum().item()
g_mean = g_sum / cnt
g_sum_sq = (xf * xf).sum().item()
g_total_sq = g_sum_sq / cnt
g_var = g_total_sq - g_mean * g_mean + eps
g_std = g_var ** 0.5

print(f"\n=== Stage 1: Pass 1 (mean/var/std) ===")
print(f"[golden] mean={g_mean:.6f} total_sq={g_total_sq:.6f} std={g_std:.6f}")

for tgt in ["ascendc", "pto"]:
    try:
        kern = build_stage1(tgt)(N, num_groups, cpg_padded, S_padded, block_S, s_num, eps, cpg, S, dtype_str)
        out = kern(x)
        torch.npu.synchronize()
        yk = out.float().cpu()
        for n_i in range(min(N, 2)):
            for g_i in range(min(num_groups, 2)):
                vals = yk[n_i, g_i].tolist()
                print(f"[{tgt}] block({n_i},{g_i}): mean={vals[0]:.6f} total_sq={vals[1]:.6f} std={vals[2]:.6f}")
    except Exception as e:
        print(f"[{tgt}] ERROR {str(e)[:200]}")

print(f"\n=== Full kernel comparison ===")
for tgt in ["ascendc", "pto"]:
    try:
        kern = build_full_pto_vs_ascendc(tgt)(N, num_groups, cpg_padded, S_padded, block_S, s_num, eps, cpg, S, dtype_str)
        src = kern.get_kernel_source()
        out_path = os.path.join(os.path.dirname(__file__), f"_debug_src_{tgt}.cpp")
        with open(out_path, "w") as f:
            f.write(src)
        y = kern(x, gamma, beta)
        torch.npu.synchronize()
        yk = y[:, :, :, :S].float().cpu()
        yg = torch.nn.functional.group_norm(
            x.float().cpu().reshape(N, C, -1), num_groups,
            gamma.float().cpu().reshape(C), beta.float().cpu().reshape(C), eps
        ).reshape(N, num_groups, cpg, S)
        d = (yk - yg).abs()
        print(f"[{tgt}] nan={torch.isnan(yk).any().item()} maxdiff={d.max().item():.4e} meandiff={d.mean().item():.4e}")
        print(f"      sample[0,0,0,:4]={yk[0,0,0,:4].tolist()}")
        print(f"      golden[0,0,0,:4]={yg[0,0,0,:4].tolist()}")
    except Exception as e:
        print(f"[{tgt}] ERROR {str(e)[:200]}")
