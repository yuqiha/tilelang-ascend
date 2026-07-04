"""Build the SAME serial kernel under ascendc and pto, dump both sources, compare outputs."""
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


def make_serial(target):
    @tilelang.jit(out_idx=[3], pass_configs=pass_configs, target=target)
    def group_norm_kernel_serial(N, G, cpg_padded, S_padded, block_S, s_num, eps=1e-5, cpg=0, S_orig=0, dtype="float32"):
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
    return group_norm_kernel_serial


def golden(x, gamma, beta, num_groups, eps):
    return torch.nn.functional.group_norm(x, num_groups, gamma, beta, eps)


import ast
shape = ast.literal_eval(os.environ.get("SHP", "[2, 8, 32, 32]"))
num_groups = int(os.environ.get("GRP", "2"))
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
x4 = x.reshape(N, num_groups, cpg, S)
g2 = gamma.reshape(num_groups, cpg)
b2 = beta.reshape(num_groups, cpg)

yg = golden(x, gamma, beta, num_groups, eps).float().cpu()

for tgt in ["ascendc", "pto"]:
    kern = make_serial(tgt)(N, num_groups, cpg_padded, S_padded, block_S, s_num, eps, cpg, S, dtype_str)
    src = kern.get_kernel_source()
    out = os.path.join(os.path.dirname(__file__), f"_src_{tgt}.cpp")
    with open(out, "w") as f:
        f.write(src)
    y = kern(x4, g2, b2)
    torch.npu.synchronize()
    yk = y[:, :, :, :S].reshape(shape).float().cpu()
    d = (yk - yg).abs()
    print(f"[{tgt}] nan={torch.isnan(yk).any().item()} maxdiff={d.max().item():.4e} meandiff={d.mean().item():.4e}")
    print(f"      kernel[0,0,0,:4]={yk.reshape(N,C,-1)[0,0,:4].tolist()}")
    print(f"      golden[0,0,0,:4]={yg.reshape(N,C,-1)[0,0,:4].tolist()}")
