"""
简化诊断：只运行 Pass 1，输出 mean 和 std，对比 PTO vs AscendC
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


def make_pass1(target):
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def kernel(N, G, cpg_padded, S_padded, block_S, s_num, eps, cpg, S_orig, dtype):
        block_num = N * G
        tile_elem = cpg * block_S
        use_fp32 = dtype in ["float16", "bfloat16"]
        cal_dtype = "float32" if use_fp32 else dtype

        @T.prim_func
        def main(
            x: T.Tensor((N, G, cpg, S_orig), dtype),
            out: T.Tensor((block_num, 2), "float32"),
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
                    data_cal = T.alloc_ub([cpg, block_S], cal_dtype)
                    ob = T.alloc_ub([2], "float32")

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
                        T.barrier_all()
                        T.copy(total, ob[0:1])
                        T.copy(std_val, ob[1:2])
                        T.barrier_all()
                        T.copy(ob, out[cid, 0:2])
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
    c = max(cpg, 1)
    per_block = c * (6 * 4 + 6 * 4)
    max_block_S = (UB_BUDGET // per_block // 16) * 16
    max_block_S = min(max_block_S, 512)
    for bs in range(max_block_S, 0, -16):
        if S % bs == 0:
            return bs
    return max(16, max_block_S)

block_S = _find_block_S(S, cpg, dtype_str)
s_num = (S + block_S - 1) // block_S
S_padded = s_num * block_S
block_num = N * num_groups
print(f"[cfg] N={N} C={C} groups={num_groups} cpg={cpg} S={S} block_S={block_S} s_num={s_num} blocks={block_num}")

x = torch.randn(shape, dtype=torch_dtype, device="npu")
x4 = x.reshape(N, num_groups, cpg, S)

# golden
xf = x.float().cpu().reshape(N, num_groups, cpg, S)
for n_i in range(N):
    for g_i in range(num_groups):
        block = xf[n_i, g_i]
        m = block.mean().item()
        v = block.var(unbiased=False).item()
        s = (v + eps) ** 0.5
        print(f"[golden] block({n_i},{g_i}): mean={m:.6f} std={s:.6f}")

print()
for tgt in ["ascendc", "pto"]:
    try:
        kern = make_pass1(tgt)(N, num_groups, cpg_padded, S_padded, block_S, s_num, eps, cpg, S, dtype_str)
        out = kern(x4)
        torch.npu.synchronize()
        yk = out.float().cpu()
        print(f"[{tgt}]")
        for cid in range(block_num):
            n_i = cid // num_groups
            g_i = cid % num_groups
            m = yk[cid, 0].item()
            s = yk[cid, 1].item()
            print(f"  block({n_i},{g_i}): mean={m:.6f} std={s:.6f}")
    except Exception as e:
        print(f"[{tgt}] ERROR: {str(e)[:200]}")
