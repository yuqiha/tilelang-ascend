"""
精确隔离 sum_sq_a 链路中出错的环节。
使用与 _dbg_cmp.py 相同的 4D 输出模式避免 crash。

Test A: 输出 sum_sq_a 累积结果（验证 fill + accumulate）
Test B: 输出 sum_sq_row（验证第一次 reduce_sum）
Test C: 输出 total_sq（验证第二次 reduce_sum）
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
cpg = C // num_groups  # 4
cpg_padded = max(((cpg + 15) // 16) * 16, 16)  # 16
block_S = 512
s_num = S // block_S  # 2
S_padded = s_num * block_S
dtype_str = "float32"
torch_dtype = torch.float32

print(f"[cfg] cpg={cpg} S={S} block_S={block_S} s_num={s_num}")


# ============================================================
# Test A: 输出 sum_sq_a 累积结果 (fill + mul + add)
# ============================================================
def make_test_a(target):
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def kernel(cpg_padded, S_padded, block_S, s_num, cpg, S_orig):
        block_num = N * num_groups
        tile_elem = cpg * block_S

        @T.prim_func
        def main(
            x: T.Tensor((N, num_groups, cpg, S_orig), "float32"),
            y: T.Tensor((N, num_groups, cpg, S_padded), "float32"),
        ):
            with T.Kernel(block_num, is_npu=True) as (cid, vid):
                n = cid // num_groups
                g = cid % num_groups
                if vid == 0:
                    sum_sq_a = T.alloc_ub([cpg, block_S], "float32")
                    data_buf = T.alloc_ub([2, cpg, block_S], "float32")
                    data_cal = T.alloc_ub([cpg, block_S], "float32")

                    with T.Scope("V"):
                        T.tile.fill(sum_sq_a, 0.0)
                        T.copy(x[n, g, 0:cpg, 0:block_S], data_buf[0, :, :])
                        T.barrier_all()

                        for si in T.serial(s_num):
                            cur = si % 2
                            nxt = (si + 1) % 2
                            if si < s_num - 1:
                                s_off = (si + 1) * block_S
                                T.copy(x[n, g, 0:cpg, s_off:s_off + block_S], data_buf[nxt, :, :])
                            T.copy(data_buf[cur, :, :], data_cal)
                            T.tile.mul(data_cal, data_cal, data_cal)
                            T.tile.add(sum_sq_a, sum_sq_a, data_cal)
                            T.barrier_all()

                        T.barrier_all()
                        T.copy(sum_sq_a, y[n, g, 0:cpg, 0:block_S])
        return main
    return kernel


# ============================================================
# Test B: 输出 sum_sq_row = reduce_sum(sum_sq_a, dim=-1)
# ============================================================
def make_test_b(target):
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def kernel(cpg_padded, S_padded, block_S, s_num, cpg, S_orig):
        block_num = N * num_groups
        tile_elem = cpg * block_S

        @T.prim_func
        def main(
            x: T.Tensor((N, num_groups, cpg, S_orig), "float32"),
            y: T.Tensor((N, num_groups, cpg, S_padded), "float32"),
        ):
            with T.Kernel(block_num, is_npu=True) as (cid, vid):
                n = cid // num_groups
                g = cid % num_groups
                if vid == 0:
                    sum_sq_a = T.alloc_ub([cpg, block_S], "float32")
                    sum_sq_row = T.alloc_ub([cpg], "float32")
                    data_buf = T.alloc_ub([2, cpg, block_S], "float32")
                    data_cal = T.alloc_ub([cpg, block_S], "float32")
                    out_buf = T.alloc_ub([cpg, block_S], "float32")

                    with T.Scope("V"):
                        T.tile.fill(sum_sq_a, 0.0)
                        T.copy(x[n, g, 0:cpg, 0:block_S], data_buf[0, :, :])
                        T.barrier_all()

                        for si in T.serial(s_num):
                            cur = si % 2
                            nxt = (si + 1) % 2
                            if si < s_num - 1:
                                s_off = (si + 1) * block_S
                                T.copy(x[n, g, 0:cpg, s_off:s_off + block_S], data_buf[nxt, :, :])
                            T.copy(data_buf[cur, :, :], data_cal)
                            T.tile.mul(data_cal, data_cal, data_cal)
                            T.tile.add(sum_sq_a, sum_sq_a, data_cal)
                            T.barrier_all()

                        T.reduce_sum(sum_sq_a, sum_sq_row, dim=-1)
                        T.barrier_all()

                        T.tile.fill(out_buf, 0.0)
                        T.barrier_all()
                        T.copy(sum_sq_row, out_buf[0:cpg, 0:1])
                        T.barrier_all()
                        T.copy(out_buf, y[n, g, 0:cpg, 0:block_S])
        return main
    return kernel


# ============================================================
# Test C: 输出 total_sq = reduce_sum(sum_sq_row, dim=-1)
# ============================================================
def make_test_c(target):
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def kernel(cpg_padded, S_padded, block_S, s_num, cpg, S_orig):
        block_num = N * num_groups
        tile_elem = cpg * block_S

        @T.prim_func
        def main(
            x: T.Tensor((N, num_groups, cpg, S_orig), "float32"),
            y: T.Tensor((N, num_groups, cpg, S_padded), "float32"),
        ):
            with T.Kernel(block_num, is_npu=True) as (cid, vid):
                n = cid // num_groups
                g = cid % num_groups
                if vid == 0:
                    sum_sq_a = T.alloc_ub([cpg, block_S], "float32")
                    sum_sq_row = T.alloc_ub([cpg], "float32")
                    total_sq = T.alloc_ub([1], "float32")
                    data_buf = T.alloc_ub([2, cpg, block_S], "float32")
                    data_cal = T.alloc_ub([cpg, block_S], "float32")
                    out_buf = T.alloc_ub([cpg, block_S], "float32")

                    with T.Scope("V"):
                        T.tile.fill(sum_sq_a, 0.0)
                        T.copy(x[n, g, 0:cpg, 0:block_S], data_buf[0, :, :])
                        T.barrier_all()

                        for si in T.serial(s_num):
                            cur = si % 2
                            nxt = (si + 1) % 2
                            if si < s_num - 1:
                                s_off = (si + 1) * block_S
                                T.copy(x[n, g, 0:cpg, s_off:s_off + block_S], data_buf[nxt, :, :])
                            T.copy(data_buf[cur, :, :], data_cal)
                            T.tile.mul(data_cal, data_cal, data_cal)
                            T.tile.add(sum_sq_a, sum_sq_a, data_cal)
                            T.barrier_all()

                        T.reduce_sum(sum_sq_a, sum_sq_row, dim=-1)
                        T.reduce_sum(sum_sq_row, total_sq, dim=-1)
                        T.barrier_all()

                        T.tile.fill(out_buf, 0.0)
                        T.barrier_all()
                        T.copy(total_sq, out_buf[0:1, 0:1])
                        T.barrier_all()
                        T.copy(out_buf, y[n, g, 0:cpg, 0:block_S])
        return main
    return kernel


# ============================================================
# 作为对照：Test A' 输出 sum_a 累积结果
# ============================================================
def make_test_a_sum(target):
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def kernel(cpg_padded, S_padded, block_S, s_num, cpg, S_orig):
        block_num = N * num_groups
        tile_elem = cpg * block_S

        @T.prim_func
        def main(
            x: T.Tensor((N, num_groups, cpg, S_orig), "float32"),
            y: T.Tensor((N, num_groups, cpg, S_padded), "float32"),
        ):
            with T.Kernel(block_num, is_npu=True) as (cid, vid):
                n = cid // num_groups
                g = cid % num_groups
                if vid == 0:
                    sum_a = T.alloc_ub([cpg, block_S], "float32")
                    data_buf = T.alloc_ub([2, cpg, block_S], "float32")
                    data_cal = T.alloc_ub([cpg, block_S], "float32")

                    with T.Scope("V"):
                        T.tile.fill(sum_a, 0.0)
                        T.copy(x[n, g, 0:cpg, 0:block_S], data_buf[0, :, :])
                        T.barrier_all()

                        for si in T.serial(s_num):
                            cur = si % 2
                            nxt = (si + 1) % 2
                            if si < s_num - 1:
                                s_off = (si + 1) * block_S
                                T.copy(x[n, g, 0:cpg, s_off:s_off + block_S], data_buf[nxt, :, :])
                            T.copy(data_buf[cur, :, :], data_cal)
                            T.tile.add(sum_a, sum_a, data_cal)
                            T.barrier_all()

                        T.barrier_all()
                        T.copy(sum_a, y[n, g, 0:cpg, 0:block_S])
        return main
    return kernel


x = torch.randn(N, num_groups, cpg, S, dtype=torch_dtype, device="npu")
xf = x.float().cpu()

# golden values for block (0, 0)
block = xf[0, 0]  # [4, 1024]
g_sum_a = block[:, :block_S].sum(dim=-1)  # [4] - sum of first 512 elements per row
g_sum_sq_a = (block[:, :block_S] ** 2).sum(dim=-1)  # [4]
g_sum_all = block.sum().item()
g_sum_sq_all = (block ** 2).sum().item()
g_sum_sq_row = (block ** 2).sum(dim=-1)  # [4]

print(f"[golden] sum_a[0,:4] per-row = {g_sum_a.tolist()}")
print(f"[golden] sum_sq_a[0,:4] per-row = {g_sum_sq_a.tolist()}")
print(f"[golden] sum_sq_row[0] per-row (full S) = {g_sum_sq_row.tolist()}")
print(f"[golden] total_sq (full S) = {g_sum_sq_all:.6f}")

tests = [
    ("A: sum_a accum", make_test_a_sum, lambda yk: yk[0, 0, :, :block_S].sum(dim=-1)),
    ("A: sum_sq_a accum", make_test_a, lambda yk: yk[0, 0, :, :block_S].sum(dim=-1)),
    ("B: sum_sq_row (1st reduce)", make_test_b, lambda yk: yk[0, 0, :, 0]),
    ("C: total_sq (2nd reduce)", make_test_c, lambda yk: yk[0, 0, 0, 0]),
]

for test_name, builder, extract_fn in tests:
    print(f"\n{'='*60}")
    print(f"Test {test_name}")
    print(f"{'='*60}")
    for tgt in ["ascendc", "pto"]:
        try:
            kern = builder(tgt)(cpg_padded, S_padded, block_S, s_num, cpg, S)
            y = kern(x)
            torch.npu.synchronize()
            yk = y.float().cpu()
            val = extract_fn(yk)
            if val.dim() == 0:
                print(f"  [{tgt:8s}] value={val.item():.6f}")
            else:
                print(f"  [{tgt:8s}] values={val.tolist()}")
        except Exception as e:
            print(f"  [{tgt:8s}] ERROR: {str(e)[:150]}")
