import tilelang
from tilelang import language as T
import torch
import numpy as np

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CAST_LOW2HIGH = "CAST_NONE"
CAST_HIGH2LOW = "CAST_RINT"


def find_block_S(S, cpg, dtype_str):
    UB_BUDGET = 192 * 1024
    cal_bytes = 4 if dtype_str in ("float16", "bfloat16") else int(dtype_str[-2:]) // 8
    dtype_bytes = cal_bytes if dtype_str == "float32" else 2
    c = max(cpg, 1)
    per_block = c * (6 * cal_bytes + 6 * dtype_bytes)
    max_block_S = (UB_BUDGET // per_block // 16) * 16
    max_block_S = min(max_block_S, 512)
    for bs in range(max_block_S, 0, -16):
        if S % bs == 0:
            return bs
    best = max_block_S
    min_pad = float("inf")
    for bs in range(max_block_S, 0, -16):
        pad = (bs - S % bs) % bs
        if pad < min_pad:
            min_pad = pad
            best = bs
    return max(16, best)


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def group_norm_kernel(
    N, G, cpg_padded, S_padded, block_S, s_num, eps=1e-5, cpg=0, S_orig=0, dtype="float32"
):
    block_num = N * G
    tile_elem = cpg * block_S

    use_fp32 = dtype in ("float16", "bfloat16")
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
                data_buf_p1 = T.alloc_ub([cpg, block_S], dtype)
                total = T.alloc_ub([1], cal_dtype)
                total_sq = T.alloc_ub([1], cal_dtype)
                mean_sq_val = T.alloc_ub([1], cal_dtype)
                var_val = T.alloc_ub([1], cal_dtype)
                std_val = T.alloc_ub([1], cal_dtype)
                mean_col = T.alloc_ub([cpg, 1], cal_dtype)
                std_col = T.alloc_ub([cpg, 1], cal_dtype)
                data_cal = T.alloc_ub([cpg, block_S], cal_dtype)

                with T.Scope("V"):
                    T.tile.fill(sum_a, 0.0)
                    T.tile.fill(sum_sq_a, 0.0)

                    for si in T.serial(s_num):
                        s_off = si * block_S
                        T.copy(x[n, g, 0:cpg, s_off : s_off + block_S], data_buf_p1)
                        if use_fp32:
                            T.tile.cast(data_cal, data_buf_p1, CAST_LOW2HIGH, tile_elem)
                        else:
                            T.copy(data_buf_p1, data_cal)
                        T.tile.add(sum_a, sum_a, data_cal)
                        T.tile.mul(data_cal, data_cal, data_cal)
                        T.tile.add(sum_sq_a, sum_sq_a, data_cal)

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
                data_buf_p2 = T.alloc_ub([cpg, block_S], dtype)
                data_cal_p2 = T.alloc_ub([cpg, block_S], cal_dtype)
                out_buf_p2 = T.alloc_ub([cpg, block_S], dtype)

                with T.Scope("V"):
                    T.tile.fill(mean_col, total)
                    T.tile.broadcast(mean_bc, mean_col)
                    T.tile.fill(std_col, std_val)
                    T.tile.broadcast(std_bc, std_col)

                    T.copy(gamma[g, 0:cpg], gamma_raw, pad_value=0.0)
                    T.copy(beta[g, 0:cpg], beta_raw, pad_value=0.0)
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

                    for si in T.serial(s_num):
                        s_off = si * block_S
                        T.copy(x[n, g, 0:cpg, s_off : s_off + block_S], data_buf_p2)
                        if use_fp32:
                            T.tile.cast(data_cal_p2, data_buf_p2, CAST_LOW2HIGH, tile_elem)
                        else:
                            T.copy(data_buf_p2, data_cal_p2)
                        T.tile.sub(data_cal_p2, data_cal_p2, mean_bc)
                        T.tile.div(data_cal_p2, data_cal_p2, std_bc)
                        T.tile.mul(data_cal_p2, data_cal_p2, gamma_bc)
                        T.tile.add(data_cal_p2, data_cal_p2, beta_bc)
                        if use_fp32:
                            T.tile.cast(out_buf_p2, data_cal_p2, CAST_HIGH2LOW, tile_elem)
                        else:
                            T.copy(data_cal_p2, out_buf_p2)
                        T.copy(out_buf_p2, y[n, g, 0:cpg, s_off : s_off + block_S])

    return main


def torch_group_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                     num_groups: int, epsilon: float) -> torch.Tensor:
    return torch.nn.functional.group_norm(
        input=x, num_groups=num_groups, weight=gamma, bias=beta, eps=epsilon,
    )


def run_group_norm(case_id: int, N: int, C: int, spatial_shape: tuple, dtype_str: str,
                   num_groups: int, epsilon: float, value_range: tuple,
                   block_S: int = None):
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[dtype_str]

    spatial_dims = list(spatial_shape)
    S = int(np.prod(spatial_dims)) if spatial_dims else 1

    x_shape = (N, C) + tuple(spatial_dims)
    x = torch.randn(x_shape, dtype=torch_dtype, device="npu")
    gamma = torch.randn(C, dtype=torch_dtype, device="npu")
    beta = torch.randn(C, dtype=torch_dtype, device="npu")

    cpg = C // num_groups
    cpg_padded = max(((cpg + 15) // 16) * 16, 16)
    if block_S is None:
        block_S = find_block_S(S, cpg, dtype_str)
    s_num = (S + block_S - 1) // block_S
    S_padded = s_num * block_S

    x_4d = x.reshape(N, num_groups, cpg, S)
    gamma_2d = gamma.reshape(num_groups, cpg)
    beta_2d = beta.reshape(num_groups, cpg)

    kernel = group_norm_kernel(
        N, num_groups, cpg_padded, S_padded, block_S, s_num, epsilon, cpg, S, dtype_str
    )
    y_4d = kernel(x_4d, gamma_2d, beta_2d)

    y_4d = y_4d[:, :, :, :S]
    y = y_4d.reshape(x_shape)

    ref = torch_group_norm(x, gamma, beta, num_groups, epsilon)

    rtol_map = {"float16": 1e-2, "float32": 1e-4, "bfloat16": 2e-2}
    atol_map = {"float16": 1e-2, "float32": 1e-5, "bfloat16": 2e-2}
    rtol = rtol_map.get(dtype_str, 1e-2)
    atol = atol_map.get(dtype_str, 1e-2)

    torch.testing.assert_close(y.cpu(), ref.cpu(), rtol=rtol, atol=atol)
    print(f"Case {case_id}: PASSED  "
          f"(N={N}, C={C}, spatial={spatial_dims}, dtype={dtype_str}, "
          f"G={num_groups}, block_S={block_S})")


if __name__ == "__main__":
    torch.manual_seed(42)

    test_cases = [
        (1,  8,   32,   (64, 64),       "float16",  8,   1e-5,  (-1, 1)),
        (2,  4,   64,   (128, 128),     "float32",  16,  1e-5,  (-2, 2)),
        (3,  2,   128,  (256, 256),     "bfloat16", 32,  1e-5,  (-3, 3)),
        (4,  16,  257,  (32, 31),       "float16",  1,   1e-5,  (-10, 10)),
        (5,  8,   512,  (17, 15),       "float32",  2,   1e-5,  (-100, 100)),
        (6,  64,  64,   (128,),         "bfloat16", 4,   1e-5,  (-5, 5)),
        (7,  2,   256,  (128, 128),     "float16",  16,  1e-5,  (-0.1, 0.1)),
        (8,  16,  127,  (31, 33),       "float32",  1,   1e-6,  (-1, 1)),
        (9,  3,   64,   (64, 64),       "bfloat16", 8,   1e-3,  (-0.5, 0.5)),
        (10, 7,   32,   (63, 65),       "float16",  4,   1e-4,  (-1, 2)),
        (11, 3,   64,   (127, 129),     "float32",  8,   1e-4,  (-50, 100)),
        (12, 5,   48,   (33, 65),       "bfloat16", 6,   1e-4,  (-3, 6)),
        (13, 1023, 257, (),             "float16",  1,   1e-6,  (-1, 1)),
        (14, 2,   60,   (5, 7, 480),    "float32",  4,   1e-5,  (-10, 10)),
        (15, 4,   31,   (251, 251),     "bfloat16", 1,   1e-8,  (float("-inf"), float("inf"))),
        (16, 2,   64,   (67, 71),       "float16",  8,   1e-7,  (float("nan"), float("nan"))),
        (17, 8,   127,  (33, 31),       "float32",  1,   1e-4,  (0, 0)),
        (18, 2,   256,  (127, 129),     "bfloat16", 16,  1e-5,  (-0.2, 0.2)),
        (19, 4,   128,  (255, 257),     "float16",  32,  1e-3,  (-65504, 65504)),
        (20, 1,   513,  (63, 63),       "float32",  3,   1e-6,  (-20, 40)),
    ]

    print("=" * 70)
    print("GroupNorm TileLang-Ascend 测试 (Multi-row Tile)")
    print(f"共 {len(test_cases)} 个测试用例")
    print("=" * 70)

    passed = 0
    failed = 0
    for case_id, N, C, spatial_shape, dtype_str, num_groups, epsilon, value_range in test_cases:
        try:
            run_group_norm(case_id, N, C, spatial_shape, dtype_str,
                           num_groups, epsilon, value_range)
            passed += 1
        except Exception as e:
            print(f"Case {case_id}: FAILED - {e}")
            failed += 1

    print("=" * 70)
    print(f"测试完成: {passed} passed, {failed} failed")
