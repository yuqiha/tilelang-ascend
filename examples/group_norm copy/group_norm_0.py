import tilelang
from tilelang import language as T
import torch
import numpy as np

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def group_norm_kernel(N, C, S, num_groups, eps, block_S, dtype="float16"):
    """GroupNorm kernel - all ops in kernel.

    Processes one channel at a time (block_C=1) to avoid dim-0 broadcast/reduce issues.
    Always uses float32 compute internally for numerical stability.
    Gamma and Beta are passed as (C, 1) to enable [1, 1] loading.

    Args:
        N: batch size
        C: channels
        S: total spatial size
        num_groups: number of groups (G)
        eps: epsilon for numerical stability
        block_S: spatial tile size
        dtype: data type string
    """
    C_g = C // num_groups
    N_G = N * num_groups
    s_num = T.ceildiv(S, block_S)
    total_elts = C_g * S

    need_cast = dtype in ("float16", "bfloat16")

    @T.prim_func
    def main(
        X: T.Tensor((N, C, S), dtype),
        Gamma: T.Tensor((C, 1), dtype),
        Beta: T.Tensor((C, 1), dtype),
        Y: T.Tensor((N, C, S), dtype),
    ):
        with T.Kernel(N_G, is_npu=True) as (task_id, vid):
            n = task_id // num_groups
            g = task_id % num_groups
            c_start = g * C_g

            x_ub = T.alloc_ub([1, block_S], dtype)
            x_cal = T.alloc_ub([1, block_S], "float32")
            x_sq = T.alloc_ub([1, block_S], "float32")

            chunk_scalar = T.alloc_ub([1, 1], "float32")
            sum_acc = T.alloc_ub([1, 1], "float32")
            sum_sq_acc = T.alloc_ub([1, 1], "float32")

            mean_val = T.alloc_ub([1, 1], "float32")
            var_val = T.alloc_ub([1, 1], "float32")
            rstd_val = T.alloc_ub([1, 1], "float32")
            total_t = T.alloc_ub([1, 1], "float32")
            eps_t = T.alloc_ub([1, 1], "float32")
            one_t = T.alloc_ub([1, 1], "float32")

            mean_s = T.alloc_ub([1, block_S], "float32")
            rstd_s = T.alloc_ub([1, block_S], "float32")

            gamma_ub = T.alloc_ub([1, 1], dtype)
            gamma_cal = T.alloc_ub([1, 1], "float32")
            gamma_s = T.alloc_ub([1, block_S], "float32")
            beta_ub = T.alloc_ub([1, 1], dtype)
            beta_cal = T.alloc_ub([1, 1], "float32")
            beta_s = T.alloc_ub([1, block_S], "float32")

            with T.Scope("V"):
                T.tile.fill(sum_acc, 0.0)
                T.tile.fill(sum_sq_acc, 0.0)

                for c_local in T.serial(C_g):
                    for s_block in T.serial(s_num):
                        if need_cast:
                            T.copy(
                                X[
                                    n,
                                    c_start + c_local,
                                    s_block * block_S: (s_block + 1) * block_S,
                                ],
                                x_ub,
                            )
                            T.tile.cast(x_cal, x_ub, CAST_MODE_LOW2HIGH, block_S)
                        else:
                            T.copy(
                                X[
                                    n,
                                    c_start + c_local,
                                    s_block * block_S: (s_block + 1) * block_S,
                                ],
                                x_cal,
                            )

                        T.copy(x_cal, x_sq)
                        T.tile.mul(x_sq, x_sq, x_sq)

                        T.reduce_sum(x_cal, chunk_scalar, dim=-1)
                        T.tile.add(sum_acc, sum_acc, chunk_scalar)

                        T.reduce_sum(x_sq, chunk_scalar, dim=-1)
                        T.tile.add(sum_sq_acc, sum_sq_acc, chunk_scalar)

                T.tile.fill(total_t, float(total_elts))
                T.tile.div(mean_val, sum_acc, total_t)

                T.tile.div(var_val, sum_sq_acc, total_t)
                T.tile.mul(chunk_scalar, mean_val, mean_val)
                T.tile.sub(var_val, var_val, chunk_scalar)

                T.tile.fill(eps_t, eps)
                T.tile.add(var_val, var_val, eps_t)
                T.tile.sqrt(var_val, var_val)
                T.tile.fill(one_t, 1.0)
                T.tile.div(rstd_val, one_t, var_val)

                T.tile.broadcast(mean_s, mean_val)
                T.tile.broadcast(rstd_s, rstd_val)

                for c_local in T.serial(C_g):
                    if need_cast:
                        T.copy(
                            Gamma[c_start + c_local: c_start + c_local + 1, 0:1],
                            gamma_ub,
                        )
                        T.copy(
                            Beta[c_start + c_local: c_start + c_local + 1, 0:1],
                            beta_ub,
                        )
                        T.tile.cast(gamma_cal, gamma_ub, CAST_MODE_LOW2HIGH, 1)
                        T.tile.cast(beta_cal, beta_ub, CAST_MODE_LOW2HIGH, 1)
                    else:
                        T.copy(
                            Gamma[c_start + c_local: c_start + c_local + 1, 0:1],
                            gamma_cal,
                        )
                        T.copy(
                            Beta[c_start + c_local: c_start + c_local + 1, 0:1],
                            beta_cal,
                        )

                    T.tile.broadcast(gamma_s, gamma_cal)
                    T.tile.broadcast(beta_s, beta_cal)

                    for s_block in T.serial(s_num):
                        if need_cast:
                            T.copy(
                                X[
                                    n,
                                    c_start + c_local,
                                    s_block * block_S: (s_block + 1) * block_S,
                                ],
                                x_ub,
                            )
                            T.tile.cast(x_cal, x_ub, CAST_MODE_LOW2HIGH, block_S)
                        else:
                            T.copy(
                                X[
                                    n,
                                    c_start + c_local,
                                    s_block * block_S: (s_block + 1) * block_S,
                                ],
                                x_cal,
                            )

                        T.tile.sub(x_cal, x_cal, mean_s)
                        T.tile.mul(x_cal, x_cal, rstd_s)
                        T.tile.mul(x_cal, x_cal, gamma_s)
                        T.tile.add(x_cal, x_cal, beta_s)

                        if need_cast:
                            T.tile.cast(x_ub, x_cal, CAST_MODE_HIGH2LOW, block_S)
                            T.copy(
                                x_ub,
                                Y[
                                    n,
                                    c_start + c_local,
                                    s_block * block_S: (s_block + 1) * block_S,
                                ],
                            )
                        else:
                            T.copy(
                                x_cal,
                                Y[
                                    n,
                                    c_start + c_local,
                                    s_block * block_S: (s_block + 1) * block_S,
                                ],
                            )

    return main


def torch_group_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                     num_groups: int, epsilon: float) -> torch.Tensor:
    """PyTorch golden reference for GroupNorm."""
    return torch.nn.functional.group_norm(
        input=x, num_groups=num_groups, weight=gamma, bias=beta, eps=epsilon,
    )


def run_group_norm(case_id: int, N: int, C: int, spatial_shape: tuple, dtype_str: str,
                   num_groups: int, epsilon: float, value_range: tuple,
                   block_S: int = None):
    """Run a single GroupNorm test case."""
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[dtype_str]

    spatial_dims = list(spatial_shape)
    S = int(np.prod(spatial_dims))

    if block_S is None:
        block_S = 256

    x_shape = (N, C) + tuple(spatial_dims)
    x = torch.randn(x_shape, dtype=torch_dtype, device="npu")
    gamma = torch.randn(C, dtype=torch_dtype, device="npu")
    beta = torch.randn(C, dtype=torch_dtype, device="npu")

    x_flat = x.reshape(N, C, S)
    gamma_flat = gamma.reshape(C, 1).contiguous()
    beta_flat = beta.reshape(C, 1).contiguous()

    kernel = group_norm_kernel(N, C, S, num_groups, epsilon, block_S, dtype_str)

    y_flat = kernel(x_flat, gamma_flat, beta_flat)
    y = y_flat.reshape(x_shape)

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
    print("GroupNorm TileLang-Ascend 测试")
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
