"""测试 PTO vs AscendC 的标量除法精度"""
import tilelang
from tilelang import language as T
import torch

tilelang.disable_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def build_scalar_div(target):
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target=target)
    def kernel():
        @T.prim_func
        def main(
            x: T.Tensor((1,), "float32"),
            out: T.Tensor((1,), "float32"),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([1], "float32")
                    ob = T.alloc_ub([1], "float32")
                    with T.Scope("V"):
                        T.copy(x, xb)
                        T.barrier_all()
                        cnt = T.cast(4096, "float32")
                        T.tile.div(ob, xb, cnt)
                        T.barrier_all()
                        T.copy(ob, out)
        return main
    return kernel


x = torch.tensor([100.0], dtype=torch.float32, device="npu")
expected = 100.0 / 4096.0

print(f"输入: {x.item()}")
print(f"期望输出: {expected}")

for tgt in ["ascendc", "pto"]:
    try:
        kern = build_scalar_div(tgt)()
        out = kern(x)
        torch.npu.synchronize()
        result = out.item()
        error = abs(result - expected)
        print(f"[{tgt:8s}] 输出: {result:.10f}, 误差: {error:.2e}")
    except Exception as e:
        print(f"[{tgt:8s}] ERROR: {str(e)[:100]}")
