"""
逐步隔离精度问题：从简单到复杂，精确定位 PTO 后端的精度问题根因
"""
import tilelang
from tilelang import language as T
import torch

tilelang.disable_cache()
torch.manual_seed(42)

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

CPG, BS = 4, 512

def test_stage1_reduce_single():
    """阶段1: 单独 reduce_sum [4, 512] -> [4]"""
    print("\n" + "="*60)
    print("阶段1: 单独 reduce_sum [4, 512] -> [4]")
    print("="*60)
    
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs)
    def kernel_ascendc(cpg, block_S):
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"), 
                 out: T.Tensor((cpg,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    ob = T.alloc_ub([cpg], "float32")
                    with T.Scope("V"):
                        T.copy(x, xb)
                        T.barrier_all()
                        T.reduce_sum(xb, ob, dim=-1)
                        T.barrier_all()
                        T.copy(ob, out)
        return main
    
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target="pto")
    def kernel_pto(cpg, block_S):
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"), 
                 out: T.Tensor((cpg,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    ob = T.alloc_ub([cpg], "float32")
                    with T.Scope("V"):
                        T.copy(x, xb)
                        T.barrier_all()
                        T.reduce_sum(xb, ob, dim=-1)
                        T.barrier_all()
                        T.copy(ob, out)
        return main
    
    x = torch.randn(CPG, BS, dtype=torch.float32, device="npu")
    golden = x.float().cpu().sum(dim=-1)
    
    for name, kern_fn in [("ascendc", kernel_ascendc), ("pto", kernel_pto)]:
        try:
            kern = kern_fn(CPG, BS)
            out = kern(x)
            torch.npu.synchronize()
            result = out.float().cpu()
            error = (result - golden).abs().max().item()
            status = "✓ PASS" if error < 1e-4 else "✗ FAIL"
            print(f"[{name:8s}] {status} max_error={error:.2e}")
            print(f"         result={result.tolist()}")
            print(f"         golden={golden.tolist()}")
        except Exception as e:
            print(f"[{name:8s}] ✗ ERROR: {str(e)[:100]}")


def test_stage2_reduce_chain():
    """阶段2: 链式 reduce_sum [4, 512] -> [4] -> [1]"""
    print("\n" + "="*60)
    print("阶段2: 链式 reduce_sum [4, 512] -> [4] -> [1]")
    print("="*60)
    
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs)
    def kernel_ascendc(cpg, block_S):
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"), 
                 out: T.Tensor((1,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    row = T.alloc_ub([cpg], "float32")
                    total = T.alloc_ub([1], "float32")
                    with T.Scope("V"):
                        T.copy(x, xb)
                        T.barrier_all()
                        T.reduce_sum(xb, row, dim=-1)
                        T.reduce_sum(row, total, dim=-1)
                        T.barrier_all()
                        T.copy(total, out)
        return main
    
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target="pto")
    def kernel_pto(cpg, block_S):
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"), 
                 out: T.Tensor((1,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    row = T.alloc_ub([cpg], "float32")
                    total = T.alloc_ub([1], "float32")
                    with T.Scope("V"):
                        T.copy(x, xb)
                        T.barrier_all()
                        T.reduce_sum(xb, row, dim=-1)
                        T.reduce_sum(row, total, dim=-1)
                        T.barrier_all()
                        T.copy(total, out)
        return main
    
    x = torch.randn(CPG, BS, dtype=torch.float32, device="npu")
    golden = x.float().cpu().sum().item()
    
    for name, kern_fn in [("ascendc", kernel_ascendc), ("pto", kernel_pto)]:
        try:
            kern = kern_fn(CPG, BS)
            out = kern(x)
            torch.npu.synchronize()
            result = out.float().cpu().item()
            error = abs(result - golden)
            status = "✓ PASS" if error < 1e-2 else "✗ FAIL"
            print(f"[{name:8s}] {status} error={error:.2e}")
            print(f"         result={result:.6f}")
            print(f"         golden={golden:.6f}")
        except Exception as e:
            print(f"[{name:8s}] ✗ ERROR: {str(e)[:100]}")


def test_stage3_reduce_and_scalar():
    """阶段3: reduce + 标量运算 (mean/var/std)"""
    print("\n" + "="*60)
    print("阶段3: reduce + 标量运算 (mean/var/std)")
    print("="*60)
    
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs)
    def kernel_ascendc(cpg, block_S, eps):
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"), 
                 out: T.Tensor((3,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    sum_a = T.alloc_ub([cpg, block_S], "float32")
                    sum_sq = T.alloc_ub([cpg, block_S], "float32")
                    row = T.alloc_ub([cpg], "float32")
                    row_sq = T.alloc_ub([cpg], "float32")
                    total = T.alloc_ub([1], "float32")
                    total_sq = T.alloc_ub([1], "float32")
                    mean_val = T.alloc_ub([1], "float32")
                    var_val = T.alloc_ub([1], "float32")
                    std_val = T.alloc_ub([1], "float32")
                    ob = T.alloc_ub([3], "float32")
                    with T.Scope("V"):
                        T.copy(x, xb)
                        T.tile.fill(sum_a, 0.0)
                        T.tile.fill(sum_sq, 0.0)
                        T.tile.add(sum_a, sum_a, xb)
                        T.tile.mul(xb, xb, xb)
                        T.tile.add(sum_sq, sum_sq, xb)
                        T.barrier_all()
                        T.reduce_sum(sum_a, row, dim=-1)
                        T.reduce_sum(sum_sq, row_sq, dim=-1)
                        T.reduce_sum(row, total, dim=-1)
                        T.reduce_sum(row_sq, total_sq, dim=-1)
                        cnt = T.cast(cpg * block_S, "float32")
                        T.tile.div(total, total, cnt)
                        T.tile.div(total_sq, total_sq, cnt)
                        T.tile.mul(mean_val, total, total)
                        T.tile.sub(var_val, total_sq, mean_val)
                        eps_v = T.cast(eps, "float32")
                        T.tile.add(var_val, var_val, eps_v)
                        T.tile.sqrt(std_val, var_val)
                        T.barrier_all()
                        T.copy(total, ob[0:1])
                        T.copy(total_sq, ob[1:2])
                        T.copy(std_val, ob[2:3])
                        T.copy(ob, out)
        return main
    
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target="pto")
    def kernel_pto(cpg, block_S, eps):
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"), 
                 out: T.Tensor((3,), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    sum_a = T.alloc_ub([cpg, block_S], "float32")
                    sum_sq = T.alloc_ub([cpg, block_S], "float32")
                    row = T.alloc_ub([cpg], "float32")
                    row_sq = T.alloc_ub([cpg], "float32")
                    total = T.alloc_ub([1], "float32")
                    total_sq = T.alloc_ub([1], "float32")
                    mean_val = T.alloc_ub([1], "float32")
                    var_val = T.alloc_ub([1], "float32")
                    std_val = T.alloc_ub([1], "float32")
                    ob = T.alloc_ub([3], "float32")
                    with T.Scope("V"):
                        T.copy(x, xb)
                        T.tile.fill(sum_a, 0.0)
                        T.tile.fill(sum_sq, 0.0)
                        T.tile.add(sum_a, sum_a, xb)
                        T.tile.mul(xb, xb, xb)
                        T.tile.add(sum_sq, sum_sq, xb)
                        T.barrier_all()
                        T.reduce_sum(sum_a, row, dim=-1)
                        T.reduce_sum(sum_sq, row_sq, dim=-1)
                        T.reduce_sum(row, total, dim=-1)
                        T.reduce_sum(row_sq, total_sq, dim=-1)
                        cnt = T.cast(cpg * block_S, "float32")
                        T.tile.div(total, total, cnt)
                        T.tile.div(total_sq, total_sq, cnt)
                        T.tile.mul(mean_val, total, total)
                        T.tile.sub(var_val, total_sq, mean_val)
                        eps_v = T.cast(eps, "float32")
                        T.tile.add(var_val, var_val, eps_v)
                        T.tile.sqrt(std_val, var_val)
                        T.barrier_all()
                        T.copy(total, ob[0:1])
                        T.copy(total_sq, ob[1:2])
                        T.copy(std_val, ob[2:3])
                        T.copy(ob, out)
        return main
    
    x = torch.randn(CPG, BS, dtype=torch.float32, device="npu")
    eps = 1e-5
    xf = x.float().cpu()
    cnt = CPG * BS
    g_mean = xf.sum().item() / cnt
    g_var = (xf * xf).sum().item() / cnt - g_mean * g_mean + eps
    g_std = g_var ** 0.5
    golden = [g_mean, g_mean * g_mean + g_var - eps, g_std]
    
    for name, kern_fn in [("ascendc", kernel_ascendc), ("pto", kernel_pto)]:
        try:
            kern = kern_fn(CPG, BS, eps)
            out = kern(x)
            torch.npu.synchronize()
            result = out.float().cpu().tolist()
            errors = [abs(result[i] - golden[i]) for i in range(3)]
            max_err = max(errors)
            status = "✓ PASS" if max_err < 1e-2 else "✗ FAIL"
            print(f"[{name:8s}] {status} max_error={max_err:.2e}")
            print(f"         mean={result[0]:.6f} (golden={golden[0]:.6f}, err={errors[0]:.2e})")
            print(f"         var ={result[1]:.6f} (golden={golden[1]:.6f}, err={errors[1]:.2e})")
            print(f"         std ={result[2]:.6f} (golden={golden[2]:.6f}, err={errors[2]:.2e})")
        except Exception as e:
            print(f"[{name:8s}] ✗ ERROR: {str(e)[:100]}")


def test_stage4_full_normalize():
    """阶段4: reduce + broadcast + normalize"""
    print("\n" + "="*60)
    print("阶段4: reduce + broadcast + normalize")
    print("="*60)
    
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs)
    def kernel_ascendc(cpg, block_S, eps):
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"), 
                 out: T.Tensor((cpg, block_S), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    sum_a = T.alloc_ub([cpg, block_S], "float32")
                    sum_sq = T.alloc_ub([cpg, block_S], "float32")
                    row = T.alloc_ub([cpg], "float32")
                    row_sq = T.alloc_ub([cpg], "float32")
                    total = T.alloc_ub([1], "float32")
                    total_sq = T.alloc_ub([1], "float32")
                    mean_val = T.alloc_ub([1], "float32")
                    var_val = T.alloc_ub([1], "float32")
                    std_val = T.alloc_ub([1], "float32")
                    mean_col = T.alloc_ub([cpg, 1], "float32")
                    std_col = T.alloc_ub([cpg, 1], "float32")
                    mean_bc = T.alloc_ub([cpg, block_S], "float32")
                    std_bc = T.alloc_ub([cpg, block_S], "float32")
                    data_cal = T.alloc_ub([cpg, block_S], "float32")
                    with T.Scope("V"):
                        T.copy(x, xb)
                        T.tile.fill(sum_a, 0.0)
                        T.tile.fill(sum_sq, 0.0)
                        T.tile.add(sum_a, sum_a, xb)
                        T.tile.mul(xb, xb, xb)
                        T.tile.add(sum_sq, sum_sq, xb)
                        T.barrier_all()
                        T.reduce_sum(sum_a, row, dim=-1)
                        T.reduce_sum(sum_sq, row_sq, dim=-1)
                        T.reduce_sum(row, total, dim=-1)
                        T.reduce_sum(row_sq, total_sq, dim=-1)
                        cnt = T.cast(cpg * block_S, "float32")
                        T.tile.div(total, total, cnt)
                        T.tile.div(total_sq, total_sq, cnt)
                        T.tile.mul(mean_val, total, total)
                        T.tile.sub(var_val, total_sq, mean_val)
                        eps_v = T.cast(eps, "float32")
                        T.tile.add(var_val, var_val, eps_v)
                        T.tile.sqrt(std_val, var_val)
                        T.tile.fill(mean_col, total)
                        T.tile.broadcast(mean_bc, mean_col)
                        T.tile.fill(std_col, std_val)
                        T.tile.broadcast(std_bc, std_col)
                        T.copy(x, data_cal)
                        T.tile.sub(data_cal, data_cal, mean_bc)
                        T.tile.div(data_cal, data_cal, std_bc)
                        T.barrier_all()
                        T.copy(data_cal, out)
        return main
    
    @tilelang.jit(out_idx=[1], pass_configs=pass_configs, target="pto")
    def kernel_pto(cpg, block_S, eps):
        @T.prim_func
        def main(x: T.Tensor((cpg, block_S), "float32"), 
                 out: T.Tensor((cpg, block_S), "float32")):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                if vid == 0:
                    xb = T.alloc_ub([cpg, block_S], "float32")
                    sum_a = T.alloc_ub([cpg, block_S], "float32")
                    sum_sq = T.alloc_ub([cpg, block_S], "float32")
                    row = T.alloc_ub([cpg], "float32")
                    row_sq = T.alloc_ub([cpg], "float32")
                    total = T.alloc_ub([1], "float32")
                    total_sq = T.alloc_ub([1], "float32")
                    mean_val = T.alloc_ub([1], "float32")
                    var_val = T.alloc_ub([1], "float32")
                    std_val = T.alloc_ub([1], "float32")
                    mean_col = T.alloc_ub([cpg, 1], "float32")
                    std_col = T.alloc_ub([cpg, 1], "float32")
                    mean_bc = T.alloc_ub([cpg, block_S], "float32")
                    std_bc = T.alloc_ub([cpg, block_S], "float32")
                    data_cal = T.alloc_ub([cpg, block_S], "float32")
                    with T.Scope("V"):
                        T.copy(x, xb)
                        T.tile.fill(sum_a, 0.0)
                        T.tile.fill(sum_sq, 0.0)
                        T.tile.add(sum_a, sum_a, xb)
                        T.tile.mul(xb, xb, xb)
                        T.tile.add(sum_sq, sum_sq, xb)
                        T.barrier_all()
                        T.reduce_sum(sum_a, row, dim=-1)
                        T.reduce_sum(sum_sq, row_sq, dim=-1)
                        T.reduce_sum(row, total, dim=-1)
                        T.reduce_sum(row_sq, total_sq, dim=-1)
                        cnt = T.cast(cpg * block_S, "float32")
                        T.tile.div(total, total, cnt)
                        T.tile.div(total_sq, total_sq, cnt)
                        T.tile.mul(mean_val, total, total)
                        T.tile.sub(var_val, total_sq, mean_val)
                        eps_v = T.cast(eps, "float32")
                        T.tile.add(var_val, var_val, eps_v)
                        T.tile.sqrt(std_val, var_val)
                        T.tile.fill(mean_col, total)
                        T.tile.broadcast(mean_bc, mean_col)
                        T.tile.fill(std_col, std_val)
                        T.tile.broadcast(std_bc, std_col)
                        T.copy(x, data_cal)
                        T.tile.sub(data_cal, data_cal, mean_bc)
                        T.tile.div(data_cal, data_cal, std_bc)
                        T.barrier_all()
                        T.copy(data_cal, out)
        return main
    
    x = torch.randn(CPG, BS, dtype=torch.float32, device="npu")
    eps = 1e-5
    xf = x.float().cpu()
    cnt = CPG * BS
    g_mean = xf.sum().item() / cnt
    g_var = (xf * xf).sum().item() / cnt - g_mean * g_mean + eps
    g_std = g_var ** 0.5
    golden = ((xf - g_mean) / g_std)
    
    for name, kern_fn in [("ascendc", kernel_ascendc), ("pto", kernel_pto)]:
        try:
            kern = kern_fn(CPG, BS, eps)
            out = kern(x)
            torch.npu.synchronize()
            result = out.float().cpu()
            error = (result - golden).abs()
            max_err = error.max().item()
            mean_err = error.mean().item()
            status = "✓ PASS" if max_err < 1e-2 else "✗ FAIL"
            print(f"[{name:8s}] {status} max_error={max_err:.2e} mean_error={mean_err:.2e}")
            print(f"         result[0,:4]={result[0,:4].tolist()}")
            print(f"         golden[0,:4]={golden[0,:4].tolist()}")
        except Exception as e:
            print(f"[{name:8s}] ✗ ERROR: {str(e)[:100]}")


if __name__ == "__main__":
    print("逐步隔离 PTO 后端精度问题")
    print("配置: CPG=4, BS=512, dtype=float32")
    
    test_stage1_reduce_single()
    test_stage2_reduce_chain()
    test_stage3_reduce_and_scalar()
    test_stage4_full_normalize()
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
