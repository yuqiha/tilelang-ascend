#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR x_handle,  GM_ADDR gamma_handle,  GM_ADDR beta_handle,  GM_ADDR y_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> x;
  x.SetGlobalBuffer((__gm__ float*)x_handle);
  AscendC::GlobalTensor<float> gamma;
  gamma.SetGlobalBuffer((__gm__ float*)gamma_handle);
  AscendC::GlobalTensor<float> beta;
  beta.SetGlobalBuffer((__gm__ float*)beta_handle);
  AscendC::GlobalTensor<float> y;
  y.SetGlobalBuffer((__gm__ float*)y_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto sum_a = ascend_ub.GetWithOffset<float>(2048, 0);
  auto sum_sq_a = ascend_ub.GetWithOffset<float>(2048, 8192);
  auto data_buf_p1 = ascend_ub.GetWithOffset<float>(4096, 16384);
  auto sum_row = ascend_ub.GetWithOffset<float>(4, 40960);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(8192, 40992);
  auto sum_sq_row = ascend_ub.GetWithOffset<float>(4, 49184);
  auto total = ascend_ub.GetWithOffset<float>(1, 49216);
  auto total_sq = ascend_ub.GetWithOffset<float>(1, 49248);
  auto mean_sq_val = ascend_ub.GetWithOffset<float>(1, 49280);
  auto var_val = ascend_ub.GetWithOffset<float>(1, 49312);
  auto std_val = ascend_ub.GetWithOffset<float>(1, 49344);
  auto out_a = ascend_ub.GetWithOffset<float>(2048, 49376);
  auto out_b = ascend_ub.GetWithOffset<float>(2048, 57568);
  auto data_cal = ascend_ub.GetWithOffset<float>(2048, 32768);
  auto vid = AscendC::GetSubBlockIdx();
  if (vid == 0) {
    if ASCEND_IS_AIV {
      tl::ascend::Fill<float>(sum_a[0], 0.000000e+00f, 2048);
      tl::ascend::Fill<float>(sum_sq_a[0], 0.000000e+00f, 2048);
      tl::ascend::copy_gm_to_ub<float, 512, 4>(data_buf_p1[0], x[(cid * 4096)], 1024, 4, 512, 0.000000e+00f);
      AscendC::PipeBarrier<PIPE_ALL>();
      for (int32_t si = 0; si < 2; ++si) {
        if (si < 1) {
          tl::ascend::copy_gm_to_ub<float, 512, 4>(data_buf_p1[((si * 2048) + 2048)], x[(((cid * 4096) + (si * 512)) + 512)], 1024, 4, 512, 0.000000e+00f);
        }
        tl::ascend::copy_ub_to_ub<float, float, 2048>(data_cal[0], data_buf_p1[(si * 2048)], 4, 512, 512, 4, 512, 512);
        AscendC::Add(sum_a[0], sum_a[0], data_cal[0], 2048);
        AscendC::Mul(data_cal[0], data_cal[0], data_cal[0], 2048);
        AscendC::Add(sum_sq_a[0], sum_sq_a[0], data_cal[0], 2048);
        AscendC::PipeBarrier<PIPE_ALL>();
      }
      tl::ascend::reduce_sum<float,  4,  512,  -1>(sum_row[0], sum_a[0], tmp_ub[0], true);
      tl::ascend::reduce_sum<float,  4,  512,  -1>(sum_sq_row[0], sum_sq_a[0], tmp_ub[0], true);
      tl::ascend::reduce_sum<float,  1,  4,  -1>(total[0], sum_row[0], tmp_ub[0], true);
      tl::ascend::reduce_sum<float,  1,  4,  -1>(total_sq[0], sum_sq_row[0], tmp_ub[0], true);
      AscendC::Muls(total[0], total[0], 1.0f / 4.096000e+03f, 1);
      AscendC::Muls(total_sq[0], total_sq[0], 1.0f / 4.096000e+03f, 1);
      AscendC::Mul(mean_sq_val[0], total[0], total[0], 1);
      AscendC::Sub(var_val[0], total_sq[0], mean_sq_val[0], 1);
      {
      AscendC::Adds(var_val[0], var_val[0], 1.000000e-05f, 1);
      }
      AscendC::Sqrt(std_val[0], var_val[0], 1);
      tl::ascend::Fill<float>(out_a[0], total.GetValue(0), 2048);
      tl::ascend::Fill<float>(out_b[0], total_sq.GetValue(0), 2048);
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::copy_ub_to_gm<float, 512, 4>(y[(cid * 4096)], out_a[0], 1024, 4, 512);
      tl::ascend::copy_ub_to_gm<float, 512, 4>(y[((cid * 4096) + 512)], out_b[0], 1024, 4, 512);
    }
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* x_handle, uint8_t* gamma_handle, uint8_t* beta_handle, uint8_t* y_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<4, nullptr, stream>>>(x_handle, gamma_handle, beta_handle, y_handle, fftsAddr);
}
