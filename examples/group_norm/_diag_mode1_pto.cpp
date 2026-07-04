#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

extern "C" __global__ AICORE void main_kernel(__gm__ uint8_t *x_handle_raw, __gm__ uint8_t *gamma_handle_raw, __gm__ uint8_t *beta_handle_raw, __gm__ uint8_t *y_handle_raw, uint64_t ffts_Addr) {
  __gm__ float *x_handle = reinterpret_cast<__gm__ float *>(x_handle_raw);
  __gm__ float *gamma_handle = reinterpret_cast<__gm__ float *>(gamma_handle_raw);
  __gm__ float *beta_handle = reinterpret_cast<__gm__ float *>(beta_handle_raw);
  __gm__ float *y_handle = reinterpret_cast<__gm__ float *>(y_handle_raw);
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  tl::ascend_pto::TileUbDataND<float, 4, 512, 4, 512> sum_a;
  TASSIGN(sum_a, 0);
  tl::ascend_pto::TileUbDataND<float, 4, 512, 4, 512> sum_sq_a;
  TASSIGN(sum_sq_a, 8192);
  tl::ascend_pto::TileUbDataND<float, 8, 512, 8, 512> data_buf_p1;
  TASSIGN(data_buf_p1, 16384);
  tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 4> sum_row;
  TASSIGN(sum_row, 40960);
  tl::ascend_pto::TileUbDataND<uint8_t, 1, 4096, 1, 4096> tmp_ub;
  TASSIGN(tmp_ub, 40992);
  tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 4> sum_sq_row;
  TASSIGN(sum_sq_row, 45088);
  tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> total;
  TASSIGN(total, 45120);
  tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> total_sq;
  TASSIGN(total_sq, 45152);
  tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> mean_sq_val;
  TASSIGN(mean_sq_val, 45184);
  tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> var_val;
  TASSIGN(var_val, 45216);
  tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> std_val;
  TASSIGN(std_val, 45248);
  tl::ascend_pto::TileUbDataND<float, 4, 512, 4, 512> out_a;
  TASSIGN(out_a, 45280);
  tl::ascend_pto::TileUbDataND<float, 4, 512, 4, 512> out_b;
  TASSIGN(out_b, 53472);
  tl::ascend_pto::TileUbDataND<float, 4, 512, 4, 512> data_cal;
  TASSIGN(data_cal, 32768);
  auto vid = get_subblockid();
  if (vid == 0) {
#if defined(__DAV_C220_VEC__)
      set_mask_norm();
      set_vector_mask(-1, -1);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      TEXPANDS(sum_a, 0.000000e+00f);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      TEXPANDS(sum_sq_a, 0.000000e+00f);
      tl::ascend_pto::copy_gm_to_ub_dynamic<float, float, 1, 1, 1, 4, 512, 16384, 8192, 4096, 1024, 1, 8, 512, pto::PadValue::Null>(x_handle + (cid * 4096), pto::Shape<1, 1, 1, 4, 512>(), pto::Stride<16384, 8192, 4096, 1024, 1>(), 16384, 0, 4, 512);
      pipe_barrier(PIPE_ALL);

  for (int32_t si = 0; si < 2; ++si) {
        if (si < 1) {
          tl::ascend_pto::copy_gm_to_ub_dynamic<float, float, 1, 1, 1, 4, 512, 16384, 8192, 4096, 1024, 1, 8, 512, pto::PadValue::Null>(x_handle + (((cid * 4096) + (si * 512)) + 512), pto::Shape<1, 1, 1, 4, 512>(), pto::Stride<16384, 8192, 4096, 1024, 1>(), 16384, ((si * 2048) + 2048), 4, 512);
        }
        tl::ascend_pto::TileUbDataND<float, 4, 512, 4, 512> data_buf_p1_temp_0;
        TASSIGN(data_buf_p1_temp_0, 16384 + (si * 2048) * 4);
        TMOV(data_cal, data_buf_p1_temp_0);
        TADD(sum_a, sum_a, data_cal);
        TMUL(data_cal, data_cal, data_cal);
        TADD(sum_sq_a, sum_sq_a, data_cal);
        pipe_barrier(PIPE_ALL);
      }
      tl::ascend_pto::TileUbDataDN<float, 8, 1, 4, 1> sum_row_temp_0;
      TASSIGN(sum_row_temp_0, 40960 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 4, 256, 4, 256> tmp_ub_temp_0;
      TASSIGN(tmp_ub_temp_0, 40992 + 0 * 4);
      TROWSUM(sum_row_temp_0, sum_a, tmp_ub_temp_0);
      tl::ascend_pto::TileUbDataDN<float, 8, 1, 4, 1> sum_sq_row_temp_0;
      TASSIGN(sum_sq_row_temp_0, 45088 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 4, 256, 4, 256> tmp_ub_temp_1;
      TASSIGN(tmp_ub_temp_1, 40992 + 0 * 4);
      TROWSUM(sum_sq_row_temp_0, sum_sq_a, tmp_ub_temp_1);
      tl::ascend_pto::TileUbDataDN<float, 8, 1, 1, 1> total_temp_0;
      TASSIGN(total_temp_0, 45120 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 4> sum_row_temp_1;
      TASSIGN(sum_row_temp_1, 40960 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 8> tmp_ub_temp_2;
      TASSIGN(tmp_ub_temp_2, 40992 + 0 * 4);
      TROWSUM(total_temp_0, sum_row_temp_1, tmp_ub_temp_2);
      tl::ascend_pto::TileUbDataDN<float, 8, 1, 1, 1> total_sq_temp_0;
      TASSIGN(total_sq_temp_0, 45152 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 4> sum_sq_row_temp_1;
      TASSIGN(sum_sq_row_temp_1, 45088 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 8> tmp_ub_temp_3;
      TASSIGN(tmp_ub_temp_3, 40992 + 0 * 4);
      TROWSUM(total_sq_temp_0, sum_sq_row_temp_1, tmp_ub_temp_3);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> total_temp_1;
      TASSIGN(total_temp_1, 45120 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> total_temp_2;
      TASSIGN(total_temp_2, 45120 + 0 * 4);
      TMULS(total_temp_2, total_temp_1, 1.0f / 4.096000e+03f);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> total_sq_temp_1;
      TASSIGN(total_sq_temp_1, 45152 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> total_sq_temp_2;
      TASSIGN(total_sq_temp_2, 45152 + 0 * 4);
      TMULS(total_sq_temp_2, total_sq_temp_1, 1.0f / 4.096000e+03f);
      pipe_barrier(PIPE_ALL);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> total_temp_3;
      TASSIGN(total_temp_3, 45120 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> total_temp_4;
      TASSIGN(total_temp_4, 45120 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> mean_sq_val_temp_0;
      TASSIGN(mean_sq_val_temp_0, 45184 + 0 * 4);
      TMUL(mean_sq_val_temp_0, total_temp_3, total_temp_4);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> total_sq_temp_3;
      TASSIGN(total_sq_temp_3, 45152 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> mean_sq_val_temp_1;
      TASSIGN(mean_sq_val_temp_1, 45184 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> var_val_temp_0;
      TASSIGN(var_val_temp_0, 45216 + 0 * 4);
      TSUB(var_val_temp_0, total_sq_temp_3, mean_sq_val_temp_1);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> var_val_temp_1;
      TASSIGN(var_val_temp_1, 45216 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> var_val_temp_2;
      TASSIGN(var_val_temp_2, 45216 + 0 * 4);
      TADDS(var_val_temp_2, var_val_temp_1, 1.000000e-05f);
      pipe_barrier(PIPE_ALL);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> var_val_temp_3;
      TASSIGN(var_val_temp_3, 45216 + 0 * 4);
      tl::ascend_pto::TileUbDataND<float, 1, 8, 1, 1> std_val_temp_0;
      TASSIGN(std_val_temp_0, 45248 + 0 * 4);
      TSQRT(std_val_temp_0, var_val_temp_3);
      pipe_barrier(PIPE_ALL);
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      TEXPANDS(out_a, total.GetValue(0));
      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      TEXPANDS(out_b, total_sq.GetValue(0));
      pipe_barrier(PIPE_ALL);
      tl::ascend_pto::copy_ub_to_gm_dynamic<float, float, 1, 1, 1, 4, 512, 16384, 8192, 4096, 1024, 1, 4, 512>(y_handle + (cid * 4096), pto::Shape<1, 1, 1, 4, 512>(), pto::Stride<16384, 8192, 4096, 1024, 1>(), 45280, 0, 4, 512);
      tl::ascend_pto::copy_ub_to_gm_dynamic<float, float, 1, 1, 1, 4, 512, 16384, 8192, 4096, 1024, 1, 4, 512>(y_handle + ((cid * 4096) + 512), pto::Shape<1, 1, 1, 4, 512>(), pto::Stride<16384, 8192, 4096, 1024, 1>(), 53472, 0, 4, 512);
#endif
  }
}

extern "C" void call(uint8_t *x_handle, uint8_t *gamma_handle, uint8_t *beta_handle, uint8_t *y_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    main_kernel<<<4, nullptr, stream>>>(x_handle, gamma_handle, beta_handle, y_handle, fftsAddr);
}
