///////////////////////////////////////////////////////////////////////

#if !defined(__aarch64__)

#include "dotproductsse.h"
#include <cstdio>
#include <cstdlib>

namespace tesseract {
    int32_t IntDotProductNEON(const int8_t* u, const int8_t* v, int n) {
        fprintf(stderr, "IntDotProductNEON can't be used\n");
        abort();
    }
}  // namespace tesseract

#else  // !defined(__aarch64__)

#include "dotproductneon.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <arm_neon.h>
#include <iostream>


static const int16x8_t zero = vmovq_n_s16(0);
static const int32x4_t zero_32 = vmovq_n_s32(0);


inline int16x8_t unpack_low_8(int16x8_t a, int16x8_t b) {
    int8x8_t a1 = vreinterpret_s8_s16(vget_low_s16(vreinterpretq_s16_s32(a)));
    int8x8_t b1 = vreinterpret_s8_s16(vget_low_s16(vreinterpretq_s16_s32(b)));
    int8x8x2_t result = vzip_s8(a1, b1);
    return vreinterpretq_s32_s8(vcombine_s8(result.val[0], result.val[1]));
}

inline int16x8_t sign_extened_8to16(int16x8_t a) { // eq. _mm_cvtepi8_epi16
 //   auto x = __sxtb16(__a);
    int16x8_t sign = vcgtq_s8(zero, a);
    return unpack_low_8(a, sign);
}

//static uint32_t __SXTB16(uint32_t x){
//    return ((uint32_t)(((((q31_t)x << 24) >> 24) & (q31_t)0x0000FFFF) |
//                       ((((q31_t)x <<  8) >>  8) & (q31_t)0xFFFF0000)  ));
//}

uint8_t udiv_function(uint32_t in)
{
    uint8_t result;
    asm volatile (
                  "sxtb %[out], %[num]"
                  :[out] "=r" (result)
                  :[num] "r" (in)
                  :
                  );
    return result;
    
    
}

inline int16x8_t load_8_of_8_b(const int8_t* a) { //_mm_loadl_epi64(reinterpret_cast<const __m128i*>
    return vcombine_s32(vld1_s32(reinterpret_cast<const int32_t*>(a)), vcreate_s32(0));
}

inline int32x4_t madd_8_of_16b_to_4_32b(const int16x8_t& a, const int16x8_t& b) { //_mm_madd_epi16
    //int32x4_t sum = zero_32;
    //sum = vmlal_s16(sum, vget_low_u64(a), vget_low_u64(b)); //TODO: use asm VMLAL
    //sum = vmlal_s16(sum, vget_high_u64(a), vget_high_u64(b));
    int32x4_t sum1 = vmull_s16(vget_low_u64(a), vget_low_u64(b));
    int32x4_t sum2 = vmull_s16(vget_high_u64(a), vget_high_u64(b));
    return vaddq_s32(sum1, sum2);
}

namespace tesseract {
    
// Computes and returns the dot product of the n-vectors u and v.
// Uses Intel NEON intrinsics to access the SIMD instruction set.
int32_t IntDotProductNEON(const int8_t* u, const int8_t* v, int n) {

  int max_offset = n - 8;
  int offset = 0;
  // Accumulate a set of 4 32-bit sums in sum, by loading 8 pairs of 8-bit
  // values, extending to 16 bit, multiplying to make 32 bit results.
  int16x8_t sum;
  if (offset <= max_offset) {
    int16x8_t packed1 = load_8_of_8_b(u);
    int16x8_t packed2 = load_8_of_8_b(v);

    sum = sign_extened_8to16(packed1);
    packed2 = sign_extened_8to16(packed2);

    sum = madd_8_of_16b_to_4_32b(sum, packed2);
    int nb_itr = max_offset/8 + 1;
    offset = 8; //max_offset + 8 - (max_offset % 8);
    while(true) {
      --nb_itr;
      if (nb_itr == 0) break;
      //NO_PROFILE(while (offset <= max_offset) {)
      packed1 = load_8_of_8_b(u + offset);
      packed2 = load_8_of_8_b(v + offset);
      packed1 = sign_extened_8to16(packed1);
      packed2 = sign_extened_8to16(packed2);
      packed1 = madd_8_of_16b_to_4_32b(packed1, packed2);
      sum = vaddq_s32(sum, packed1);
      offset += 8;
    }
  }
  int64x2_t temp = vpaddlq_s32(sum);
  int64x1_t temp2 = vadd_s64(vget_high_s64(temp), vget_low_s64(temp));
  int32_t result = vget_lane_s32(temp2, 0);
  while (offset < n) {
      result += u[offset] * v[offset];
      ++offset;
  }
  return result;
}

}  // namespace tesseract.
#endif
