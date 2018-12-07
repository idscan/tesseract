#if !(defined(__arm__) || defined(__arm64__))

#include "dotproductneon.h"
#include <cstdio>
#include <cstdlib>

namespace tesseract {
    int32_t IntDotProductNEON(const int8_t* u, const int8_t* v, int n) {
        fprintf(stderr, "IntDotProductNEON can't be used\n");
        abort();
    }
}  // namespace tesseract

#else  // !(defined(__arm__) || defined(__arm64__))

#include "dotproductneon.h"
#include <arm_neon.h>

namespace tesseract {
    
    inline int32x4_t madd_16_of_8b_to_4_32b(const int8x16_t& a, const int8x16_t& b) {
        int16x8_t mul1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
        int16x8_t mul2 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        int16x8_t sum = vaddq_s16(mul1, mul2);
        return vpaddlq_s16(sum);
    }

    int32_t IntDotProductNEON(const int8_t* u, const int8_t* v, int n) {
        int max_offset = n - 16;
        int offset = 0;
        int32x4_t sum_4_32 = vmovq_n_s32(0);
        if (offset <= max_offset) {
            offset = 16;
            int8x16_t packed1 = vld1q_s8(u);
            int8x16_t packed2 = vld1q_s8(v);
            sum_4_32 = madd_16_of_8b_to_4_32b(packed1, packed2);
            while(offset <= max_offset) {
                    packed1 = vld1q_s8(u + offset);
                    packed2 = vld1q_s8(v + offset);
                    offset += 16;
                    int32x4_t sum_4_32_part = madd_16_of_8b_to_4_32b(packed1, packed2);
                    sum_4_32 = vaddq_s32(sum_4_32, sum_4_32_part);
            }
        }
        int64x2_t sum_2_64 = vpaddlq_s32(sum_4_32);
        int32x2_t sum_2_32 = vmovn_s64(sum_2_64);
        int32_t result = vget_lane_s32(sum_2_32, 0) + vget_lane_s32(sum_2_32, 1);

        while (offset < n) {
            result += u[offset] * v[offset];
            ++offset;
        }
        return result;
    }

}  // namespace tesseract.
#endif
