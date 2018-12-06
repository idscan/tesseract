#if !(defined(__arm__) || defined(__arm64__))

#include "dotproductsse.h"
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
        int16x8_t sum = vmull_s8(vget_high_s8(a), vget_high_s8(b));
        sum = vmlal_s8(sum, vget_low_s8(a), vget_low_s8(b));
        return vaddl_s16(vget_low_s16(sum), vget_high_s16(sum));
    }

    // Computes and returns the dot product of the n-vectors u and v.
    // Uses Intel NEON intrinsics to access the SIMD instruction set.

    int32_t IntDotProductNEON(const int8_t* u, const int8_t* v, int n) {
        int max_offset = n - 16;
        int offset = 0;
        int32x4_t sum = vmovq_n_s32(0);
        if (offset <= max_offset) {
            offset = 16;
            auto packed1 = vld1q_s8(u);
            auto packed2 = vld1q_s8(v);
            sum = madd_16_of_8b_to_4_32b(packed1, packed2);
            while(offset <= max_offset) {
                    packed1 = vld1q_s8(u + offset);
                    packed2 = vld1q_s8(v + offset);
                    offset += 16;
                    auto sum_part = madd_16_of_8b_to_4_32b(packed1, packed2);
                    sum = vaddq_s32(sum, sum_part);
            }
        }
        int64x2_t temp = vpaddlq_s32(sum);
        int32_t result = static_cast<int32_t>(vpaddd_s64(temp));

        while (offset < n) {
            result += u[offset] * v[offset];
            ++offset;
        }
        return result;
    }

}  // namespace tesseract.
#endif
