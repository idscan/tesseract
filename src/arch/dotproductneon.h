///////////////////////////////////////////////////////////////////////

#ifndef TESSERACT_ARCH_DOTPRODUCTNEON_H_
#define TESSERACT_ARCH_DOTPRODUCTNEON_H_

#include <cstdint>      // for int32_t

namespace tesseract {

// Computes and returns the dot product of the n-vectors u and v.
// Uses NEON intrinsics to access the SIMD instruction set.
int32_t IntDotProductNEON(const int8_t* u, const int8_t* v, int n);

}  // namespace tesseract.

#endif  // TESSERACT_ARCH_DOTPRODUCTNEON_H_
