///////////////////////////////////////////////////////////////////////

#include "intsimdmatrixneon.h"
#include "dotproductneon.h"
#include <cstdint>

namespace tesseract {

#if defined(__arm__) || defined(__arm64__)
// Computes part of matrix.vector v = Wu. Computes 1 result.
static void PartialMatrixDotVector1(const int8_t* wi, const double* scales,
                                    const int8_t* u, int num_in, int /*num_out*/,
                                    double* v) {
  auto total = IntDotProductNEON(u, wi, num_in);
  // Add in the bias and correct for integer values.
  *v = (static_cast<double>(total) / INT8_MAX + wi[num_in]) * *scales;
}
#endif

IntSimdMatrixNEON::IntSimdMatrixNEON() {
#if defined(__arm__) || defined(__arm64__)
  partial_funcs_ = {PartialMatrixDotVector1};
#endif
}

}  // namespace tesseract.
