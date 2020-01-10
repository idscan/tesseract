///////////////////////////////////////////////////////////////////////
#ifndef TESSERACT_ARCH_INTSIMDMATRIXNEON_H_
#define TESSERACT_ARCH_INTSIMDMATRIXNEON_H_

#include "intsimdmatrix.h"

namespace tesseract {

// NEON implementation of IntSimdMatrix.
class IntSimdMatrixNEON : public IntSimdMatrix {
 public:
  IntSimdMatrixNEON();
};

}  // namespace tesseract

#endif  // TESSERACT_ARCH_INTSIMDMATRIXNEON_H_
