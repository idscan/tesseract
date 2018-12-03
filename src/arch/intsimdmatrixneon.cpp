///////////////////////////////////////////////////////////////////////

#include "intsimdmatrixneon.h"

#include <cstdint>
#include <vector>
#include "dotproductneon.h"

// run on the generate output:
// for i in {40..120}; do s="\["$i"\]"; grep ${s} output.txt | tail -1; done

#define PRINT_TIME(INSTRUCTION, COUNTER, LINE) \
auto start_##COUNTER = std::chrono::high_resolution_clock::now(); \
INSTRUCTION;\
auto finish_##COUNTER = std::chrono::high_resolution_clock::now();\
std::chrono::duration<double> elapsed_##COUNTER = finish_##COUNTER - start_##COUNTER;\
static float processing_time_##COUNTER = 0.f;\
static float printed_time_##COUNTER = 0.f;\
processing_time_##COUNTER += elapsed_##COUNTER.count();\
if (processing_time_##COUNTER - printed_time_##COUNTER > 0.01) {\
printed_time_##COUNTER = processing_time_##COUNTER;\
std::cout << "[" << #LINE << "] " << #INSTRUCTION << " processing_time " << processing_time_##COUNTER << std::endl;\
}\

#define PRINT_TIME2(INSTRUCTION, COUNTER, LINE) \
PRINT_TIME(INSTRUCTION, COUNTER, LINE)

#define PROFILE(INSTRUCTION) \
PRINT_TIME2(INSTRUCTION, __COUNTER__, __LINE__);

#define NO_PROFILE(INSTRUCTION) INSTRUCTION;

#include <iostream>

namespace tesseract {

#ifdef __aarch64__
// Computes part of matrix.vector v = Wu. Computes 1 result.
static void PartialMatrixDotVector1(const int8_t* wi, const double* scales,
                                    const int8_t* u, int num_in, int /*num_out*/,
                                    double* v) {
  auto total = IntDotProductNEON(u, wi, num_in);
  // Add in the bias and correct for integer values.
  *v = (static_cast<double>(total) / INT8_MAX + wi[num_in]) * *scales;
}
#endif  // __aarch64__

IntSimdMatrixNEON::IntSimdMatrixNEON() {
#ifdef __aarch64__
  partial_funcs_ = {PartialMatrixDotVector1};
#endif
}

}  // namespace tesseract.
