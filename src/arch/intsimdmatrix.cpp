///////////////////////////////////////////////////////////////////////
// File:        intsimdmatrix.cpp
// Description: Base class for 8-bit int SIMD matrix multipliers.
// Author:      Ray Smith
// Created:     Tue Aug 15 08:01:32 PST 2017
//
// (C) Copyright 2017, Google Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////

#include "intsimdmatrix.h"
#include "genericvector.h"      // for GenericVector
#include "matrix.h"             // for GENERIC_2D_ARRAY
#include "simddetect.h"         // for SIMDDetect
#include <iostream>
#include <Eigen/Dense>

namespace tesseract {

const IntSimdMatrix* IntSimdMatrix::intSimdMatrix = nullptr;

// Computes a reshaped copy of the weight matrix w.
void IntSimdMatrix::Init(const GENERIC_2D_ARRAY<int8_t>& w,
                         std::vector<int8_t>& shaped_w) const {
  const int num_out = w.dim1();
  const int num_in = w.dim2() - 1;
  // The rounded-up sizes of the reshaped weight matrix, excluding biases.
  int rounded_num_in = Roundup(num_in, num_inputs_per_group_);
  int rounded_num_out = RoundOutputs(num_out);
  // Add the bias and compute the required size.
  shaped_w.resize((rounded_num_in + 1) * rounded_num_out, 0);
  int shaped_index = 0;
  int output = 0;
  // Each number of registers needs a different format! Iterates over the
  // different numbers of registers (each a power of 2).
  for (int num_registers = max_output_registers_; num_registers >= 1;
       num_registers /= 2) {
    // The number of outputs that we will generate with this many registers.
    int num_outputs_per_register_set =
        num_registers * num_outputs_per_register_;
    // Use the max number of registers until we have to go fewer.
    while (output + num_outputs_per_register_set <= rounded_num_out) {
      // Accumulating outputs in registers saves iterating over the inputs, so
      // we only have to do it once per output register set.
      for (int input = 0; input < num_in; input += num_inputs_per_group_) {
        // Iterate over the number of outputs in a register set.
        for (int j = 0; j < num_outputs_per_register_set; ++j) {
          // Inner-most loop corresponds to the number of inputs in an input
          // group.
          for (int i = 0; i < num_inputs_per_group_; ++i) {
            int8_t weight = 0;
            if (output + j < num_out && input + i < num_in)
              weight = w(output + j, input + i);
            shaped_w[shaped_index++] = weight;
          }
        }
      }
      // Append the bias weights for the register set.
      for (int j = 0; j < num_outputs_per_register_set; ++j) {
        int8_t weight = 0;
        if (output + j < num_out) weight = w(output + j, num_in);
        shaped_w[shaped_index++] = weight;
      }
      output += num_outputs_per_register_set;
    }
  }
}
//
// void dotprod_rrrf_run(int8_t *      _h,
//                          int8_t *      _x,
//                          unsigned int _n,
//                          int *      _y)
//    {
//        int8x8x4_t x;   // input vector
//        int8x8x4_t h;   // coefficients vector
//        int16x8x4_t p; // product
//        int64 s;   // dot product
//        
//        // load zeros into sum register
//        static const int zeros[32] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
//        int32x4x4_t_t sum = vld1q_(zeros);
//        
//        // t = 32*(floor(_n/32))
//        unsigned int t = (_n >> 5) << 5;
//        
//        //
//        unsigned int i;
//        for (i=0; i<t; i+=4) {
//            // load inputs into register (unaligned)
//            v = vld1q_f32(&_x[i]);
//            
//            // load coefficients into register (aligned)
//            h = vld1q_f32(&_h[i]);
//            
//            // compute multiplication
//            s = vmulq_f32(h,v);
//            
//            // parallel addition
//            sum = vaddq_f32(sum, s);
//        }
//        
//        // unload packed array
//        float w[4];
//        vst1q_f32(w, sum);
//        float total = w[0] + w[1] + w[2] + w[3];
//        
//        // cleanup
//        for (; i<_n; i++)
//            total += _x[i] * _h[i];
//        
//        // set return value
//        *_y = total;
//    }

// Computes matrix.vector v = Wu.
// u is of size W.dim2() - 1 and the output v is of size W.dim1().
// u is imagined to have an extra element at the end with value 1, to
// implement the bias, but it doesn't actually have it.

void IntSimdMatrix::MatrixDotVector(const GENERIC_2D_ARRAY<int8_t>& w,
                                    const GenericVector<double>& scales,
                                    const int8_t* u, double* v) {
  int num_out = w.dim1();
  int num_in = w.dim2() - 1;
  // Base implementation.
  for (int i = 0; i < num_out; ++i) {
    const int8_t* wi = w[i];
    int total = 0;
    for (int j = 0; j < num_in; ++j) total += wi[j] * u[j];
    // Add in the bias and correct for integer values.
    v[i] = (static_cast<double>(total) / INT8_MAX + wi[num_in]) * scales[i];
  }
}

}  // namespace tesseract
