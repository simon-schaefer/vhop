#ifndef VHOP_CERES_ZERO_MOTION_H
#define VHOP_CERES_ZERO_MOTION_H

#include "base_residual.h"


namespace vhop {

template<size_t N_TIME_STEPS, size_t N_PARAMS>
class ZeroMotionError : public vhop::ResidualBase {

 public:

  bool operator()(const double *x, double *reprojection_error) const override {
    for(int t = 0; t < N_TIME_STEPS; t++) {
      for(int i = 0; i < N_PARAMS; i++) {
        reprojection_error[t * N_PARAMS + i] = x[(t + 1) * N_PARAMS + i] - x[t * N_PARAMS + i];
      }
    }
    return true;
  }

  static constexpr int getNumParams() { return N_TIME_STEPS * N_PARAMS; }
  static constexpr int getNumResiduals() { return (N_TIME_STEPS - 1) * N_PARAMS; }
};

}


#endif //VHOP_CERES_ZERO_MOTION_H
