#ifndef VHOP_CERES_ZERO_MOTION_H
#define VHOP_CERES_ZERO_MOTION_H

#include "base_residual.h"

namespace vhop {

template<size_t N_PARAMS, size_t N_TIME_STEPS>
class ConstantMotionError : public vhop::ResidualBase {

 public:

  bool operator()(const double *x, double *error) const override {
    for(int i = 0; i < N_PARAMS; i++) {
      // In case of only 2 time steps, a zero motion error is applied, i.e. we the parameters are constrained
      // to be equal in the first and second time step.
      if(N_TIME_STEPS == 2) {
        error[i] = x[N_PARAMS + i] - x[i];
      }

      // When there are more than 2 time steps, we apply a constant motion error, i.e.
      // p(t) = p(t-1) + v(t-1) * dt = p(t-1) + (p(t-1) - p(t-2)) = 2 * p(t-1) - p(t-2)
      else {
        for(int t = 1; t < N_TIME_STEPS; t++) {
          error[t * N_PARAMS + i] = 2 * x[(t - 1) * N_PARAMS + i] - x[(t - 2) * N_PARAMS + i];
        }
      }
    }
    return true;
  }

  static constexpr int getNumParams() { return N_TIME_STEPS * N_PARAMS; }
  static constexpr int getNumResiduals() { return (N_TIME_STEPS - 1) * N_PARAMS; }
};

}


#endif //VHOP_CERES_ZERO_MOTION_H
