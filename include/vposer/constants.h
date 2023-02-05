#ifndef VHOP_INCLUDE_VPOSER_CONSTANTS_H_
#define VHOP_INCLUDE_VPOSER_CONSTANTS_H_

namespace vposer {

constexpr size_t LATENT_DIM = 32;

template<typename T>
using latent_t = Eigen::Matrix<T, LATENT_DIM, 1>;

}

#endif //VHOP_INCLUDE_VPOSER_CONSTANTS_H_
