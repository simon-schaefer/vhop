#ifndef VHOP_INCLUDE_VHOP_UTILITY_IMPL_HPP_
#define VHOP_INCLUDE_VHOP_UTILITY_IMPL_HPP_


template<typename T>
Eigen::Matrix<T, 3, 3> vhop::utility::rodriguesMatrix(const Eigen::Vector<T, 3> &r) {
  T theta = r.norm();
  Eigen::Matrix<T, 3, 3> R = Eigen::Matrix<T, 3, 3>::Identity();
  if (theta > 1e-8) {
    Eigen::Vector<T, 3> r_normalized = r / theta;
    Eigen::Matrix<T, 3, 3> r_cross = Eigen::Matrix<T, 3, 3>::Zero();
    r_cross(0, 1) = -r_normalized(2);
    r_cross(0, 2) = r_normalized(1);
    r_cross(1, 0) = r_normalized(2);
    r_cross(1, 2) = -r_normalized(0);
    r_cross(2, 0) = -r_normalized(1);
    r_cross(2, 1) = r_normalized(0);
    R += sin(theta) * r_cross + (1.0 - cos(theta)) * r_cross * r_cross;
  }
  return R;
}

template<typename T>
Eigen::Vector<T, 3> vhop::utility::rodriguesVector(const Eigen::Matrix<T, 3, 3> &R) {
  Eigen::AngleAxis<T> r(R);
  return r.angle() * r.axis();
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 2> vhop::utility::project(const Eigen::Matrix<T, Eigen::Dynamic, 3>& p,
                                                           const Eigen::Matrix3d& K) {
  Eigen::Matrix<T, Eigen::Dynamic, 2> p2d(p.rows(), 2);
  for (int i = 0; i < p.rows(); i++) {
    p2d(i, 0) = p(i, 0) * K(0, 0) / p(i, 2) + K(0, 2);
    p2d(i, 1) = p(i, 1) * K(1, 1) / p(i, 2) + K(1, 2);
  }
  return p2d;
}

#endif //VHOP_INCLUDE_VHOP_UTILITY_IMPL_HPP_
