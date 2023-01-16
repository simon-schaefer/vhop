#include "ceres/ceres.h"
#include <Eigen/Core>


class ReprojectionError {
 public:
  ReprojectionError() = default;

  template <typename T>
  bool operator()(const T* input_point, T* reprojection_error) const {
      Eigen::Matrix<T, 1, 4> point;
      point << input_point[0], input_point[1], input_point[2], 1.0;
      reprojection_error[0] = point.squaredNorm();
      return true;
  }
};



int main(int argc, char** argv)
{
    Eigen::Vector4d point(1, 1, 1, 1);

    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ReprojectionError, 1, 4>(
            new ReprojectionError()
        ),
        nullptr, point.data()
    );

    ceres::Solver::Options options;
    options.max_num_iterations = 10;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << std::endl;

    return 0;
}
