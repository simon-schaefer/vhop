//
// Created by Burak on 9.01.2023.
//

#ifndef VHOP_FILEUTILS_H
#define VHOP_FILEUTILS_H

#include <Eigen/Dense>

namespace vhop::utility {

    Eigen::MatrixXd loadDoubleMatrix(const std::string weightFilePath, int row, int col);
    Eigen::VectorXd loadVector(const std::string weightFilePath, int row)
    ;
}

#endif //VHOP_FILEUTILS_H
