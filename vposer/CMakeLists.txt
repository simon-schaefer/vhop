cmake_minimum_required(VERSION 3.9)
project(vhop)

set(CMAKE_CXX_STANDARD 17)


find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(vhop main.cpp include/BaseLayer.h include/Linear.h src/Linear.cpp include/BatchNorm.h src/BatchNorm.cpp include/LeakyRelu.h src/LeakyRelu.cpp include/VPoser.h src/VPoser.cpp include/FileUtils.h src/FileUtils.cpp include/LatentDist.h src/LatentDist.cpp include/Transformations.h src/Transformations.cpp)
target_link_libraries(vhop Eigen3::Eigen)