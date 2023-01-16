cmake_minimum_required(VERSION 3.19)
project(vhop)
set(CMAKE_CXX_STANDARD 20)

# Package - Eigen
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
find_package(Eigen3 REQUIRED)
include_directories( SYSTEM ${EIGEN3_INCLUDE_DIRS} )

# Package - Ceres
find_package(Ceres REQUIRED)

# Package - cnpy
set( CNPY_SOURCES ${PROJECT_SOURCE_DIR}/thirdparty/cnpy/cnpy.cpp )
include_directories( SYSTEM ${PROJECT_SOURCE_DIR}/thirdparty/cnpy)

# Package - zlib, as required by cnpy
find_package(ZLIB REQUIRED)

# Package - Google Tests
find_package(GTest REQUIRED)
include_directories( SYSTEM ${GTEST_INCLUDE_DIRS} )

include_directories(include)
add_library(${PROJECT_NAME}
    src/utility.cpp
    src/smpl_model.cpp
    src/vposer/BatchNorm.cpp
    src/vposer/LatentDist.cpp
    src/vposer/LeakyRelu.cpp
    src/vposer/Linear.cpp
    src/vposer/VPoser.cpp
    ${CNPY_SOURCES}
)
target_link_libraries(${PROJECT_NAME}
   Eigen3::Eigen
   ZLIB::ZLIB
   ${CERES_LIBRARIES}
)

add_executable(vhop_main main.cpp)
target_link_libraries(vhop_main ${PROJECT_NAME} ${DEPENDENCIES})

# Testing with google tests.
enable_testing()
add_executable(vhop_test_utility tests/test_utility.cpp)
target_link_libraries(vhop_test_utility ${PROJECT_NAME} GTest::gtest GTest::gtest_main)
add_test(vhop_test_utility test_utility)

add_executable(vhop_test_smpl tests/test_smpl_model.cpp)
target_link_libraries(vhop_test_smpl ${PROJECT_NAME} GTest::gtest GTest::gtest_main ${DEPENDENCIES})
add_test(vhop_test_smpl test_smpl)

add_executable(vhop_test_vposer tests/test_vposer.cpp)
target_link_libraries(vhop_test_vposer ${PROJECT_NAME} GTest::gtest GTest::gtest_main ${DEPENDENCIES})
add_test(vhop_test_vposer test_vposer)
