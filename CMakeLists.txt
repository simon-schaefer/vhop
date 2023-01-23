cmake_minimum_required(VERSION 3.19)
project(vhop)
set(CMAKE_CXX_STANDARD 20)

# Package - Eigen
set(USE_SYSTEM_EIGEN ON CACHE BOOL "Use system Eigen")
if (USE_SYSTEM_EIGEN)
    find_package(Eigen3 REQUIRED)
else()
    add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/eigen-3.4.0)
endif ()

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

# Package - OpenCV for visualization
find_package(OpenCV REQUIRED)
include_directories( SYSTEM ${OpenCV_INCLUDE_DIRS} )

include_directories(include)
add_library(${PROJECT_NAME}
    src/utility.cpp
    src/visualization.cpp
    src/BatchNorm.cpp
    src/LatentDist.cpp
    src/LeakyRelu.cpp
    src/Linear.cpp
    src/VPoser.cpp
    ${CNPY_SOURCES}
)
target_link_libraries(${PROJECT_NAME}
   Eigen3::Eigen
   ZLIB::ZLIB
   ${CERES_LIBRARIES}
   ${OpenCV_LIBRARIES}
)

add_executable(process_dataset apps/process_dataset.cpp)
target_link_libraries(process_dataset ${PROJECT_NAME} ${DEPENDENCIES} ${CERES_LIBRARIES} ${OpenCV_LIBRARIES})

# Testing with google tests.
enable_testing()
add_executable(vhop_tests tests/test_main.cpp)
target_link_libraries(vhop_tests ${PROJECT_NAME} GTest::gtest GTest::gtest_main ${OpenCV_LIBRARIES})
add_test(vhop_tests vhop_tests)
