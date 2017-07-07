
cmake_minimum_required(VERSION 2.8)
project(STEP2_3_MIP_single )
find_package( OpenCV REQUIRED )
add_executable(analyze STEP2_3_MIP_single.cpp )
target_link_libraries( analyze ${OpenCV_LIBS} )
