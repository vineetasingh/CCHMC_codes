
cmake_minimum_required(VERSION 2.8)
project(STEP1MIP )
find_package( OpenCV REQUIRED )
add_executable(analyze STEP1MIP.cpp )
target_link_libraries( analyze ${OpenCV_LIBS} )
