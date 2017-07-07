
cmake_minimum_required(VERSION 2.8)
project(Source )
find_package( OpenCV REQUIRED )
add_executable(analyze Source.cpp )
target_link_libraries( analyze ${OpenCV_LIBS} )
