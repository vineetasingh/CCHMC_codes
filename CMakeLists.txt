
cmake_minimum_required(VERSION 2.8)
project( ChangHui )
find_package( OpenCV REQUIRED )
add_executable(analyze ChangHui.cpp )
target_link_libraries( analyze ${OpenCV_LIBS} )
