
cmake_minimum_required(VERSION 2.8)
project( elilil_mod )
find_package( OpenCV REQUIRED )
add_executable(analyze elilil_mod.cpp )
target_link_libraries( analyze ${OpenCV_LIBS} )
