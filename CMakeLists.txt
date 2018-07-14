cmake_minimum_required(VERSION 2.8.12)
project("Dlib Face Technologies" C CXX)

add_subdirectory(./submodules/dlib dlib_build)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main src/main.cpp)
target_link_libraries(main dlib::dlib ${OpenCV_LIBS} )
