cmake_minimum_required(VERSION 3.11)
project("Dlib Face Technologies" CXX)

# use C++ 17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Enable ccache if it exists
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Add OpenCV
find_package(OpenCV REQUIRED)

# Add Dlib
include(FetchContent)
set(DLIB_TAG v19.18)
FetchContent_Declare(
    dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        ${DLIB_TAG}
)
message("-- Fetching dlib ${DLIB_TAG}")
FetchContent_MakeAvailable(dlib)

# Main executable
add_executable(face src/main.cpp)
target_include_directories(face PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(face dlib::dlib ${OpenCV_LIBS} )
target_compile_options(face PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS face DESTINATION bin)
