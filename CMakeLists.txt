cmake_minimum_required(VERSION 3.16)
project(c07)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 4.0.0 REQUIRED)
message("OpenCV Version: ${OoenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

add_executable(overlayFacemask overlayFacemask.cpp)
target_link_libraries(overlayFacemask ${OpenCV_LIBS})