cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(torchExample)

if(${CMAKE_SYSTEM} MATCHES Linux)
	set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/build_opencv_linux")
	set(CMAKE_CXX_FLAGS "-ldl -lpthread -lm")
else()
	set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/build_opencv_14")
endif()
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(example-app
miniaudio.h
Detection.cpp)

target_link_libraries(example-app "${OpenCV_LIBS}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

