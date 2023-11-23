cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(drowsiness-detection)

if(${CMAKE_SYSTEM} MATCHES Linux)
	set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/build_opencv_linux")
	set(CMAKE_CXX_FLAGS "-ldl -lpthread -lm")
else()
	set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/build_opencv_14")
endif()
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable("${CMAKE_PROJECT_NAME}"
miniaudio.h
Detection.cpp)

target_link_libraries("${CMAKE_PROJECT_NAME}" "${OpenCV_LIBS}" stdc++fs)
set_property(TARGET "${CMAKE_PROJECT_NAME}" PROPERTY CXX_STANDARD 14)

install(TARGETS "${CMAKE_PROJECT_NAME}" RUNTIME DESTINATION bin)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sound" DESTINATION .)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/model" DESTINATION .)
set(CPACK_GENERATOR "TGZ")
include(CPack)

if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
