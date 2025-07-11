cmake_minimum_required(VERSION 3.10)

project(Object_detection)

set(OpenCV_DIR C:/Users/HP/Downloads/opencv/build)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} Main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})