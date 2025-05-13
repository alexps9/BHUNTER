cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(pcl_downsample)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#add_executable (pcl_downsample pcl_downsample.cpp)
add_library(pcl_downsample SHARED pcl_downsample.cpp)
target_link_libraries (pcl_downsample ${PCL_LIBRARIES})