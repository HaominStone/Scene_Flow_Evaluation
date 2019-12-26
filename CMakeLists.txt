cmake_minimum_required(VERSION 3.0)
project(ICP)
SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O2")
add_executable(ICP ICP.cpp)
find_package(PkgConfig REQUIRED)
pkg_search_module(PKG_PNG REQUIRED libpng)

set(BOOST_COMPONENTS date_time chrono filesystem iostreams program_options regex system thread unit_test_framework)
find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})
include_directories(${Boost_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(PKG_OPENCV REQUIRED opencv)


find_package(PCL 1.8)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#version number
set( ICP_VERSION_MAJOR 1 )
set( ICP_VERSION_MINOR 0 )

#set build type
set(CMAKE_BUILD_TYPE Debug)



target_link_libraries( ICP ${PKG_PNG_LDFLAGS} )
target_link_libraries( ICP ${PKG_OPENCV_LDFLAGS} )
target_link_libraries( ICP ${PKG_PCL_LDFLAGS} )
target_link_libraries(ICP ${Boost_LIBRARIES})
target_link_libraries(ICP ${PCL_COMMON_LIBRARIES} ${PCL_KDTREE_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_SEGMENTATION_LIBRARIES} ${PCL_VISUALIZATION_LIBRARIES} ${PCL_SAMPLE_CONSENSUS_LIBRARIES})
