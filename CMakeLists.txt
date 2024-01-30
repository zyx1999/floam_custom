cmake_minimum_required(VERSION 2.8.3)
project(floam)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  tf
  eigen_conversions
  cv_bridge
)

find_package(Eigen3 REQUIRED)
if(NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})  # Not strictly necessary as Eigen is head only
  # Possibly map additional variables to the EIGEN3_ prefix.
else()
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()

# for signed_distance_field
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
# for signed_distance_field
find_package(OpenCV 4 REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS} 
  ${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

link_directories(
  include
  ${PCL_LIBRARY_DIRS}
  ${CERES_LIBRARY_DIRS}
)


catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL Ceres 
  INCLUDE_DIRS include
)



add_executable(floam_laser_processing_node src/laserProcessingNode.cpp src/laserProcessingClass.cpp src/lidar.cpp src/distanceFieldClass.cpp src/signed_distance_field_2d.cpp src/grid_map_2d.cpp)
target_link_libraries(floam_laser_processing_node ${EIGEN3_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(floam_odom_estimation_node src/odomEstimationNode.cpp src/lidarOptimization.cpp src/lidar.cpp src/odomEstimationClass.cpp)
target_link_libraries(floam_odom_estimation_node ${EIGEN3_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable(floam_laser_mapping_node src/laserMappingNode.cpp src/laserMappingClass.cpp src/lidar.cpp)
target_link_libraries(floam_laser_mapping_node ${EIGEN3_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable(save_pose_node src/savePoseNode.cpp)
target_link_libraries(save_pose_node ${EIGEN3_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable(signed_distance_field_2d_test src/signed_distance_field_2d_test.cpp src/grid_map_2d.cpp src/signed_distance_field_2d.cpp)
target_link_libraries(signed_distance_field_2d_test  ${EIGEN3_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS})

add_executable(testProj src/testProjection.cpp)
target_link_libraries(testProj ${EIGEN3_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS})

