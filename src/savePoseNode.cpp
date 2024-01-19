// c++ lib
#include <chrono>
#include <cmath>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// ros lib
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

// pcl lib
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

void path_save(nav_msgs::Odometry odomAftMapped)
{
    //保存轨迹，path_save是文件目录,txt文件提前建好,/home/xxx/xxx.txt,
    std::ofstream pose1("/home/yuxuanzhao/ros_workspace/floam_ws/src/floam/"
                        "result/pose_trans.txt",
                        std::ios::app);
    pose1.setf(std::ios::scientific, std::ios::floatfield);
    pose1.precision(6);
    // TODO: 坐标系有误, gt是右手坐标，odom
    // 从四元数创建旋转矩阵
    Eigen::Quaterniond q(odomAftMapped.pose.pose.orientation.x,
                         odomAftMapped.pose.pose.orientation.y,
                         odomAftMapped.pose.pose.orientation.z,
                         odomAftMapped.pose.pose.orientation.w);
    Eigen::Matrix3d R_world_velo = q.normalized().toRotationMatrix();

    // 创建4x4的齐次变换矩阵
    Eigen::Matrix4d T_world_velo = Eigen::Matrix4d::Identity();
    T_world_velo.block<3,3>(0,0) = R_world_velo;
    T_world_velo(0,3) = odomAftMapped.pose.pose.position.x;
    T_world_velo(1,3) = odomAftMapped.pose.pose.position.y;
    T_world_velo(2,3) = odomAftMapped.pose.pose.position.z;

    Eigen::Matrix3d R_velo_cam;
    R_velo_cam << 7.027555e-03, -9.999753e-01, 2.599616e-05, -2.254837e-03,
        -4.184312e-05, -9.999975e-01, 9.999728e-01, 7.027479e-03, -2.255075e-03;
    Eigen::Matrix4d T_velo_cam = Eigen::Matrix4d::Identity();
    T_velo_cam.block<3,3>(0,0) = R_velo_cam;
    T_velo_cam(0,3) = -7.137748e-03;
    T_velo_cam(1,3) = -7.482656e-02;
    T_velo_cam(2,3) = -3.336324e-01;

    Eigen::Matrix4d T_world_cam = T_velo_cam * T_world_velo;
    // 将矩阵输出为KITTI格式
    int count{1};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            pose1 << T_world_cam(i, j);
            if (count != 12) {
                pose1 << " ";
            }
            count++;
        }
    }
    pose1 << std::endl;
    pose1.close();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "save_pose_node");
    ros::NodeHandle nh;
    ros::Subscriber save_path =
        nh.subscribe<nav_msgs::Odometry>("/odom", 100, path_save);

    ros::spin();
}