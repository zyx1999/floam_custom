// Author of FLOAM: Wang Han
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro

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

// local lib
#include "lidar.h"
#include "odomEstimationClass.h"

std::mutex mutex_lock;
lidar::Lidar lidar_param;
OdomEstimationClass odomEstimation;
std::queue<sensor_msgs::PointCloud2ConstPtr> pointCloudEdgeBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> pointCloudSurfBuf;

ros::Publisher pubLaserOdometry;
void velodyneSurfHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
    std::lock_guard<std::mutex> lock(mutex_lock);
    pointCloudSurfBuf.push(laserCloudMsg);
}
void velodyneEdgeHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
    std::lock_guard<std::mutex> lock(mutex_lock);
    pointCloudEdgeBuf.push(laserCloudMsg);
}

bool is_odom_inited = false;
double total_time   = 0;
int total_frame     = 0;
void odom_estimation()
{
    while (1) {
        if (!pointCloudEdgeBuf.empty() && !pointCloudSurfBuf.empty()) {
            // 检查时间戳，数据时间戳对齐
            mutex_lock.lock();
            if (!pointCloudSurfBuf.empty() &&
                (pointCloudSurfBuf.front()->header.stamp.toSec() <
                 pointCloudEdgeBuf.front()->header.stamp.toSec() -
                     0.5 * lidar_param.scan_period)) {
                pointCloudSurfBuf.pop();
                ROS_WARN_ONCE("time stamp unaligned with extra point cloud, "
                              "pls check your data --> odom correction");
                mutex_lock.unlock();
                continue;
            }
            if (!pointCloudEdgeBuf.empty() &&
                (pointCloudEdgeBuf.front()->header.stamp.toSec() <
                 pointCloudSurfBuf.front()->header.stamp.toSec() -
                     0.5 * lidar_param.scan_period)) {
                pointCloudEdgeBuf.pop();
                ROS_WARN_ONCE("time stamp unaligned with extra point cloud, "
                              "pls check your data --> odom correction");
                mutex_lock.unlock();
                continue;
            }
            // 时间戳对齐完成、组帧
            pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_surf_in(
                new pcl::PointCloud<pcl::PointXYZI>());
            pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_edge_in(
                new pcl::PointCloud<pcl::PointXYZI>());
            pcl::fromROSMsg(*pointCloudEdgeBuf.front(), *pointcloud_edge_in);
            pcl::fromROSMsg(*pointCloudSurfBuf.front(), *pointcloud_surf_in);
            ros::Time pointcloud_time =
                (pointCloudSurfBuf.front())->header.stamp;
            pointCloudEdgeBuf.pop();
            pointCloudSurfBuf.pop();
            mutex_lock.unlock();
            // 初始化里程计
            if (is_odom_inited == false) {
                odomEstimation.initMapWithPoints(pointcloud_edge_in,
                                                 pointcloud_surf_in);
                is_odom_inited = true;
                ROS_INFO("odom inited");
            }
            else {
                // 使用新的点云更新里程计估计
                std::chrono::time_point<std::chrono::system_clock> start, end;
                start = std::chrono::system_clock::now();
                // 该方法将更新odom所代表的变换，同时更新全局地图
                odomEstimation.updatePointsToMap(pointcloud_edge_in,
                                                 pointcloud_surf_in);
                end = std::chrono::system_clock::now();
                std::chrono::duration<float> elapsed_seconds = end - start;
                total_frame++;
                float time_temp = elapsed_seconds.count() * 1000;
                total_time += time_temp;
                ROS_INFO("average odom estimation time %f ms \n \n",
                         total_time / total_frame);
                ROS_INFO("Size of laserCloudCornerMap = %ld",odomEstimation.laserCloudCornerMap->points.size());
                ROS_INFO("Size of laserCloudEdgeMap = %ld",odomEstimation.laserCloudSurfMap->points.size());
            }
            // 从odom中获取最新变换
            Eigen::Quaterniond q_current(odomEstimation.odom.rotation());
            Eigen::Vector3d t_current = odomEstimation.odom.translation();
            // 使用tf广播当前位姿变换
            static tf::TransformBroadcaster br;
            tf::Transform transform;
            transform.setOrigin(
                tf::Vector3(t_current.x(), t_current.y(), t_current.z()));
            tf::Quaternion q(q_current.x(), q_current.y(), q_current.z(),
                             q_current.w());
            transform.setRotation(q);
            // 发布从坐标系/map到/base_link的变换（与时间戳关联）
            br.sendTransform(tf::StampedTransform(transform, ros::Time::now(),
                                                  "map", "base_link"));

            // 发布里程计数据
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id         = "map";
            laserOdometry.child_frame_id          = "base_link";
            laserOdometry.header.stamp            = pointcloud_time;
            laserOdometry.pose.pose.orientation.x = q_current.x();
            laserOdometry.pose.pose.orientation.y = q_current.y();
            laserOdometry.pose.pose.orientation.z = q_current.z();
            laserOdometry.pose.pose.orientation.w = q_current.w();
            laserOdometry.pose.pose.position.x    = t_current.x();
            laserOdometry.pose.pose.position.y    = t_current.y();
            laserOdometry.pose.pose.position.z    = t_current.z();
            pubLaserOdometry.publish(laserOdometry);
        }
        // sleep 2 ms every time
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}
/*
1.时间戳对齐检查：检查两个缓冲区中最早的点云消息的时间戳是否对齐，以确保它们是同一时刻的数据。如果时间戳不对齐，则弹出不匹配的数据并继续循环。
2.里程计初始化：如果还未初始化，则使用初始点云数据初始化里程计估计。
3.更新里程计估计：使用新的点云数据更新里程计估计。记录处理时间，计算平均处理时间。
4.广播变换：使用 tf 广播当前的位姿估计，这通常包括平移（位置）和旋转（姿态）。
5.发布里程计消息：构造并发布 nav_msgs/Odometry 消息，包含当前的位置和姿态。
*/
int main(int argc, char** argv)
{
    ros::init(argc, argv, "main");
    ros::NodeHandle nh;

    int scan_line         = 64;
    double vertical_angle = 2.0;
    double scan_period    = 0.1;
    double max_dis        = 60.0;
    double min_dis        = 2.0;
    double map_resolution = 0.4;

    // 从参数服务器读取参数
    nh.getParam("/scan_period", scan_period);
    nh.getParam("/vertical_angle", vertical_angle);
    nh.getParam("/max_dis", max_dis);
    nh.getParam("/min_dis", min_dis);
    nh.getParam("/scan_line", scan_line);
    nh.getParam("/map_resolution", map_resolution);

    // 设置lidar参数
    lidar_param.setScanPeriod(scan_period);
    lidar_param.setVerticalAngle(vertical_angle);
    lidar_param.setLines(scan_line);
    lidar_param.setMaxDistance(max_dis);
    lidar_param.setMinDistance(min_dis);

    // 用lidar初始化odom
    odomEstimation.init(lidar_param, map_resolution);

    // 订阅laser发布的两个topic
    ros::Subscriber subEdgeLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(
        "/laser_cloud_edge", 100, velodyneEdgeHandler);
    ros::Subscriber subSurfLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(
        "/laser_cloud_surf", 100, velodyneSurfHandler);

    // 发布odom话题，laserMappingNode订阅了/odom 话题
    pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/odom", 100);
    std::thread odom_estimation_process{odom_estimation};

    ros::spin();

    return 0;
}
