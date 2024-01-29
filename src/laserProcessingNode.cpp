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
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

// pcl lib
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

// cv
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

// local lib
#include "distanceFieldClass.h"
#include "laserProcessingClass.h"
#include "lidar.h"

std::mutex mutex_lock;
lidar::Lidar lidar_param;
LaserProcessingClass laserProcessing;
std::queue<sensor_msgs::PointCloud2ConstPtr> pointCloudBuf;

// ros publisher
ros::Publisher pubFilteredGroundDebug;
ros::Publisher pubDistanceField;
ros::Publisher pubEdgePoints;
ros::Publisher pubSurfPoints;
ros::Publisher pubLaserCloudFiltered;
ros::Publisher pubSDFKeypoints;

void velodyneHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
    std::lock_guard<std::mutex> lock(mutex_lock);
    pointCloudBuf.push(laserCloudMsg);
}

double total_time    = 0;
int total_frame      = 0;

// load from rosparam
float cloud_filter_x = 40;
float cloud_filter_y = 40;
float cloud_filter_z = -1.73;
float sdf_x_bound    = 10;
float sdf_y_bound    = 10;
float sdf_z_lower    = -5;
float sdf_resolution = 1;  // 体素网格的分辨率
bool useFilteredGroundPoints{false};
// sdfmin sdfmax: filter sdf with value in [sdfmin, sdfmax]
float sdfmin = 0;
float sdfmax = 10;

void groundFilter(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_in,
                  pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_out)
{
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(cloud_in);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-cloud_filter_x, cloud_filter_x);
    pass.filter(*cloud_out);

    pass.setInputCloud(cloud_out);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-cloud_filter_y, cloud_filter_y);
    pass.filter(*cloud_out);

    pass.setInputCloud(cloud_out);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-10, cloud_filter_z);
    pass.filter(*cloud_out);

    pass.setInputCloud(cloud_out);
    pass.setFilterFieldName("intensity");
    pass.setFilterLimits(0.35, 0.45);
    pass.filter(*cloud_out);
}

void segPlane(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_in,
              pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_plane_out)
{
    // 创建用于平面分割的 SACSegmentation 对象
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    // 设置分割模型为平面模型
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(1000);
    seg.setDistanceThreshold(0.01);  // 设置到模型的距离阈值

    // 输入点云
    seg.setInputCloud(cloud_in);
    // 执行分割
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
        std::cerr << "Could not estimate a planar model for the given dataset."
                  << std::endl;
        return;
    }

    // 提取分割出的平面
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(cloud_in);
    extract.setIndices(inliers);
    extract.setNegative(false);  // 如果设置为 true，则提取除平面外的所有点

    // 获取平面点云
    extract.filter(*cloud_plane_out);
}

void laser_processing()
{
    while (1) {
        if (!pointCloudBuf.empty()) {
            // read data
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(
                new pcl::PointCloud<pcl::PointXYZI>());
            ros::Time pointcloud_time;
            mutex_lock.lock();
            pcl::fromROSMsg(*pointCloudBuf.front(), *cloud_in);
            pointcloud_time = (pointCloudBuf.front())->header.stamp;
            pointCloudBuf.pop();
            mutex_lock.unlock();

            /*  在featureExtraction()前过滤出地面点云，发布一个新话题用于Debug。
                rage X in(-80, 80), Y in(-80, 80), Z in(-25, 3) */
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filterground(
                new pcl::PointCloud<pcl::PointXYZI>());

            groundFilter(cloud_in, cloud_filterground);

            // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_plane(
            //     new pcl::PointCloud<pcl::PointXYZI>());
            // segPlane(cloud_filterground, cloud_plane);

            std::chrono::time_point<std::chrono::system_clock> start_0, end_0;
            start_0 = std::chrono::system_clock::now();
            pcl::PointCloud<pcl::PointXYZI>::Ptr distance_field(
                new pcl::PointCloud<pcl::PointXYZI>());
            pcl::PointCloud<pcl::PointXYZI>::Ptr sdf_keypoints(
                new pcl::PointCloud<pcl::PointXYZI>());
            // 计算2D SDF
            DistanceField df(sdf_x_bound, sdf_y_bound, sdf_z_lower,
                             cloud_filter_z, sdf_resolution);
            df.computeSignedDistanceField(cloud_filterground);
            df.getSDFPointCloud(distance_field, sdfmin, sdfmax);
            sdf_keypoints = distance_field;
            // df.detectKeypoints(distance_field, sdf_keypoints);
            ROS_INFO("Num of kps: %ld", sdf_keypoints->size());
            end_0 = std::chrono::system_clock::now();
            std::chrono::duration<float> elapsed_seconds_0 = end_0 - start_0;
            float time_temp_0 = elapsed_seconds_0.count() * 1000;
            ROS_INFO("SDF processing time {%f} ms", time_temp_0);

            // 调用laserProcessingClass::featureExtraction处理点云特征
            pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_edge(
                new pcl::PointCloud<pcl::PointXYZI>());
            pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_surf(
                new pcl::PointCloud<pcl::PointXYZI>());
            std::chrono::time_point<std::chrono::system_clock> start, end;
            start = std::chrono::system_clock::now();
            if (useFilteredGroundPoints) {
                ROS_INFO("Using SDF Points");
                laserProcessing.featureExtraction(
                    distance_field, pointcloud_edge, pointcloud_surf);
            }
            else {
                laserProcessing.featureExtraction(cloud_in, pointcloud_edge,
                                                  pointcloud_surf);
            }
            end = std::chrono::system_clock::now();
            std::chrono::duration<float> elapsed_seconds = end - start;
            total_frame++;
            float time_temp = elapsed_seconds.count() * 1000;
            total_time += time_temp;
            // ROS_INFO("average laser processing time %f ms \n \n",
            // total_time/total_frame);

            sensor_msgs::PointCloud2 laserCloudFilteredMsg;
            pcl::PointCloud<pcl::PointXYZI>::Ptr pointcloud_filtered(
                new pcl::PointCloud<pcl::PointXYZI>());
            *pointcloud_filtered += *pointcloud_edge;
            *pointcloud_filtered += *pointcloud_surf;
            pcl::toROSMsg(*pointcloud_filtered, laserCloudFilteredMsg);
            laserCloudFilteredMsg.header.stamp    = pointcloud_time;
            laserCloudFilteredMsg.header.frame_id = "base_link";
            pubLaserCloudFiltered.publish(laserCloudFilteredMsg);

            sensor_msgs::PointCloud2 edgePointsMsg;
            pcl::toROSMsg(*pointcloud_edge, edgePointsMsg);
            edgePointsMsg.header.stamp    = pointcloud_time;
            edgePointsMsg.header.frame_id = "base_link";
            pubEdgePoints.publish(edgePointsMsg);

            sensor_msgs::PointCloud2 surfPointsMsg;
            pcl::toROSMsg(*pointcloud_surf, surfPointsMsg);
            surfPointsMsg.header.stamp    = pointcloud_time;
            surfPointsMsg.header.frame_id = "base_link";
            pubSurfPoints.publish(surfPointsMsg);

            sensor_msgs::PointCloud2 filteredGroundPointsMsg;
            pcl::toROSMsg(*cloud_filterground, filteredGroundPointsMsg);
            filteredGroundPointsMsg.header.stamp    = pointcloud_time;
            filteredGroundPointsMsg.header.frame_id = "base_link";
            pubFilteredGroundDebug.publish(filteredGroundPointsMsg);
            ROS_INFO("Before: {%ld}   <===>   After {%ld}",
                     cloud_in->points.size(),
                     cloud_filterground->points.size());

            sensor_msgs::PointCloud2 distanceFieldMsg;
            pcl::toROSMsg(*distance_field, distanceFieldMsg);
            distanceFieldMsg.header.stamp    = pointcloud_time;
            distanceFieldMsg.header.frame_id = "base_link";
            pubDistanceField.publish(distanceFieldMsg);

            sensor_msgs::PointCloud2 sdfKeypointsMsg;
            pcl::toROSMsg(*sdf_keypoints, sdfKeypointsMsg);
            sdfKeypointsMsg.header.stamp    = pointcloud_time;
            sdfKeypointsMsg.header.frame_id = "base_link";
            pubSDFKeypoints.publish(sdfKeypointsMsg);
        }
        // sleep 2 ms every time
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}
/*
1. 初始化laserProcessingClass（从参数服务器读取数据）
2.
subLaserCloud订阅rosbag发布的velodyne_points话题，在callback中将点云msg加入队列
3. 绑定要发布的话题：velodyne_points_filtered, laser_cloud_edge,
laser_cloud_surf
4. 创建线程，laser_processing()处理原始点云数据
    1.不断从队列中弹出点云
    2.调用laserProcessingClass::featureExtraction处理点云，得到edge和surf特征
    3.发布点云话题（filtered, edge, surf）
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

    nh.getParam("/scan_period", scan_period);
    nh.getParam("/vertical_angle", vertical_angle);
    nh.getParam("/max_dis", max_dis);
    nh.getParam("/min_dis", min_dis);
    nh.getParam("/scan_line", scan_line);
    nh.getParam("/cloud_filter_x", cloud_filter_x);
    nh.getParam("/cloud_filter_y", cloud_filter_y);
    nh.getParam("/cloud_filter_z", cloud_filter_z);
    nh.getParam("/sdf_x_bound", sdf_x_bound);
    nh.getParam("/sdf_y_bound", sdf_y_bound);
    nh.getParam("/sdf_z_lower", sdf_z_lower);
    nh.getParam("/sdf_resolution", sdf_resolution);
    nh.getParam("/useFilteredGroundPoints", useFilteredGroundPoints);
    nh.getParam("/sdfmin", sdfmin);
    nh.getParam("/sdfmax", sdfmax);

    lidar_param.setScanPeriod(scan_period);
    lidar_param.setVerticalAngle(vertical_angle);
    lidar_param.setLines(scan_line);
    lidar_param.setMaxDistance(max_dis);
    lidar_param.setMinDistance(min_dis);

    // 初始化laserProcessingClass
    laserProcessing.init(lidar_param);
    // 订阅testbag里的topic: velodyne_points
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(
        "/velodyne_points", 100, velodyneHandler);

    pubFilteredGroundDebug =
        nh.advertise<sensor_msgs::PointCloud2>("/ground_points", 100);

    pubDistanceField =
        nh.advertise<sensor_msgs::PointCloud2>("/distance_field", 100);

    pubSDFKeypoints =
        nh.advertise<sensor_msgs::PointCloud2>("/sdf_keypoints", 100);

    // filtered = egde + surf
    pubLaserCloudFiltered = nh.advertise<sensor_msgs::PointCloud2>(
        "/velodyne_points_filtered", 100);
    // edge feature
    pubEdgePoints =
        nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_edge", 100);
    // surf feature
    pubSurfPoints =
        nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf", 100);

    // 创建线程，调用laser_processing()方法
    std::thread laser_processing_process{laser_processing};

    ros::spin();

    return 0;
}
