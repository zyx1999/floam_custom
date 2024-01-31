// c++ lib
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
// ros lib
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
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

#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

std::mutex mutex_lock;
std::queue<sensor_msgs::PointCloud2ConstPtr> pointCloudBuf;
std::queue<sensor_msgs::ImageConstPtr> imageBuf;
ros::Publisher pubProjection;
ros::Publisher pubCloudLane;
void Projection()
{
    Eigen::Matrix<float, 3, 4> P_rect_2;
    P_rect_2 << 7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02,
        4.485728000000e+01, 0.000000000000e+00, 7.215377000000e+02,
        1.728540000000e+02, 2.163791000000e-01, 0.000000000000e+00,
        0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03;

    Eigen::Matrix4f R_rect_0;
    R_rect_0 << 9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03,
        0.000000000000e+00, -9.869795000000e-03, 9.999421000000e-01,
        -4.278459000000e-03, 0.000000000000e+00, 7.402527000000e-03,
        4.351614000000e-03, 9.999631000000e-01, 0.000000000000e+00,
        0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00,
        1.000000000000e+00;

    Eigen::Matrix4f Tr_velo_to_cam;
    Tr_velo_to_cam << 7.533745000000e-03, -9.999714000000e-01,
        -6.166020000000e-04, -4.069766000000e-03, 1.480249000000e-02,
        7.280733000000e-04, -9.998902000000e-01, -7.631618000000e-02,
        9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02,
        -2.717806000000e-01, 0.000000000000e+00, 0.000000000000e+00,
        0.000000000000e+00, 1.000000000000e+00;
    // 读取图像
    // std::string imgName =
    // "/home/yuxuanzhao/Desktop/kitti-velo2cam/data_object_image_2/testing/image_2/000007.png"
    std::string imgName =
        "/home/yuxuanzhao/Downloads/kitti-raw/2011_09_26/"
        "2011_09_26_drive_0029_sync/image_02/data/0000000123.png";
    cv::Mat image = cv::imread(imgName);
    int IMG_H     = image.rows;
    int IMG_W     = image.cols;
    // std::string pointcloud = "/home/yuxuanzhao/Desktop/kitti-velo2cam/"
    //                          "data_object_velodyne/testing/velodyne/000007.bin";
    std::string pointcloud =
        "/home/yuxuanzhao/Downloads/kitti-raw/2011_09_26/"
        "2011_09_26_drive_0029_sync/velodyne_points/data/0000000123.bin";
    // 打开文件
    std::ifstream file(pointcloud, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file\n";
        return;
    }
    // 获取文件大小并读取数据
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "Failed to read file\n";
        return;
    }

    std::chrono::time_point<std::chrono::system_clock> start_0, end_0;
    start_0 = std::chrono::system_clock::now();

    // 将数据转换为 Eigen 矩阵
    Eigen::MatrixXf data = Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        buffer.data(), buffer.size() / 4, 4);
    // 提取 XYZ 坐标
    Eigen::MatrixXf xyz = data.leftCols(3);

    // 将 1 添加到每个点的末尾，转换为齐次坐标
    Eigen::MatrixXf velo(4, xyz.rows());
    velo.topRows(3) = xyz.transpose();
    velo.row(3).setOnes();

    // 计算需要保留的点的索引
    std::vector<int> indices;
    for (int i = 0; i < velo.cols(); ++i) {
        if (velo(0, i) >= 0) {
            indices.push_back(i);
        }
    }
    // 构建新的过滤后的矩阵
    Eigen::MatrixXf velo_filtered(4, indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        velo_filtered.col(i) = velo.col(indices[i]);
    }

    // 计算矩阵乘积 Y = P_rect_2 * R_rect_0 * T_v2c * X
    Eigen::MatrixXf cam = P_rect_2 * R_rect_0 * Tr_velo_to_cam * velo_filtered;
    // 处理点并绘制
    for (int i = 0; i < cam.cols(); ++i) {
        // 归一化 u 和 v
        float u = cam(0, i) / cam(2, i);
        float v = cam(1, i) / cam(2, i);
        float z = cam(2, i);

        // 过滤掉画布范围外的点
        if (u >= 0 && u <= IMG_W && v >= 0 && v <= IMG_H && z >= 0) {
            // 根据深度值 z 设置颜色
            cv::Scalar color = cv::Scalar(255, 0, 0);  // 红色
            cv::circle(image, cv::Point2f(u, v), 2, color, -1);
        }
    }

    end_0 = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsed_seconds_0 = end_0 - start_0;
    float time_temp_0 = elapsed_seconds_0.count() * 1000;
    printf("processing time {%f} ms", time_temp_0);

    // 保存图像
    // cv::imwrite("/home/yuxuanzhao/Desktop/projection_image.png", image);
    // 显示图像
    cv::imshow("Projection", image);
    cv::waitKey(0);
}
void velodyneHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{
    mutex_lock.lock();
    pointCloudBuf.push(laserCloudMsg);
    mutex_lock.unlock();
}
void imageHandler(const sensor_msgs::ImageConstPtr& imageMsg)
{
    mutex_lock.lock();
    imageBuf.push(imageMsg);
    mutex_lock.unlock();
}

Eigen::MatrixXf projecting(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_in,
                           const Eigen::MatrixXf& P_rect_2,
                           const Eigen::MatrixXf& R_rect_0,
                           const Eigen::MatrixXf& Tr_velo_to_cam)
{
    Eigen::MatrixXf velo(4, cloud_in->points.size());
    for (size_t i = 0; i < cloud_in->points.size(); ++i) {
        velo(0, i) = cloud_in->points[i].x;
        velo(1, i) = cloud_in->points[i].y;
        velo(2, i) = cloud_in->points[i].z;
        velo(3, i) = 1.0;  // 齐次坐标
    }

    // 计算矩阵乘积 Y = P_rect_2 * R_rect_0 * T_v2c * X
    Eigen::MatrixXf y = P_rect_2 * R_rect_0 * Tr_velo_to_cam * velo;
    return y;
}
Eigen::Matrix<float, 3, 4> P_rect_2;
Eigen::Matrix4f R_rect_0;
Eigen::Matrix4f Tr_velo_to_cam;
uint8_t LANE_THRESHOLD = 100;
void doProjection()
{
    while (1) {
        if (!pointCloudBuf.empty() && !imageBuf.empty()) {
            // 检查时间戳，数据时间戳对齐
            mutex_lock.lock();
            if (pointCloudBuf.front()->header.stamp.toSec() <
                imageBuf.front()->header.stamp.toSec() - 0.5 * 0.1) {
                pointCloudBuf.pop();
                ROS_WARN_ONCE("time stamp unaligned with extra point cloud, "
                              "pls check your data --> odom correction");
                mutex_lock.unlock();
                continue;
            }
            if (imageBuf.front()->header.stamp.toSec() <
                pointCloudBuf.front()->header.stamp.toSec() - 0.5 * 0.1) {
                imageBuf.pop();
                ROS_WARN_ONCE("time stamp unaligned with extra point cloud, "
                              "pls check your data --> odom correction");
                mutex_lock.unlock();
                continue;
            }

            ros::Time pointcloud_time;
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in(
                new pcl::PointCloud<pcl::PointXYZI>());
            pointcloud_time = (pointCloudBuf.front())->header.stamp;
            pcl::fromROSMsg(*pointCloudBuf.front(), *cloud_in);
            pointCloudBuf.pop();
            cv_bridge::CvImagePtr cv_ptr;
            cv_ptr        = cv_bridge::toCvCopy(*imageBuf.front(),
                                         sensor_msgs::image_encodings::MONO8);
            cv::Mat image = cv_ptr->image;
            int IMG_H     = image.rows;
            int IMG_W     = image.cols;
            imageBuf.pop();
            mutex_lock.unlock();

            // filter cloud_in with x>=0
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_positive_x(
                new pcl::PointCloud<pcl::PointXYZI>());
            pcl::PassThrough<pcl::PointXYZI> pass;
            pass.setInputCloud(cloud_in);
            pass.setFilterFieldName("x");
            pass.setFilterLimits(0, 100);
            pass.filter(*cloud_positive_x);

            Eigen::MatrixXf proj_res = projecting(cloud_positive_x, P_rect_2,
                                                  R_rect_0, Tr_velo_to_cam);
            ROS_INFO("Projection finished!");
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_on_lane(
                new pcl::PointCloud<pcl::PointXYZI>());
            // 处理点并绘制
            for (int i = 0; i < proj_res.cols(); ++i) {
                // 归一化 u 和 v
                float u = proj_res(0, i) / proj_res(2, i);
                float v = proj_res(1, i) / proj_res(2, i);
                float z = proj_res(2, i);

                // 过滤掉画布范围外的点
                /*
                    0---------->x(cols, IMG_W)
                    |
                    |
                    y
                    (rows, IMG_H)
                */
                if (u >= 0 && u <= IMG_W && v >= 0 && v <= IMG_H && z >= 0) {
                    if (image.at<uint8_t>(cv::Point2f(u, v)) >= LANE_THRESHOLD) {
                        cv::circle(image, cv::Point2f(u, v), 2, 128, -1);
                        cloud_on_lane->push_back(cloud_positive_x->points[i]);
                    }
                }
            }
            ROS_INFO("Size of Cloud on lane: %ld",
                     cloud_on_lane->points.size());
            sensor_msgs::ImagePtr projMsg =
                cv_bridge::CvImage(std_msgs::Header(), "mono8", image)
                    .toImageMsg();
            pubProjection.publish(projMsg);

            sensor_msgs::PointCloud2 lanePointsMsg;
            pcl::toROSMsg(*cloud_on_lane, lanePointsMsg);
            lanePointsMsg.header.stamp    = pointcloud_time;
            lanePointsMsg.header.frame_id = "world";
            pubCloudLane.publish(lanePointsMsg);
        }
        // sleep 2 ms every time
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "projection");
    ros::NodeHandle nh;

    std::vector<float> v_f_P, v_f_R, v_f_Tr;
    nh.param<std::vector<float>>("P_rect_2", v_f_P, std::vector<float>());
    nh.param<std::vector<float>>("R_rect_0", v_f_R, std::vector<float>());
    nh.param<std::vector<float>>("Tr_velo_to_cam", v_f_Tr,
                                 std::vector<float>());

    P_rect_2 = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(
        v_f_P.data(), 3, 4);
    R_rect_0 = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(
        v_f_R.data(), 4, 4);
    Tr_velo_to_cam =
        Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(
            v_f_Tr.data(), 4, 4);

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(
        "/velodyne_points", 100, velodyneHandler);

    ros::Subscriber subCamera =
        nh.subscribe<sensor_msgs::Image>("/cam04/image_raw", 100, imageHandler);

    pubProjection = nh.advertise<sensor_msgs::Image>("/projection", 100);

    pubCloudLane = nh.advertise<sensor_msgs::PointCloud2>("/lane_points", 100);

    std::thread projection_process{doProjection};
    ros::spin();
    return 0;
}
