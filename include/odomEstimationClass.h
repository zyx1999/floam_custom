// Author of FLOAM: Wang Han
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro
#ifndef _ODOM_ESTIMATION_CLASS_H_
#define _ODOM_ESTIMATION_CLASS_H_

// std lib
#include <math.h>
#include <string>
#include <vector>

// PCL
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// LOCAL LIB
#include "lidar.h"
#include "lidarOptimization.h"
#include <ros/ros.h>

class OdomEstimationClass {
  public:
    OdomEstimationClass();

    void init(lidar::Lidar lidar_param, double map_resolution);

    /// @brief
    /// 用首次观测到的特征点云初始化局部地图数据，局部地图中的点云都在世界坐标系下，
    /// 因为首次观测时当前坐标系与世界坐标系重合，因此初始化时可以无需坐标变换。
    /// @param edge_in 边缘特征点云
    /// @param surf_in 平面特征点云
    void
    initMapWithPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& edge_in,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_in,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr& sdf_kpts_in);

    /// @brief 优化里程计位姿变换，更新局部地图
    /// @param edge_in 边缘特征点
    /// @param surf_in 平面特征点
    void
    updatePointsToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& edge_in,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_in,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr& sdf_kpts_in);

    void getMap(pcl::PointCloud<pcl::PointXYZI>::Ptr& laserCloudMap);

    /// @brief 存储最新的位姿估计，是一个表示旋转和平移的刚体变换
    Eigen::Isometry3d odom;

    /// @brief 全局地图（世界坐标系），存储边缘特征
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerMap;

    /// @brief 全局地图（世界坐标系），存储平面特征
    pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfMap;

    /// @brief 全局地图（世界坐标系），存储sdf关键点
    pcl::PointCloud<pcl::PointXYZI>::Ptr sdfKeyPointsMap;

  private:
    /// @brief 初始化用作优化问题中的参数，前4个表示四元数([0, 0, 0,
    /// 1]表示单位四元数)，后3个表示平移向量
    double parameters[7] = {0, 0, 0, 1, 0, 0, 0};

    /// @brief 从当前坐标系到世界坐标系的旋转，用Eigen::Map将参数映射到四元数
    Eigen::Map<Eigen::Quaterniond> q_w_curr =
        Eigen::Map<Eigen::Quaterniond>(parameters);

    /// @brief 从当前坐标系到世界坐标系的平移，用Eigen::Map将参数映射到平移向量
    Eigen::Map<Eigen::Vector3d> t_w_curr =
        Eigen::Map<Eigen::Vector3d>(parameters + 4);

    /// @brief 存储上一次的位姿估计，是一个表示旋转和平移的刚体变换
    Eigen::Isometry3d last_odom;

    /// @brief kd-tree
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeEdgeMap;
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfMap;

    /// @brief points downsampling before add to map
    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterEdge;
    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterSurf;

    /// @brief 局部地图，剪裁盒滤波用于剪裁点云数据，只保留特定区域的点
    pcl::CropBox<pcl::PointXYZI> cropBoxFilter;

    /// @brief optimization count
    int optimization_count;

    /// @brief
    /// 遍历当前帧中的特征点，计算当前特征点坐标在全局地图中的邻域内的几何中心，
    /// 协方差矩阵，特征向量。获取领域内特征点的主方向，构造线段。创建损失函数，添加残差块。
    /// @param pc_in 当前帧中的降采样后的边缘特征点云
    /// @param map_in 存放边缘特征的全局地图
    /// @param problem
    /// @param loss_function
    void addEdgeCostFactor(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
                           const pcl::PointCloud<pcl::PointXYZI>::Ptr& map_in,
                           ceres::Problem& problem,
                           ceres::LossFunction* loss_function);

    /// @brief
    /// @param pc_in 当前帧中的降采样后的表面特征点云
    /// @param map_in 存放表面特征的全局地图
    /// @param problem
    /// @param loss_function
    void addSurfCostFactor(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
                           const pcl::PointCloud<pcl::PointXYZI>::Ptr& map_in,
                           ceres::Problem& problem,
                           ceres::LossFunction* loss_function);

    /// @brief
    /// @param pc_in 当前帧中的SDF关键点
    /// @param map_in 存放SDF关键点的全局地图
    /// @param problem
    /// @param loss_function
    void addSDFKPCostFactor(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
                            const pcl::PointCloud<pcl::PointXYZI>::Ptr& map_in,
                            ceres::Problem& problem,
                            ceres::LossFunction* loss_function);

    /// @brief 更新局部地图
    /// @param downsampledEdgeCloud 降采样后的边缘特征点云
    /// @param downsampledSurfCloud 降采样后的平面特征点云
    void addPointsToMap(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& downsampledEdgeCloud,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& downsampledSurfCloud,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& sdfkptsCloud);

    /// @brief 完成从当前坐标系到世界坐标系的变换
    /// @param pi 输入点（当前坐标系）
    /// @param po 输出点（世界坐标系）
    void pointAssociateToMap(pcl::PointXYZI const* const pi,
                             pcl::PointXYZI* const po);

    /// @brief 降采样输入的边缘特征点云和表面特征点云。
    /// @param edge_pc_in 输入的边缘特征点云
    /// @param edge_pc_out 降采样后的边缘特征点云
    /// @param surf_pc_in 输入的表面特征点云
    /// @param surf_pc_out 降采样后的表面特征点云
    void
    downSamplingToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& edge_pc_in,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr& edge_pc_out,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc_in,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc_out);
};

#endif  // _ODOM_ESTIMATION_CLASS_H_
