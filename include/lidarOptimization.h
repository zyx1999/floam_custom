// Author of FLOAM: Wang Han
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro
#ifndef _LIDAR_OPTIMIZATION_ANALYTIC_H_
#define _LIDAR_OPTIMIZATION_ANALYTIC_H_

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

/// @brief 从se(3)的李代数转换到四元数和平移向量
/// @param se3 李代数
/// @param q 四元数
/// @param t 平移向量
void getTransformFromSe3(const Eigen::Matrix<double, 6, 1>& se3,
                         Eigen::Quaterniond& q,
                         Eigen::Vector3d& t);

/// @brief 创建李代数旋转分量的斜对称矩阵
/// @param mat_in 李代数的旋转分量
/// @return 返回斜对称矩阵
Eigen::Matrix3d skew(Eigen::Vector3d& mat_in);

class EdgeAnalyticCostFunction : public ceres::SizedCostFunction<1, 7> {
  public:
    /// @brief 设置lidar帧特征点和其坐标在全局特征点云邻域内的主方向线段端点。
    /// @param curr_point_ 最新lidar帧中的特征点（边缘）
    /// @param last_point_a_ 特征点坐标在的全局特征点云的邻域中的主方向线段的端点a
    /// @param last_point_b_ 特征点坐标在的全局特征点云的邻域中的主方向线段的端点b
    EdgeAnalyticCostFunction(Eigen::Vector3d curr_point_,
                             Eigen::Vector3d last_point_a_,
                             Eigen::Vector3d last_point_b_);
    virtual ~EdgeAnalyticCostFunction() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    Eigen::Vector3d curr_point;
    Eigen::Vector3d last_point_a;
    Eigen::Vector3d last_point_b;
};

class SurfNormAnalyticCostFunction : public ceres::SizedCostFunction<1, 7> {
  public:
    /// @brief 设置lidar帧特征点和其坐标在全局特征点云邻域的平面法线。
    /// @param curr_point_ 当前帧的特征点
    /// @param plane_unit_norm_ 当前帧的特征点在全局地图上的邻域的表面法线
    /// @param negative_OA_dot_norm_ d = 1/sqrt(a^2+b^2+c^2)
    SurfNormAnalyticCostFunction(Eigen::Vector3d curr_point_,
                                 Eigen::Vector3d plane_unit_norm_,
                                 double negative_OA_dot_norm_);
    virtual ~SurfNormAnalyticCostFunction() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

    Eigen::Vector3d curr_point;
    Eigen::Vector3d plane_unit_norm;
    double negative_OA_dot_norm;
};

class PoseSE3Parameterization : public ceres::LocalParameterization {
  public:
    PoseSE3Parameterization() {}
    virtual ~PoseSE3Parameterization() {}
    virtual bool
    Plus(const double* x, const double* delta, double* x_plus_delta) const;
    virtual bool ComputeJacobian(const double* x, double* jacobian) const;
    virtual int GlobalSize() const
    {
        return 7;
    }
    virtual int LocalSize() const
    {
        return 6;
    }
};

#endif  // _LIDAR_OPTIMIZATION_ANALYTIC_H_
