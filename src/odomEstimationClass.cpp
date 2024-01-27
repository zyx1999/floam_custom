// Author of FLOAM: Wang Han
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro

#include "odomEstimationClass.h"

void OdomEstimationClass::init(lidar::Lidar lidar_param, double map_resolution)
{
    // init local map
    laserCloudCornerMap = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>());
    laserCloudSurfMap = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>());
    sdfKeyPointsMap = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>());
    // downsampling size
    downSizeFilterEdge.setLeafSize(map_resolution, map_resolution,
                                   map_resolution);
    downSizeFilterSurf.setLeafSize(map_resolution * 2, map_resolution * 2,
                                   map_resolution * 2);

    // kd-tree
    kdtreeEdgeMap = pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr(
        new pcl::KdTreeFLANN<pcl::PointXYZI>());
    kdtreeSurfMap = pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr(
        new pcl::KdTreeFLANN<pcl::PointXYZI>());
    kdtreeSDFMap = pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr(
        new pcl::KdTreeFLANN<pcl::PointXYZI>());

    odom               = Eigen::Isometry3d::Identity();
    last_odom          = Eigen::Isometry3d::Identity();
    optimization_count = 2;
}

void OdomEstimationClass::initMapWithPoints(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& edge_in,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_in,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& sdf_kpts_in)
{
    *laserCloudCornerMap += *edge_in;
    *laserCloudSurfMap += *surf_in;
    *sdfKeyPointsMap += *sdf_kpts_in;
    optimization_count = 12;
}

void OdomEstimationClass::updatePointsToMap(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& edge_in,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_in,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& sdf_kpts_in)
{
    if (optimization_count > 2)
        optimization_count--;

    // 更新里程计计数
    Eigen::Isometry3d odom_prediction = odom * (last_odom.inverse() * odom);
    last_odom                         = odom;
    odom                              = odom_prediction;

    // 分解为旋转与平移
    q_w_curr = Eigen::Quaterniond(odom.rotation());
    t_w_curr = odom.translation();

    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampledEdgeCloud(
        new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampledSurfCloud(
        new pcl::PointCloud<pcl::PointXYZI>());
    // 降采样点云
    downSamplingToMap(edge_in, downsampledEdgeCloud, surf_in,
                      downsampledSurfCloud);
    // ROS_WARN("point nyum%d,%d",(int)downsampledEdgeCloud->points.size(),
    // (int)downsampledSurfCloud->points.size());
    if (laserCloudCornerMap->points.size() > 10 &&
        laserCloudSurfMap->points.size() > 50) {
        // KD树设置目标点云
        kdtreeEdgeMap->setInputCloud(laserCloudCornerMap);
        kdtreeSurfMap->setInputCloud(laserCloudSurfMap);
        kdtreeSDFMap->setInputCloud(sdfKeyPointsMap);
        // 非线性优化求解
        for (int iterCount = 0; iterCount < optimization_count; iterCount++) {
            ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
            ceres::Problem::Options problem_options;
            ceres::Problem problem(problem_options);
            // 设置优化参数
            problem.AddParameterBlock(parameters, 7,
                                      new PoseSE3Parameterization());
            // addEdgeCostFactor(downsampledEdgeCloud, laserCloudCornerMap,
            //                   problem, loss_function);
            // addSurfCostFactor(downsampledSurfCloud, laserCloudSurfMap, problem,
            //                   loss_function);
            addSDFKPCostFactor(sdf_kpts_in, sdfKeyPointsMap, problem,
                               loss_function);
            ceres::Solver::Options options;
            options.linear_solver_type                = ceres::DENSE_QR;
            options.max_num_iterations                = 4;
            options.minimizer_progress_to_stdout      = false;
            options.check_gradients                   = false;
            options.gradient_check_relative_precision = 1e-4;
            ceres::Solver::Summary summary;

            ceres::Solve(options, &problem, &summary);
        }
    }
    else {
        printf("not enough points in map to associate, map error");
    }
    /*
    q_w_curr和t_w_curr与parameters通过Eigen::Map完成映射，
    因此在优化过程中parameters值的调整同样使得q_w_curr和t_w_curr改变。
    由此odom将获得最新的位姿变换（从当前到世界）
    */
    odom               = Eigen::Isometry3d::Identity();
    odom.linear()      = q_w_curr.toRotationMatrix();
    odom.translation() = t_w_curr;
    addPointsToMap(downsampledEdgeCloud, downsampledSurfCloud, sdf_kpts_in);
}

void OdomEstimationClass::pointAssociateToMap(pcl::PointXYZI const* const pi,
                                              pcl::PointXYZI* const po)
{
    Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
    po->x                   = point_w.x();
    po->y                   = point_w.y();
    po->z                   = point_w.z();
    po->intensity           = pi->intensity;
    // po->intensity = 1.0;
}

void OdomEstimationClass::downSamplingToMap(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& edge_pc_in,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& edge_pc_out,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc_in,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc_out)
{
    downSizeFilterEdge.setInputCloud(edge_pc_in);
    downSizeFilterEdge.filter(*edge_pc_out);
    downSizeFilterSurf.setInputCloud(surf_pc_in);
    downSizeFilterSurf.filter(*surf_pc_out);
}

void OdomEstimationClass::addEdgeCostFactor(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& map_in,
    ceres::Problem& problem,
    ceres::LossFunction* loss_function)
{
    int corner_num = 0;
    for (int i = 0; i < (int)pc_in->points.size(); i++) {
        pcl::PointXYZI point_temp;
        // 当前帧点云变换到世界坐标系
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        // 从全局特征点云中寻找与当前点（新帧）最近的5个点，并返回索引与距离（按平方距离升序排列）
        kdtreeEdgeMap->nearestKSearch(point_temp, 5, pointSearchInd,
                                      pointSearchSqDis);
        // printf("===> kdtreeEdgeMap: pointSearchSqDis[4]=%.3f\n", pointSearchSqDis[4]);
        // 最远的邻域点满足阈值(1.0)，则认为邻域点足够近 pointSearchSqDis[4] < 1.0
        if (pointSearchSqDis[4] < 1.0) {
            std::vector<Eigen::Vector3d> nearCorners;
            Eigen::Vector3d center(0, 0, 0);
            for (int j = 0; j < 5; j++) {
                Eigen::Vector3d tmp(map_in->points[pointSearchInd[j]].x,
                                    map_in->points[pointSearchInd[j]].y,
                                    map_in->points[pointSearchInd[j]].z);
                center = center + tmp;
                nearCorners.push_back(tmp);
            }
            // 计算邻域点的几何中心
            center = center / 5.0;
            /* 计算协方差矩阵：
                协方差矩阵用于描述点集在空间中的扩散程度和方向。通过计算协方差矩阵的特征值和特征向量，
                可以获得点集的主要分布方向和形状（例如，是否形成线性或平面结构）。*/
            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
            for (int j = 0; j < 5; j++) {
                // 计算邻域点与几何中心的差值
                Eigen::Matrix<double, 3, 1> tmpZeroMean =
                    nearCorners[j] - center;
                covMat += tmpZeroMean * tmpZeroMean.transpose();
            }
            // 对协方差矩阵使用特征值分解
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
            // 提取具有最大特征值的特征向量，这是点集的主要方向
            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
            Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y,
                                       pc_in->points[i].z);
            // 当最大的特征值是次大特征值的3倍以上时
            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
                // 根据几何中心和主方向定义一条线段（用point_a, point_b表示）
                Eigen::Vector3d point_on_line = center;
                Eigen::Vector3d point_a, point_b;
                point_a = 0.1 * unit_direction + point_on_line;
                point_b = -0.1 * unit_direction + point_on_line;
                /*  创建边缘点代价函数，优化的目标是：
                    最小化“当前帧中的当前特征点”到“该特征点坐标在的全局特征点云的邻域中的主方向线段”的误差
                */
                ceres::CostFunction* cost_function =
                    new EdgeAnalyticCostFunction(curr_point, point_a, point_b);
                // ceres::CostFunction* cost_function =
                // new SDFAnalyticCostFunction(curr_point, center);
                // 将残差块加入problem
                problem.AddResidualBlock(cost_function, loss_function,
                                         parameters);
                corner_num++;
            }
        }
    }
    if (corner_num < 20) {
        printf("not enough correct Corner points: [%d] \n", corner_num);
    }
}

void OdomEstimationClass::addSurfCostFactor(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& map_in,
    ceres::Problem& problem,
    ceres::LossFunction* loss_function)
{
    int surf_num = 0;
    for (int i = 0; i < (int)pc_in->points.size(); i++) {
        pcl::PointXYZI point_temp;
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeSurfMap->nearestKSearch(point_temp, 5, pointSearchInd,
                                      pointSearchSqDis);

        Eigen::Matrix<double, 5, 3> matA0;
        Eigen::Matrix<double, 5, 1> matB0 =
            -1 * Eigen::Matrix<double, 5, 1>::Ones();
        if (pointSearchSqDis[4] < 1.0) {
            for (int j = 0; j < 5; j++) {
                matA0(j, 0) = map_in->points[pointSearchInd[j]].x;
                matA0(j, 1) = map_in->points[pointSearchInd[j]].y;
                matA0(j, 2) = map_in->points[pointSearchInd[j]].z;
            }
            // 求解平面法向量
            Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
            // negative_OA_dot_norm = 1/sqrt(sqrt(a^2+b^2+c^2))
            double negative_OA_dot_norm = 1 / norm.norm();
            // 单位法向量
            norm.normalize();
            // 平面公式: ax + by + cz + d = 0, n = (a, b, c), p = (x, y, z)
            // Dis = ||ax+by+cz+d||/sqrt(a^2+b^2+c^2)
            // 检查平面拟合效果
            bool planeValid = true;
            for (int j = 0; j < 5; j++) {
                // if ax+by+cz+d > 0.2, then plane is not fit well
                if (fabs(norm(0) * map_in->points[pointSearchInd[j]].x +
                         norm(1) * map_in->points[pointSearchInd[j]].y +
                         norm(2) * map_in->points[pointSearchInd[j]].z +
                         negative_OA_dot_norm) > 0.2) {
                    planeValid = false;
                    break;
                }
            }
            Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y,
                                       pc_in->points[i].z);
            if (planeValid) {
                ceres::CostFunction* cost_function =
                    new SurfNormAnalyticCostFunction(curr_point, norm,
                                                     negative_OA_dot_norm);
                problem.AddResidualBlock(cost_function, loss_function,
                                         parameters);

                surf_num++;
            }
        }
    }
    if (surf_num < 20) {
        printf("not enough correct Surf points: [%d] \n", surf_num);
    }
}

void OdomEstimationClass::addSDFKPCostFactor(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& map_in,
    ceres::Problem& problem,
    ceres::LossFunction* loss_function)
{
    int kpts_num = 0;
    for (int i = 0; i < (int)pc_in->points.size(); i++) {
        pcl::PointXYZI point_temp;
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        // 从全局特征点云中寻找与当前点（新帧）最近的5个点，并返回索引与距离（按平方距离升序排列）
        kdtreeSDFMap->nearestKSearch(point_temp, 3, pointSearchInd,
                                     pointSearchSqDis);
        // 最远的邻域点满足阈值(1.0)，则认为邻域点足够近
        if (pointSearchSqDis[2] < 2) {
            std::vector<Eigen::Vector3d> nearCorners;
            Eigen::Vector3d center(0, 0, 0);
            for (int j = 0; j < 3; j++) {
                Eigen::Vector3d tmp(map_in->points[pointSearchInd[j]].x,
                                    map_in->points[pointSearchInd[j]].y,
                                    map_in->points[pointSearchInd[j]].z);
                center = center + tmp;
                nearCorners.push_back(tmp);
            }
            // 计算邻域点的几何中心
            center = center / 3.0;
            Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y,
                                       pc_in->points[i].z);
            ceres::CostFunction* cost_function =
                new SDFAnalyticCostFunction(curr_point, center);
            problem.AddResidualBlock(cost_function, loss_function, parameters);
            kpts_num++;
        }
    }
    if (kpts_num < 20) {
        printf("not enough correct SDF points: [%d] \n", kpts_num);
    }
}

void OdomEstimationClass::addPointsToMap(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& downsampledEdgeCloud,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& downsampledSurfCloud,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& sdfkptsCloud)
{
    for (int i = 0; i < (int)downsampledEdgeCloud->points.size(); i++) {
        pcl::PointXYZI point_temp;
        pointAssociateToMap(&downsampledEdgeCloud->points[i], &point_temp);
        laserCloudCornerMap->push_back(point_temp);
    }

    for (int i = 0; i < (int)downsampledSurfCloud->points.size(); i++) {
        pcl::PointXYZI point_temp;
        pointAssociateToMap(&downsampledSurfCloud->points[i], &point_temp);
        laserCloudSurfMap->push_back(point_temp);
    }

    for (int i = 0; i < (int)sdfkptsCloud->points.size(); i++) {
        pcl::PointXYZI point_temp;
        pointAssociateToMap(&sdfkptsCloud->points[i], &point_temp);
        sdfKeyPointsMap->push_back(point_temp);
    }
    double x_min = +odom.translation().x() - 100;
    double y_min = +odom.translation().y() - 100;
    double z_min = +odom.translation().z() - 100;
    double x_max = +odom.translation().x() + 100;
    double y_max = +odom.translation().y() + 100;
    double z_max = +odom.translation().z() + 100;

    // ROS_INFO("size : %f,%f,%f,%f,%f,%f", x_min, y_min, z_min,x_max, y_max,
    // z_max);
    cropBoxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
    cropBoxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));
    cropBoxFilter.setNegative(false);

    pcl::PointCloud<pcl::PointXYZI>::Ptr tmpCorner(
        new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr tmpSurf(
        new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr tmpSDFkpts(
        new pcl::PointCloud<pcl::PointXYZI>());
    cropBoxFilter.setInputCloud(laserCloudSurfMap);
    cropBoxFilter.filter(*tmpSurf);
    cropBoxFilter.setInputCloud(laserCloudCornerMap);
    cropBoxFilter.filter(*tmpCorner);
    cropBoxFilter.setInputCloud(sdfKeyPointsMap);
    cropBoxFilter.filter(*tmpSDFkpts);

    downSizeFilterSurf.setInputCloud(tmpSurf);
    downSizeFilterSurf.filter(*laserCloudSurfMap);
    downSizeFilterEdge.setInputCloud(tmpCorner);
    downSizeFilterEdge.filter(*laserCloudCornerMap);
    downSizeFilterEdge.setInputCloud(tmpSDFkpts);
    downSizeFilterEdge.filter(*sdfKeyPointsMap);
}

void OdomEstimationClass::getMap(
    pcl::PointCloud<pcl::PointXYZI>::Ptr& laserCloudMap)
{
    *laserCloudMap += *laserCloudSurfMap;
    *laserCloudMap += *laserCloudCornerMap;
}

OdomEstimationClass::OdomEstimationClass() {}
