#include "distanceFieldClass.h"
#include "signed_distance_field_2d.h"

// void DistanceField::computeDistanceField(
//     const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
//     pcl::PointCloud<pcl::PointXYZI>::Ptr& distance_field)
// {
//     Eigen::Vector3f grid_min(-sdf_x_bound_, -sdf_y_bound_, sdf_z_lower_);
//     Eigen::Vector3f grid_max(sdf_x_bound_, sdf_y_bound_, sdf_z_upper_);
//     // 创建 KdTree 以加速最近邻搜索
//     pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
//     kdtree.setInputCloud(cloud);
//     // 遍历体素网格
//     for (float x = grid_min.x(); x <= grid_max.x(); x += sdf_resolution_) {
//         for (float y = grid_min.y(); y <= grid_max.y(); y += sdf_resolution_) {
//             // for (float z = grid_min.z(); z <= grid_max.z(); z += sdf_resolution_) {
//                 float z = sdf_z_lower_;
//                 pcl::PointXYZI searchPoint;
//                 searchPoint.x = x;
//                 searchPoint.y = y;
//                 searchPoint.z = z;
//                 // 执行最近邻搜索
//                 std::vector<int> pointIdxNKNSearch(1);
//                 std::vector<float> pointNKNSquaredDistance(1);
//                 if (kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch,
//                                           pointNKNSquaredDistance) > 0) {
//                     // 计算距离并存储在距离场中
//                     pcl::PointXYZI point;
//                     point.x         = x;
//                     point.y         = y;
//                     point.z         = z;
//                     point.intensity = std::sqrt(pointNKNSquaredDistance[0]);
//                     distance_field->push_back(point);
//                 }
//             // }
//         }
//     }
// }

void DistanceField::ComputeSignedDistanceField(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_in,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& signed_distance_field)
{
    // Get pointcloud size
    int pointPerRow = 2 * sdf_y_bound_ / sdf_resolution_ + 1;
    int pointPerCol = 2 * sdf_x_bound_ / sdf_resolution_ + 1;
    std::array<int, 2> size{pointPerRow, pointPerCol};
    planning::GridMap2D<uint8_t> grid_map;
    grid_map.set_cell_number(size);
    grid_map.set_resolution(std::array<double, 2>{1, 1});
    // fill specify z into grid_map
    for (int i = 0; i < (int)cloud_in->points.size(); ++i) {
        float ptx = cloud_in->points[i].x;
        float pty = cloud_in->points[i].y;
        float ptz = cloud_in->points[i].z;
        // filter pointcloud with sdf boundary
        if (ptx > -sdf_x_bound_ && ptx < sdf_x_bound_ && pty > -sdf_y_bound_ &&
            pty < sdf_y_bound_ && ptz > sdf_z_lower_ &&
            ptz < sdf_z_lower_ + sdf_resolution_) {
            int x = pty + pointPerRow / 2;
            int y = ptx + pointPerCol / 2;
            grid_map.SetValue(Eigen::Vector2i(x, y), 1);
        }
    }
    planning::SignedDistanceField2D sdf(std::move(grid_map));
    sdf.UpdateSDF();
    auto sdf_data = sdf.esdf().data();
    // convert esdf into pointcloud
    for (int address = 0; address < (int)sdf_data.size(); address++) {
        // address to index
        int x = address % pointPerRow;
        int y = address / pointPerRow;
        pcl::PointXYZI point;
        point.x         = y - pointPerCol / 2;
        point.y         = x - pointPerRow / 2;
        point.z         = (2 * sdf_z_lower_ + sdf_resolution_) / 2;
        point.intensity = sdf_data[address];
        signed_distance_field->push_back(point);
    }
}
// 计算给定点的Hessian矩阵
Eigen::Matrix2f DistanceField::computeHessianAtPoint(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& distance_field,
    int point_index,
    const std::vector<int>& neighbors_ids)
{
    // 根据中心差分估算偏导数
    int delta_x     = 3;
    int delta_y     = 3;
    auto& right     = distance_field->points[neighbors_ids[0]];
    auto& left      = distance_field->points[neighbors_ids[1]];
    auto& up        = distance_field->points[neighbors_ids[2]];
    auto& down      = distance_field->points[neighbors_ids[3]];
    auto& upright   = distance_field->points[neighbors_ids[4]];
    auto& upleft    = distance_field->points[neighbors_ids[5]];
    auto& downright = distance_field->points[neighbors_ids[6]];
    auto& downleft  = distance_field->points[neighbors_ids[7]];
    float dxx =
        (up.intensity - 2 * distance_field->points[point_index].intensity +
         down.intensity) /
        (delta_x * delta_x);
    float dyy =
        (right.intensity - 2 * distance_field->points[point_index].intensity +
         left.intensity) /
        (delta_y * delta_y);
    float dxy = (upright.intensity - upleft.intensity - downright.intensity +
                 downleft.intensity) /
                (4 * delta_x * delta_y);
    // 构造Hessian矩阵
    Eigen::Matrix2f hessian;
    hessian << dxx, dxy, dxy, dyy;
    return hessian;
}

void DistanceField::getNeighborsIds(int index, std::vector<int>& neighbors_ids)
{
    neighbors_ids.push_back(index - 1);            // right
    neighbors_ids.push_back(index + 1);            // left
    neighbors_ids.push_back(index + pointPerRow);  // up
    neighbors_ids.push_back(index - pointPerRow);  // down

    neighbors_ids.push_back(index + pointPerRow - 1);  // up + right
    neighbors_ids.push_back(index + pointPerRow + 1);  // up + left
    neighbors_ids.push_back(index - pointPerRow - 1);  // down + right
    neighbors_ids.push_back(index - pointPerRow + 1);  // down + left
}
// 主方法：检测关键点
void DistanceField::detectKeypoints(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& distance_field,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& sdf_keypoints)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr sdf_curvature(
        new pcl::PointCloud<pcl::PointXYZI>());

    // compute curvature
    for (int index = 0; index < (int)distance_field->points.size(); ++index) {
        int rowIdx = index / pointPerRow;
        int colIdx = index % pointPerRow;
        std::vector<int> neighbors_ids;
        pcl::PointXYZI point;
        point.x = distance_field->points[index].x;
        point.y = distance_field->points[index].y;
        point.z = distance_field->points[index].z;
        // check index
        if (rowIdx > 0 && rowIdx < pointPerCol - 1 && colIdx > 0 &&
            colIdx < pointPerRow - 1) {
            getNeighborsIds(index, neighbors_ids);
            Eigen::Matrix2f hessian =
                computeHessianAtPoint(distance_field, index, neighbors_ids);
            float doh =
                hessian(0, 0) * hessian(1, 1) - hessian(0, 1) * hessian(1, 0);
            point.intensity = doh;
            sdf_curvature->push_back(point);
        }
        else {
            point.intensity = 0;
            sdf_curvature->push_back(point);
        }
    }
    // search local maximum curvature
    for (int index = 0; index < (int)sdf_curvature->points.size(); index++) {
        int rowIdx = index / pointPerRow;
        int colIdx = index % pointPerRow;
        std::vector<int> neighbors_ids;
        // check index
        if (rowIdx > 0 && rowIdx < pointPerCol - 1 && colIdx > 0 &&
            colIdx < pointPerRow - 1) {
            getNeighborsIds(index, neighbors_ids);
            bool islocalMaximum{true};
            for (auto& neighbor_id : neighbors_ids) {
                if (sdf_curvature->points[index].intensity <=
                    sdf_curvature->points[neighbor_id].intensity) {
                    islocalMaximum = false;
                    break;
                }
            }
            if (islocalMaximum) {
                pcl::PointXYZI point;
                point.x         = distance_field->points[index].x;
                point.y         = distance_field->points[index].y;
                point.z         = distance_field->points[index].z;
                point.intensity = 1;
                sdf_keypoints->push_back(point);
            }
        }
    }
}