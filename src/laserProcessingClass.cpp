// Author of FLOAM: Wang Han
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro
#include "laserProcessingClass.h"

void LaserProcessingClass::init(lidar::Lidar lidar_param_in)
{
    lidar_param = lidar_param_in;
}

void LaserProcessingClass::featureExtraction(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
    pcl::PointCloud<pcl::PointXYZI>::Ptr&       pc_out_edge,
    pcl::PointCloud<pcl::PointXYZI>::Ptr&       pc_out_surf)
{
    // 按照lidar线数初始化空vector，用于保存XYZI点云指针
    int N_SCANS = lidar_param.num_lines;
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> laserCloudScans(N_SCANS);
    for (int i = 0; i < N_SCANS; ++i) {
        laserCloudScans[i] =
            boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    }
    // loop1：
    // 遍历输入的点云pc_in，将点云中的点根据计算出的角度分配到对应的激光线中。
    for (int i = 0; i < (int)pc_in->points.size(); i++) {
        int scanID = 0;
        // 计算当前点云点到原点的距离
        double distance = sqrt(pc_in->points[i].x * pc_in->points[i].x +
                               pc_in->points[i].y * pc_in->points[i].y);
        // 只处理满足距离要求的点 [min < dis < max]
        if (distance < lidar_param.min_distance ||
            distance > lidar_param.max_distance)
            continue;
        // 计算平面上斜直线的倾角：atan反正切函数计算弧度，然后转为角度
        double angle = atan(pc_in->points[i].z / distance) * 180 / M_PI;
        // 按lidar线数计算当前点落在哪条线。
        if (N_SCANS == 16) {
            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID > (N_SCANS - 1) || scanID < 0) {
                continue;
            }
        }
        else if (N_SCANS == 32) {
            scanID = int((angle + 92.0 / 3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0) {
                continue;
            }
        }
        else if (N_SCANS == 64) {
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);
            // velodyne-hdl-64E激光雷达的垂直视场角为+2~-24.8，超过这个范围的角度都是无效角度
            if (angle > 2 || angle < -24.33 || scanID > 63 || scanID < 0) {
                continue;
            }
        }
        else {
            printf("wrong scan number\n");
        }
        laserCloudScans[scanID]->push_back(pc_in->points[i]);
    }
    // loop2：
    for (int i = 0; i < N_SCANS; i++) {
        // 不处理点数少于131的雷达线
        if (laserCloudScans[i]->points.size() < 131) {
            continue;
        }
        // 计算曲率点，曲率点总数 = 点云总数 - 10
        std::vector<Double2d> cloudCurvature;
        int total_points = laserCloudScans[i]->points.size() - 10;
        // 启发式的曲率估计，而非几何曲率。曲率在平滑处数值较小，在角点或边缘处数值较大
        for (int j = 5; j < (int)laserCloudScans[i]->points.size() - 5; j++) {
            double diffX = laserCloudScans[i]->points[j - 5].x +
                           laserCloudScans[i]->points[j - 4].x +
                           laserCloudScans[i]->points[j - 3].x +
                           laserCloudScans[i]->points[j - 2].x +
                           laserCloudScans[i]->points[j - 1].x -
                           10 * laserCloudScans[i]->points[j].x +
                           laserCloudScans[i]->points[j + 1].x +
                           laserCloudScans[i]->points[j + 2].x +
                           laserCloudScans[i]->points[j + 3].x +
                           laserCloudScans[i]->points[j + 4].x +
                           laserCloudScans[i]->points[j + 5].x;
            double diffY = laserCloudScans[i]->points[j - 5].y +
                           laserCloudScans[i]->points[j - 4].y +
                           laserCloudScans[i]->points[j - 3].y +
                           laserCloudScans[i]->points[j - 2].y +
                           laserCloudScans[i]->points[j - 1].y -
                           10 * laserCloudScans[i]->points[j].y +
                           laserCloudScans[i]->points[j + 1].y +
                           laserCloudScans[i]->points[j + 2].y +
                           laserCloudScans[i]->points[j + 3].y +
                           laserCloudScans[i]->points[j + 4].y +
                           laserCloudScans[i]->points[j + 5].y;
            double diffZ = laserCloudScans[i]->points[j - 5].z +
                           laserCloudScans[i]->points[j - 4].z +
                           laserCloudScans[i]->points[j - 3].z +
                           laserCloudScans[i]->points[j - 2].z +
                           laserCloudScans[i]->points[j - 1].z -
                           10 * laserCloudScans[i]->points[j].z +
                           laserCloudScans[i]->points[j + 1].z +
                           laserCloudScans[i]->points[j + 2].z +
                           laserCloudScans[i]->points[j + 3].z +
                           laserCloudScans[i]->points[j + 4].z +
                           laserCloudScans[i]->points[j + 5].z;
            Double2d distance(j, diffX * diffX + diffY * diffY + diffZ * diffZ);
            cloudCurvature.emplace_back(distance);
        }
        // 将曲率点分为6个扇区，并计算每个扇区的边界
        for (int j = 0; j < 6; j++) {
            int sector_length = (int)(total_points / 6);
            int sector_start  = sector_length * j;
            int sector_end    = sector_length * (j + 1) - 1;
            if (j == 5) {
                sector_end = total_points - 1;
            }
            // 创建子扇区拷贝
            std::vector<Double2d> subCloudCurvature(
                cloudCurvature.begin() + sector_start,
                cloudCurvature.begin() + sector_end);
            // 以第i线点云和子扇区作为输入，计算特征
            featureExtractionFromSector(laserCloudScans[i], subCloudCurvature,
                                        pc_out_edge, pc_out_surf);
        }
    }
}
/*
排序曲率值：
1.首先，函数对 cloudCurvature 中的点按照其曲率值（value）进行升序排序。这意味着曲率较小（更平坦）的点会排在前面，而曲率较大（更尖锐或弯曲）的点会排在后面。
选取边缘点：
2.接下来，从曲率值最大（排序后的列表末尾）的点开始，选取前 20 个曲率最大的点作为边缘点，并将它们添加到 pc_out_edge 点云中。
在选取过程中，使用 picked_points 记录已经被选取的点的索引，以避免重复选取。
3.检查邻近点以避免选取过于靠近的点：
对于每个被选取的边缘点，检查其前后各 5 个邻近点。如果这些邻近点与当前点的差异大于某个阈值（这里是 0.05），则停止在这个方向上的选取。这样可以确保选取的边缘点不会过于集中在局部区域。
4.选取表面点：最后，对于排序后曲率值较小的点（即更平坦的区域），如果这些点尚未被选为边缘点，则将它们添加到 pc_out_surf 点云中，作为表面点。
*/
void LaserProcessingClass::featureExtractionFromSector(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in,
    std::vector<Double2d>&                      cloudCurvature,
    pcl::PointCloud<pcl::PointXYZI>::Ptr&       pc_out_edge,
    pcl::PointCloud<pcl::PointXYZI>::Ptr&       pc_out_surf)
{
    // 按点云的曲率升序排列
    std::sort(
        cloudCurvature.begin(), cloudCurvature.end(),
        [](const Double2d& a, const Double2d& b) { return a.value < b.value; });

    int              largestPickedNum = 0;
    std::vector<int> picked_points; // 查找过程中记录搜索过的下标
    // 从曲率值最大（排序后的列表末尾）的点开始，选取前 20 个曲率最大的点作为边缘点，并将它们添加到 pc_out_edge 点云中。
    for (int i = cloudCurvature.size() - 1; i >= 0; i--) {
        int ind = cloudCurvature[i].id;
        // 当前曲率点索引不存在于picked_points中时，进入if
        if (std::find(picked_points.begin(), picked_points.end(), ind) ==
            picked_points.end()) {
            if (cloudCurvature[i].value <= 0.1) {
                break;
            }

            largestPickedNum++;
            picked_points.push_back(ind);

            if (largestPickedNum <= 20) {
                pc_out_edge->push_back(pc_in->points[ind]);
            }
            else {
                break;
            }
            // 检查邻近点以避免选取过于靠近的点
            for (int k = 1; k <= 5; k++) {
                double diffX =
                    pc_in->points[ind + k].x - pc_in->points[ind + k - 1].x;
                double diffY =
                    pc_in->points[ind + k].y - pc_in->points[ind + k - 1].y;
                double diffZ =
                    pc_in->points[ind + k].z - pc_in->points[ind + k - 1].z;
                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
                    break;
                }
                picked_points.push_back(ind + k);
            }
            for (int k = -1; k >= -5; k--) {
                double diffX =
                    pc_in->points[ind + k].x - pc_in->points[ind + k + 1].x;
                double diffY =
                    pc_in->points[ind + k].y - pc_in->points[ind + k + 1].y;
                double diffZ =
                    pc_in->points[ind + k].z - pc_in->points[ind + k + 1].z;
                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
                    break;
                }
                picked_points.push_back(ind + k);
            }
        }
    }
    // 对于排序后曲率值较小的点（即更平坦的区域），如果这些点尚未被选为边缘点，则将它们添加到 pc_out_surf 点云中，作为表面点。
    for (int i = 0; i <= (int)cloudCurvature.size() - 1; i++) {
        int ind = cloudCurvature[i].id;
        if (std::find(picked_points.begin(), picked_points.end(), ind) ==
            picked_points.end()) {
            pc_out_surf->push_back(pc_in->points[ind]);
        }
    }
}
LaserProcessingClass::LaserProcessingClass() {}

Double2d::Double2d(int id_in, double value_in)
{
    id    = id_in;
    value = value_in;
};

PointsInfo::PointsInfo(int layer_in, double time_in)
{
    layer = layer_in;
    time  = time_in;
};
