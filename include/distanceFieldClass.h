#ifndef _DISTANCE_FIELD_CLASS_H_
#define _DISTANCE_FIELD_CLASS_H_

#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

class DistanceField {
  public:
    DistanceField() {}
    DistanceField(int sdf_x_bound,
                  int sdf_y_bound,
                  int sdf_z_lower,
                  int sdf_z_upper,
                  int sdf_resolution)
        : sdf_x_bound_(sdf_x_bound), sdf_y_bound_(sdf_y_bound),
          sdf_z_lower_(sdf_z_lower), sdf_z_upper_(sdf_z_upper),
          sdf_resolution_(sdf_resolution)
    {
        pointPerRow = 2 * sdf_y_bound_ / sdf_resolution_ + 1;
        pointPerCol = 2 * sdf_x_bound_ / sdf_resolution_ + 1;
    }
    DistanceField(pcl::PointCloud<pcl::PointXYZI>::Ptr distance_field)
        : distance_field_(distance_field)
    {
    }
    void
    detectKeypoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& distance_field,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr& sdf_keypoints);
    void getNeighborsIds(int index, std::vector<int>& neighbors_ids);
    Eigen::Matrix2f computeHessianAtPoint(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& distance_field,
        int point_index,
        const std::vector<int>& neighbors_ids);
    void
    computeDistanceField(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                         float resolution,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr& distance_field);

  private:
    pcl::PointCloud<pcl::PointXYZI>::Ptr distance_field_;
    int sdf_x_bound_;
    int sdf_y_bound_;
    int sdf_z_lower_;
    int sdf_z_upper_;
    int sdf_resolution_;
    int pointPerRow;
    int pointPerCol;
};

#endif  // _DISTANCE_FIELD_H_