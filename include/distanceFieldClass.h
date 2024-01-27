#ifndef _DISTANCE_FIELD_CLASS_H_
#define _DISTANCE_FIELD_CLASS_H_

#include "signed_distance_field_2d.h"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

class DistanceField {
  public:
    DistanceField() {}
    DistanceField(float sdf_x_bound,
                  float sdf_y_bound,
                  float sdf_z_lower,
                  float sdf_z_upper,
                  float sdf_resolution)
        : sdf_x_bound_(sdf_x_bound), sdf_y_bound_(sdf_y_bound),
          sdf_z_lower_(sdf_z_lower), sdf_z_upper_(sdf_z_upper),
          sdf_resolution_(sdf_resolution)
    {
        pointPerRow_ = 2 * sdf_y_bound_ / sdf_resolution_ + 1;
        pointPerCol_ = 2 * sdf_x_bound_ / sdf_resolution_ + 1;
    }
    DistanceField(pcl::PointCloud<pcl::PointXYZI>::Ptr distance_field)
        : distance_field_(distance_field)
    {
    }
    void getSDFPointCloud(
        pcl::PointCloud<pcl::PointXYZI>::Ptr& signed_distance_field, float sdfmin, float sdfmax);
    void
    detectKeypoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& distance_field,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr& sdf_keypoints);

    void computeSignedDistanceField(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

    void detectGaussianCurvatureAndEigen(const cv::Mat& src,
                                         int ksize,
                                         cv::Mat& dst_doh,
                                         cv::Mat& dst_eigenvalue1,
                                         cv::Mat& dst_eigenvalue2);
    void gridMap2PointCloud(const std::vector<double>& gridmap,
                            float sdfmin,
                            float sdfmax,
                            pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_out);
    std::vector<double> mat2GridMap(const cv::Mat& mat_in);
    void toMat(cv::Mat& mat_out);
    void keypointDetection(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_out);
    void find_extrema_points(const cv::Mat& src_doh,
                             std::vector<cv::Point>& dst_extrema_points);
    void
    classify_extrema_points(const std::vector<cv::Point>& src_extrema_points,
                            cv::Mat& src_eigenvalue1,
                            cv::Mat& src_eigenvalue2,
                            std::vector<std::vector<cv::Point>>& dst);

    void cvpt2PointCloud(const std::vector<cv::Point>& points,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_out,
                         float z_val);

  private:
    pcl::PointCloud<pcl::PointXYZI>::Ptr distance_field_;
    float sdf_x_bound_;
    float sdf_y_bound_;
    float sdf_z_lower_;
    float sdf_z_upper_;
    float sdf_resolution_;
    int pointPerRow_;
    int pointPerCol_;
    std::shared_ptr<planning::SignedDistanceField2D> sdf_ptr_;

    void getNeighborsIds(int index, std::vector<int>& neighbors_ids);
    Eigen::Matrix2f computeHessianAtPoint(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& distance_field,
        int point_index,
        const std::vector<int>& neighbors_ids);
};

#endif  // _DISTANCE_FIELD_H_