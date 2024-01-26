#include "distanceFieldClass.h"

void DistanceField::toMat(cv::Mat& mat_out)
{
    auto copy_data = sdf_ptr_->esdf().data();
    auto cell_num  = sdf_ptr_->esdf().cell_num();
    // cell_num[0] == pointPerRow, cell_num[1] = pointPerCol
    cv::Mat grid_mat(cell_num[1], cell_num[0], CV_32FC1, copy_data.data());
    mat_out = grid_mat;
}

std::vector<double> DistanceField::mat2GridMap(const cv::Mat& mat_in)
{
    std::vector<double> gridmap;
    for (int i = 0; i < mat_in.rows; ++i) {
        for (int j = 0; j < mat_in.cols; ++j) {
            double val = mat_in.at<float>(i, j);
            gridmap.push_back(val);
        }
    }
    return gridmap;
}

void DistanceField::gridMap2PointCloud(
    const std::vector<double>& gridmap,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_out)
{
    // auto& sdf_data = sdf_ptr_->esdf().data();
    for (int address = 0; address < (int)gridmap.size(); address++) {
        // address to index
        int x = address % pointPerRow_;
        int y = address / pointPerRow_;
        pcl::PointXYZI point;
        point.x         = y - pointPerCol_ / 2;
        point.y         = x - pointPerRow_ / 2;
        point.z         = (2 * sdf_z_lower_ + sdf_resolution_) / 2;
        point.intensity = gridmap[address];
        cloud_out->push_back(point);
    }
}
void DistanceField::cvpt2PointCloud(
    const std::vector<cv::Point>& points,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_out,
    float z_val)
{
    for (const auto& pt : points) {
        pcl::PointXYZI point;
        point.x         = pt.x - pointPerCol_ / 2;
        point.y         = pt.y - pointPerRow_ / 2;
        point.z         = (2 * sdf_z_lower_ + sdf_resolution_) / 2;
        point.intensity = z_val;
        cloud_out->push_back(point);
    }
}
void DistanceField::keypointDetection(
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_out)
{
    // cv::Mat src_sdf_ = cv_ptr->image;
    // cv::Mat doh_, eigenValue1_, eigenValue2_;
    // std::vector<cv::Point> extrema_points_;
    // std::vector<std::vector<cv::Point>> classified_extrema_points_;

    // // keypoint detector
    // detectGaussianCurvatureAndEigen(src_sdf_, 3, doh_, eigenValue1_,
    //                                     eigenValue2_);
    // find_extrema_points(doh_, extrema_points_);
    // classify_extrema_points(extrema_points_, eigenValue1_, eigenValue2_,
    //                         classified_extrema_points_);
    cv::Mat sdf_mat;
    cv::Mat doh, eigen_1, eigen_2;
    std::vector<cv::Point> extrema_points;
    std::vector<std::vector<cv::Point>> classified_extrema_points;

    toMat(sdf_mat);
    detectGaussianCurvatureAndEigen(sdf_mat, 3, doh, eigen_1, eigen_2);
    find_extrema_points(doh, extrema_points);
    // cvpt2PointCloud(extrema_points, cloud_out, 1);
    classify_extrema_points(extrema_points, eigen_1, eigen_2,
                            classified_extrema_points);
    for(int i = 0; i < 4; i++){
        std::cout<<"Class ["<<i<<"] = "<<classified_extrema_points[i].size()<<std::endl;
    }
    cvpt2PointCloud(classified_extrema_points[1], cloud_out, 1);
    // gridMap2PointCloud(mat2GridMap(doh), cloud_out);
}

void DistanceField::classify_extrema_points(
    const std::vector<cv::Point>& src_extrema_points,
    cv::Mat& src_eigenvalue1,
    cv::Mat& src_eigenvalue2,
    std::vector<std::vector<cv::Point>>& dst)
{
    dst = std::vector<std::vector<cv::Point>>(4);
    // 0: extrema max; 1: extrema min, 2: extrema saddle
    for (const auto& pt : src_extrema_points) {
        float ev1 = src_eigenvalue1.at<float>(pt.x, pt.y);
        float ev2 = src_eigenvalue2.at<float>(pt.x, pt.y);
        // local maximal
        if (ev1 < 0 && ev2 < 0) {
            dst[0].push_back(pt);
        }
        // local minimal
        if (ev1 > 0 && ev2 > 0) {
            dst[1].push_back(pt);
        }
        // saddle
        if (ev1 * ev2 < 0) {
            dst[2].push_back(pt);
        }
        // critical
        if (ev1 * ev2 == 0) {
            dst[3].push_back(pt);
        }
    }
}

void DistanceField::find_extrema_points(
    const cv::Mat& src_doh,
    std::vector<cv::Point>& dst_extrema_points)
{
    // loop through each pixel in the image
    cv::Mat extrema = cv::Mat::zeros(src_doh.size(), CV_8UC1);
    int radius_     = 10;
    for (int i = radius_; i < src_doh.rows - radius_; i++) {
        for (int j = radius_; j < src_doh.cols - radius_; j++) {
            // check if the current pixel is an extremum
            float value      = src_doh.at<float>(i, j);
            bool is_extremum = true;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    if (value < src_doh.at<float>(i + k, j + l)) {
                        is_extremum = false;
                        break;
                    }
                }
                if (!is_extremum) {
                    break;
                }
            }
            if (is_extremum) {
                extrema.at<uchar>(i, j) = 255;
            }
        }
    }
    cv::Mat extrema_trans;
    cv::transpose(extrema, extrema_trans);
    cv::findNonZero(extrema_trans, dst_extrema_points);
}
void DistanceField::detectGaussianCurvatureAndEigen(const cv::Mat& src,
                                                    int ksize,
                                                    cv::Mat& dst_doh,
                                                    cv::Mat& dst_eigenvalue1,
                                                    cv::Mat& dst_eigenvalue2)
{
    cv::Mat gaussBlur, dx, dy, dxx, dxy, dyy;
    // 1. gaussian blue
    // cv::GaussianBlur(src, gaussBlur, cv::Size(5, 5), 0, 0);
    gaussBlur = src;
    // 2. hessian
    cv::Sobel(gaussBlur, dx, -1, 1, 0, ksize);
    cv::Sobel(gaussBlur, dy, -1, 0, 1, ksize);
    cv::Sobel(dx, dxx, -1, 1, 0, ksize);
    cv::Sobel(dx, dxy, -1, 0, 1, ksize);
    cv::Sobel(dy, dyy, -1, 0, 1, ksize);

    // 3. DoH & gaussian curvature & eigen
    dst_doh.create(src.size(), CV_32FC1);
    dst_eigenvalue1.create(src.size(), CV_32FC1);
    dst_eigenvalue2.create(src.size(), CV_32FC1);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // float fx = dx.at<float>(i, j);
            // float fy = dy.at<float>(i, j);
            float fxx = dxx.at<float>(i, j);
            float fxy = dxy.at<float>(i, j);
            float fyy = dyy.at<float>(i, j);
            // DoH
            float doh = fxx * fyy - fxy * fxy;
            // gaussian curvature
            // float k = doh / pow(fx * fx + fy * fy + 1e-8, 2);

            // eigen value & eigen vector
            cv::Mat hessianAtEachPoint_(2, 2, CV_32FC1);
            hessianAtEachPoint_.at<float>(0, 0) = fxx;
            hessianAtEachPoint_.at<float>(0, 1) = fxy;
            hessianAtEachPoint_.at<float>(1, 0) = fxy;
            hessianAtEachPoint_.at<float>(1, 1) = fyy;
            cv::Mat eigenValue_, eigenVector_;
            cv::eigen(hessianAtEachPoint_, eigenValue_, eigenVector_);
            // float gaussCurv_ = eigenValue_.at<float>(0, 0) *
            // eigenValue_.at<float>(1, 0); gaussianCurvature_.at<float>(i, j) =
            // gaussCurv_;

            dst_doh.at<float>(i, j)         = doh;
            dst_eigenvalue1.at<float>(i, j) = eigenValue_.at<float>(0, 0);
            dst_eigenvalue2.at<float>(i, j) = eigenValue_.at<float>(1, 0);

            // dst.at<float>(i, j) = k;
            // dst.at<float>(i, j) = gaussCurv_;
        }
    }
}

void DistanceField::getSDFPointCloud(
    pcl::PointCloud<pcl::PointXYZI>::Ptr& signed_distance_field)
{
    auto& sdf_data = sdf_ptr_->esdf().data();
    gridMap2PointCloud(sdf_data, signed_distance_field);
}

void DistanceField::computeSignedDistanceField(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_in)
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
    sdf_ptr_ =
        std::make_shared<planning::SignedDistanceField2D>(std::move(grid_map));
    sdf_ptr_->UpdateSDF();
    // sdf_data_ = sdf.esdf().data();
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
    neighbors_ids.push_back(index - 1);             // right
    neighbors_ids.push_back(index + 1);             // left
    neighbors_ids.push_back(index + pointPerRow_);  // up
    neighbors_ids.push_back(index - pointPerRow_);  // down

    neighbors_ids.push_back(index + pointPerRow_ - 1);  // up + right
    neighbors_ids.push_back(index + pointPerRow_ + 1);  // up + left
    neighbors_ids.push_back(index - pointPerRow_ - 1);  // down + right
    neighbors_ids.push_back(index - pointPerRow_ + 1);  // down + left
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
        int rowIdx = index / pointPerRow_;
        int colIdx = index % pointPerRow_;
        std::vector<int> neighbors_ids;
        pcl::PointXYZI point;
        point.x = distance_field->points[index].x;
        point.y = distance_field->points[index].y;
        point.z = distance_field->points[index].z;
        // check index
        if (rowIdx > 0 && rowIdx < pointPerCol_ - 1 && colIdx > 0 &&
            colIdx < pointPerRow_ - 1) {
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
        int rowIdx = index / pointPerRow_;
        int colIdx = index % pointPerRow_;
        std::vector<int> neighbors_ids;
        // check index
        if (rowIdx > 0 && rowIdx < pointPerCol_ - 1 && colIdx > 0 &&
            colIdx < pointPerRow_ - 1) {
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