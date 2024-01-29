#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

int main()
{
    Eigen::Matrix<float, 3, 4> P2;
    P2 << 7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02,
        4.485728000000e+01, 0.000000000000e+00, 7.215377000000e+02,
        1.728540000000e+02, 2.163791000000e-01, 0.000000000000e+00,
        0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03;

    Eigen::Matrix4f R0_rect;
    R0_rect << 9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03,
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
    cv::Mat image =
        cv::imread("/home/yuxuanzhao/Desktop/kitti-velo2cam/"
                   "data_object_image_2/testing/image_2/000007.png");
    int IMG_H              = image.rows;
    int IMG_W              = image.cols;
    std::string pointcloud = "/home/yuxuanzhao/Desktop/kitti-velo2cam/"
                             "data_object_velodyne/testing/velodyne/000007.bin";
    // 打开文件
    std::ifstream file(pointcloud, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file\n";
        return -1;
    }
    // 获取文件大小并读取数据
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        std::cerr << "Failed to read file\n";
        return -1;
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

    // 计算矩阵乘积
    Eigen::MatrixXf cam = P2 * R0_rect * Tr_velo_to_cam * velo_filtered;
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

    return 0;
}
