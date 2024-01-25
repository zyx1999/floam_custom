/* Copyright 2021 Unity-Drive Inc. All rights reserved */

#include "signed_distance_field_2d.h"
#include <chrono>

// #include "common/matplotlib-cpp/matplotlibcpp.h"
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace planning;
using namespace std;
using namespace Eigen;

void TestBasic();
void TestSignedDistance();
void TestSignedPolyLines();
void TestSignedGradient();

int main(int argc, char const *argv[]) {
  TestBasic();
  TestSignedDistance();
  TestSignedPolyLines();
  TestSignedGradient();
  return 0;
}

void TestBasic() {
  int dim = 10;
  std::array<int, 2> size{dim, dim};

  GridMap2D<uint8_t> grid_map;
  grid_map.set_cell_number(size);
  grid_map.set_origin(std::array<double, 2>{0.0, 0.0});
  grid_map.set_resolution(std::array<double, 2>{1, 1});

  for (int i = 0; i < dim; ++i) {
    grid_map.SetValue(Eigen::Vector2i(i, i), 1);
  }
  // cv::imshow("grid map", grid_map.BinaryImage());
  SignedDistanceField2D sdf(std::move(grid_map));
  sdf.UpdateSDF();

  auto esdf = sdf.esdf();

  cout << esdf.Matrix() << endl;
  // cv::imshow("sdf", sdf.esdf().ImageSec());
  // cv::waitKey(0);
  // EXPECT_EQ(esdf.GetValue(Vector2i(0, 2)), 0.1 * std::sqrt(2));
  // EXPECT_EQ(esdf.GetValue(Vector2i(0, 1)), 0.1 * 1);
}
void TestSignedDistance() {
  int dim = 500;
  std::array<int, 2> size{800, 200};

  GridMap2D<uint8_t> grid_map;
  grid_map.set_cell_number(size);
  grid_map.set_origin(std::array<double, 2>{0.0, 0.0});
  grid_map.set_resolution(std::array<double, 2>{0.1, 0.1});

  grid_map.FillCircle(Eigen::Vector2d(10, 10), 5);
  grid_map.FillConvexPoly(vector_Eigen<Vector2d>{
      Vector2d(30, 15), Vector2d(40, 15), Vector2d(40, 20), Vector2d(30, 20)});
  grid_map.FillPoly(vector_Eigen<Vector2d>{Vector2d(0, 20), Vector2d(20, 20),
                                           Vector2d(25, 25), Vector2d(15, 22),
                                           Vector2d(0, 30)});
  cv::imshow("grid map", grid_map.BinaryImage());

  SignedDistanceField2D sdf(std::move(grid_map));
  auto t0 = chrono::high_resolution_clock::now();

  sdf.UpdateSDF();
  auto t1 = chrono::high_resolution_clock::now();

  double total_ms =
      chrono::duration_cast<chrono::microseconds>(t1 - t0).count() / 1000.0;

  cout << "time for 500x500 sdf: " << total_ms << " ms" << endl;
  cv::imshow("sdf", sdf.esdf().ImageSec());
  cv::waitKey(0);
}
void TestSignedPolyLines() {
  std::array<int, 2> size{800, 200};
  GridMap2D<uint8_t> grid_map;
  grid_map.set_cell_number(size);
  grid_map.set_origin(std::array<double, 2>{0.0, 0.0});
  grid_map.set_resolution(std::array<double, 2>{0.1, 0.1});

  vector_Eigen<Vector2d> points;
  for (int i = 0; i < 140; ++i) {
    points.emplace_back(Eigen::Vector2d(0.5 * i, 10 + 3 * std::sin(0.5 * i)));
  }

  grid_map.PolyLine(points);
  cv::imshow("f(x) = 3*sin(x)", grid_map.BinaryImage());
  SignedDistanceField2D sdf(std::move(grid_map));
  sdf.UpdateSDF();
  cv::imshow("sdf", sdf.esdf().ImageSec());
  cv::waitKey(0);
}
void TestSignedGradient() {
  int dim = 50;
  std::array<int, 2> size{dim, dim};

  GridMap2D<uint8_t> grid_map;
  grid_map.set_cell_number(size);
  grid_map.set_origin(std::array<double, 2>{0.0, 0.0});
  grid_map.set_resolution(std::array<double, 2>{1.0, 1.0});
  grid_map.FillCircle(Eigen::Vector2d(25, 25), 10);
  grid_map.FillPoly(vector_Eigen<Vector2d>{Vector2d(10, 10), Vector2d(20, 10),
                                           Vector2d(20, 20), Vector2d(10, 20)});

  SignedDistanceField2D sdf(std::move(grid_map));
  sdf.UpdateSDF();

  auto esdf = sdf.esdf();

  std::vector<double> x, y, u, v;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      x.emplace_back(i + 0.5);
      y.emplace_back(j + 0.5);
      Eigen::Vector2d grad;
      esdf.GetValueBilinear(Eigen::Vector2d(i + 0.5, j + 0.5), &grad);
      u.emplace_back(grad.x());
      v.emplace_back(grad.y());
    }
  }

  auto image = esdf.ImageSec();
  cv::Mat scaled_image;
  cv::resize(image, scaled_image, {0, 0}, 10.0, 10.0);
  cv::imshow("sdf", scaled_image);
  cv::waitKey(0);

  // namespace plt = matplotlibcpp;
  // plt::quiver(x, y, u, v);
  // plt::axis("equal");
  // plt::show();
}