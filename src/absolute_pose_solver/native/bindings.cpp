#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <opengv/absolute_pose/AbsoluteAdapterBase.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

namespace py = pybind11;

namespace {

using Array = py::array_t<double, py::array::c_style>;

void require_shape(const Array &array, const std::vector<py::ssize_t> &shape,
                   const char *name) {
  if (array.ndim() != static_cast<py::ssize_t>(shape.size())) {
    throw py::value_error(std::string(name) + " has the wrong rank.");
  }
  for (py::ssize_t axis = 0; axis < array.ndim(); ++axis) {
    if (shape[axis] >= 0 && array.shape(axis) != shape[axis]) {
      throw py::value_error(std::string(name) + " has the wrong shape.");
    }
  }
}

class ArrayAbsoluteAdapter final
    : public opengv::absolute_pose::AbsoluteAdapterBase {
 public:
  ArrayAbsoluteAdapter(const Array &bearings, const Array &points,
                       const Array &offsets, const Array &rotations) {
    const auto bearing_view = bearings.unchecked<2>();
    const auto point_view = points.unchecked<2>();
    const auto offset_view = offsets.unchecked<2>();
    const auto rotation_view = rotations.unchecked<3>();
    const auto count = bearings.shape(0);
    bearings_.reserve(count);
    points_.reserve(count);
    offsets_.reserve(count);
    rotations_.reserve(count);
    for (py::ssize_t i = 0; i < count; ++i) {
      opengv::bearingVector_t bearing;
      opengv::point_t point;
      opengv::translation_t offset;
      opengv::rotation_t rotation;
      for (int row = 0; row < 3; ++row) {
        bearing[row] = bearing_view(i, row);
        point[row] = point_view(i, row);
        offset[row] = offset_view(i, row);
        for (int col = 0; col < 3; ++col) {
          rotation(row, col) = rotation_view(i, row, col);
        }
      }
      bearings_.push_back(bearing);
      points_.push_back(point);
      offsets_.push_back(offset);
      rotations_.push_back(rotation);
    }
  }

  opengv::bearingVector_t getBearingVector(size_t index) const override {
    return bearings_[index];
  }
  double getWeight(size_t) const override { return 1.0; }
  opengv::translation_t getCamOffset(size_t index) const override {
    return offsets_[index];
  }
  opengv::rotation_t getCamRotation(size_t index) const override {
    return rotations_[index];
  }
  opengv::point_t getPoint(size_t index) const override {
    return points_[index];
  }
  size_t getNumberCorrespondences() const override { return points_.size(); }

 private:
  opengv::bearingVectors_t bearings_;
  opengv::points_t points_;
  opengv::translations_t offsets_;
  opengv::rotations_t rotations_;
};

py::tuple solve_ransac_upnp(
    const Array &bearings, const Array &points, const Array &offsets,
    const Array &rotations, bool use_generalized_ransac, double threshold,
    int max_iterations, double probability) {
  require_shape(bearings, {-1, 3}, "bearings_C");
  const auto count = bearings.shape(0);
  require_shape(points, {count, 3}, "points_W");
  require_shape(offsets, {count, 3}, "camera_offsets_B");
  require_shape(rotations, {count, 3, 3}, "camera_rotations_B_C");
  if (count < 4) {
    throw py::value_error("At least four correspondences are required.");
  }
  if (!std::isfinite(threshold) || threshold <= 0.0 || threshold >= 2.0) {
    throw py::value_error("RANSAC threshold must be finite and between 0 and 2.");
  }
  if (max_iterations <= 0 || !std::isfinite(probability) || probability <= 0.0 ||
      probability >= 1.0) {
    throw py::value_error("Invalid RANSAC iteration count or probability.");
  }

  ArrayAbsoluteAdapter adapter(bearings, points, offsets, rotations);
  using Problem =
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem;
  const auto algorithm = use_generalized_ransac ? Problem::GP3P : Problem::KNEIP;
  auto problem = std::make_shared<Problem>(adapter, algorithm, false);
  opengv::sac::Ransac<Problem> ransac;
  ransac.sac_model_ = problem;
  ransac.threshold_ = threshold;
  ransac.max_iterations_ = max_iterations;
  ransac.probability_ = probability;

  const bool ransac_success = ransac.computeModel();
  if (!ransac_success || ransac.inliers_.size() < 3) {
    return py::make_tuple(
        false,
        py::array_t<double>(py::array::ShapeContainer{
            py::ssize_t(0), py::ssize_t(3), py::ssize_t(4)}),
        py::array_t<int>(py::array::ShapeContainer{py::ssize_t(0)}));
  }

  opengv::transformations_t candidates;
  opengv::transformation_t refined_model;
  problem->optimizeModelCoefficients(
      ransac.inliers_, ransac.model_coefficients_, refined_model);
  if (refined_model.allFinite()) {
    candidates.push_back(refined_model);
  } else if (ransac.model_coefficients_.allFinite()) {
    candidates.push_back(ransac.model_coefficients_);
  }
  py::array_t<double> candidate_array(py::array::ShapeContainer{
      static_cast<py::ssize_t>(candidates.size()), py::ssize_t(3),
      py::ssize_t(4)});
  auto candidate_view = candidate_array.mutable_unchecked<3>();
  for (py::ssize_t i = 0;
       i < static_cast<py::ssize_t>(candidates.size()); ++i) {
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 4; ++col) {
        candidate_view(i, row, col) = candidates[i](row, col);
      }
    }
  }

  py::array_t<int> inlier_array(py::array::ShapeContainer{
      static_cast<py::ssize_t>(ransac.inliers_.size())});
  auto inlier_view = inlier_array.mutable_unchecked<1>();
  for (py::ssize_t i = 0;
       i < static_cast<py::ssize_t>(ransac.inliers_.size()); ++i) {
    inlier_view(i) = ransac.inliers_[i];
  }
  return py::make_tuple(!candidates.empty(), candidate_array, inlier_array);
}

}  // namespace

PYBIND11_MODULE(_opengv_native, module) {
  module.doc() =
      "Minimal OpenGV RANSAC binding; candidate transforms are T_W_B.";
  module.def("solve_ransac_upnp", &solve_ransac_upnp,
             py::arg("bearings_C"), py::arg("points_W"),
             py::arg("camera_offsets_B"), py::arg("camera_rotations_B_C"),
             py::arg("use_generalized_ransac"), py::arg("ransac_threshold"),
             py::arg("ransac_max_iterations"), py::arg("ransac_probability"));
}
