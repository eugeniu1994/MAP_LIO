#pragma once

#include <stdint.h>
#include <vector>

#include "point_definitions.hpp"
#include "utils.h"

namespace voxel_grid {
typedef int64_t i64;

enum error_codes {
  EC_SUCCESS,
  EC_INVALID_INPUT,
  EC_VOXEL_OVERFLOW,
  EC_ID_OVERFLOW
};

union point {
  struct {
    float x, y, z, i;
  };
  float data[4];
};

struct subgrid {
  std::vector<union point> points;
  std::vector<int> counts;
  std::vector<unsigned int> ids;
  float zmin;
  unsigned int id;
};

struct grid {
  std::shared_ptr<struct subgrid> subgrids[9];
  double subgrid_width, voxel_width;
  i64 n_voxels_per_axis, n_voxels_per_subgrid;
  float xmin, ymin;
};

/// @param crop_range Must be greater than zero. Large ranges case many cells.
/// @param voxel_width Must be greater than zero. Small widths cause many
/// voxels.
enum error_codes grid_init(double crop_range, double voxel_width,
                           struct grid *p_grid);

void grid_update(struct grid &grid, const union point center,
                 const PointCloudXYZI &pc);

/// @brief Find the `k_nn` nearest neighbors within distance `max_radius`.
/// @param neighbors Size set to the number of found neighbors: in `[0, k_nn]`.
void grid_search_knn(const struct grid &grid, const PointType &ptp,
                     const int k_nn, const double max_radius,
                     PointVector &neighbors);

void grid_to_pc(const struct grid grid, PointCloudXYZI &pc);
} // namespace voxel_grid
