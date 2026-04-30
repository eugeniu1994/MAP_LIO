#include "grid.hpp"

#include <math.h>
#include <vector>

#include "point_definitions.hpp"
#include "utils.h"

namespace voxel_grid {
union vidx {
  struct {
    i64 x, y, z;
  };
  i64 data[3];
};

struct grid_index {
  union vidx vidx;
  int ci;
};

struct dist_ind {
  double dist_squared;
  i64 vi;
  int gi;
};

static struct grid_index point_to_index(const union point p,
                                        const struct grid &grid) {
  if (p.x < grid.xmin || p.x >= grid.xmin + grid.subgrid_width * 2.999999) {
    return { {}, -1 };
  }
  if (p.y < grid.ymin || p.y >= grid.ymin + grid.subgrid_width * 2.999999) {
    return { {}, -1 };
  }
  i64 vx = (i64)((p.x - grid.xmin) / grid.voxel_width);
  i64 vy = (i64)((p.y - grid.ymin) / grid.voxel_width);

  const i64 cx = vx / grid.n_voxels_per_axis;
  const i64 cy = vy / grid.n_voxels_per_axis;
  assert(0 <= cx && cx < 3);
  assert(0 <= cy && cy < 3);
  const int ci = (int)(cx + cy * 3);

  const float zdiff = p.z - grid.subgrids[ci]->zmin;
  if (zdiff < 0.0 ||
      zdiff >= grid.voxel_width * grid.n_voxels_per_axis * 0.999) {
    return { {}, -1 };
  }

  vx -= cx * grid.n_voxels_per_axis;
  vy -= cy * grid.n_voxels_per_axis;
  const i64 vz = (i64)(zdiff / grid.voxel_width);
  assert(0 <= vx && vx < grid.n_voxels_per_axis);
  assert(0 <= vy && vy < grid.n_voxels_per_axis);
  assert(0 <= vz && vz < grid.n_voxels_per_axis);
  const union vidx vidx = { vx, vy, vz };
  return { vidx, ci };
}

static i64 vidx_to_vi(const union vidx vidx, const i64 n_voxels) {
  return vidx.x + vidx.y * n_voxels + vidx.z * n_voxels * n_voxels;
}

enum error_codes grid_init(double crop_range, double voxel_width,
                           struct grid *p_grid) {
  if (voxel_width <= 0.0 || crop_range < voxel_width) {
    return EC_INVALID_INPUT;
  }
  crop_range = voxel_width * std::ceil(crop_range / voxel_width);
  if (crop_range / voxel_width * 3.0 > (double)(INT64_MAX - 1)) {
    return EC_VOXEL_OVERFLOW;
  }

  const i64 n_voxels_per_axis = (i64)(crop_range / voxel_width);
  if (n_voxels_per_axis >
      INT64_MAX / n_voxels_per_axis / n_voxels_per_axis / 9) {
    return EC_VOXEL_OVERFLOW;
  }

  i64 n_voxels_per_subgrid =
      n_voxels_per_axis * n_voxels_per_axis * n_voxels_per_axis;

  struct grid grid = {};
  grid.subgrid_width = crop_range;
  grid.voxel_width = voxel_width;
  grid.n_voxels_per_axis = n_voxels_per_axis;
  grid.n_voxels_per_subgrid = n_voxels_per_subgrid;
  grid.xmin = -INFINITY;
  grid.ymin = -INFINITY;
  for (int i = 0; i < 9; ++i) {
    std::shared_ptr<struct subgrid> subgrid(new struct subgrid());
    subgrid->points.resize(n_voxels_per_subgrid);
    subgrid->counts.resize(n_voxels_per_subgrid);
    subgrid->ids.resize(n_voxels_per_subgrid);
    subgrid->zmin = -INFINITY;
    subgrid->id = 1;
    grid.subgrids[i] = subgrid;
  }
  *p_grid = grid;
  return EC_SUCCESS;
}

static void shift_pointers(std::shared_ptr<struct subgrid> *ptrs[3],
                           const float zmin, const int shift) {
  std::shared_ptr<struct subgrid> temp[3];
  for (int k = 0; k < 3; ++k) {
    temp[k] = *ptrs[k];
  }

  for (int k = 0; k < 3; ++k) {
    int t = k + shift;
    if (t < 0 || t > 2) {
      temp[k]->zmin = zmin;
      temp[k]->id += 1;
      t = ((t % 3) + 3) % 3;
    }
    assert(0 <= t && t < 3);
    *ptrs[t] = temp[k];
  }
}

static void shift_subgrids(struct grid &grid, const union point center) {
  const float dx = center.x - grid.xmin;
  const float dy = center.y - grid.ymin;
  const float zmin = center.z - grid.subgrid_width / 3.0;
  /* Every subgrid is invalid */
  if (std::max(std::abs(dx), std::abs(dy)) > grid.subgrid_width * 4.0) {
    for (int i = 0; i < 9; ++i) {
      grid.subgrids[i]->zmin = zmin;
      grid.subgrids[i]->id += 1;
    }
    grid.xmin =
        grid.subgrid_width * (std::floor(center.x / grid.subgrid_width) - 1);
    grid.ymin =
        grid.subgrid_width * (std::floor(center.y / grid.subgrid_width) - 1);
    return;
  }

  const i64 vx_shift =
      1 - (i64)std::floor((center.x - grid.xmin) / grid.subgrid_width);
  const i64 vy_shift =
      1 - (i64)std::floor((center.y - grid.ymin) / grid.subgrid_width);
  if (vx_shift == 0 && vy_shift == 0) {
    return;
  }

  /* Every subgrid is invalid */
  if (std::abs(vx_shift) > 2 || std::abs(vy_shift) > 2) {
    for (int i = 0; i < 9; ++i) {
      grid.subgrids[i]->zmin = zmin;
      grid.subgrids[i]->id += 1;
    }
  } else {
    std::shared_ptr<struct subgrid> *arr[3];
    for (int y = 0; y < 3; ++y) {
      for (int x = 0; x < 3; ++x) {
        arr[x] = &grid.subgrids[x + y * 3];
      }
      shift_pointers(arr, zmin, (int)vx_shift);
    }

    for (int x = 0; x < 3; ++x) {
      for (int y = 0; y < 3; ++y) {
        arr[y] = &grid.subgrids[x + y * 3];
      }
      shift_pointers(arr, zmin, (int)vy_shift);
    }
  }
  grid.xmin -= vx_shift * grid.subgrid_width;
  grid.ymin -= vy_shift * grid.subgrid_width;
}

void grid_update(struct grid &grid, const union point center,
                 const PointCloudXYZI &pc) {
  if (pc.size() == 0) {
    return;
  }

  shift_subgrids(grid, center);

  const int n_pts = (int)pc.size();
  for (int i = 0; i < n_pts; ++i) {
    const PointType &ptp = pc.points[i];
    const union point p = { ptp.x, ptp.y, ptp.z, ptp.intensity };
    struct grid_index index = point_to_index(p, grid);
    if (index.ci == -1) {
      continue;
    }

    struct subgrid &subgrid = *grid.subgrids[index.ci];
    const i64 vi = vidx_to_vi(index.vidx, grid.n_voxels_per_axis);
    union point &gp = subgrid.points[vi];
    int &count = subgrid.counts[vi];
    unsigned int &id = subgrid.ids[vi];
    if (id != subgrid.id) {
      gp = p;
      count = 1;
      id = subgrid.id;
    } else {
      const float inv = 1.0f / (count + 1);
      for (int k = 0; k < 4; ++k) {
        /* average = (old * count + new) / (count + 1)
         *         = old + (new - old) / (count + 1)
         */
        gp.data[k] += (p.data[k] - gp.data[k]) * inv;
      }
      ++count;
    }
  }
}

static void insert_keep_sorted(struct dist_ind *arr, const int n,
                               struct dist_ind elem) {
  if (elem.dist_squared >= arr[n - 1].dist_squared) {
    return;
  }

  for (int i = n - 2; i >= 0; --i) {
    if (elem.dist_squared >= arr[i].dist_squared) {
      arr[i + 1] = elem;
      return;
    }

    arr[i + 1] = arr[i];
  }

  arr[0] = elem;
}

void grid_search_knn(const struct grid &grid, const PointType &ptp,
                     const int k_nn, const double max_radius,
                     PointVector &neighbors) {
  assert(k_nn > 0);

  neighbors.clear();
  neighbors.reserve(k_nn);

  const double radius_squared = max_radius * max_radius;
  std::vector<struct dist_ind> di_neighbors(k_nn);
  for (int i = 0; i < k_nn; ++i) {
    di_neighbors[i].dist_squared = INFINITY;
  }

  const union point p = { ptp.x, ptp.y, ptp.z, 0 };
  for (int gx = 0; gx < 3; ++gx) {
    for (int gy = 0; gy < 3; ++gy) {
      const int gi = gx + gy * 3;
      const struct subgrid &subgrid = *grid.subgrids[gi];
      struct grid_index index = {};
      index.vidx.x =
          (i64)((p.x - grid.xmin - gx * grid.subgrid_width) / grid.voxel_width);
      index.vidx.y =
          (i64)((p.y - grid.ymin - gy * grid.subgrid_width) / grid.voxel_width);
      index.vidx.z = (i64)((p.z - subgrid.zmin) / grid.voxel_width);

      union vidx vmin, vmax;
      const int n_dvidx = (int)std::ceil(max_radius / grid.voxel_width);
      {
        for (int k = 0; k < 3; ++k) {
          vmin.data[k] = std::max((i64)0, index.vidx.data[k] - n_dvidx);
          vmax.data[k] = std::min((i64)grid.n_voxels_per_axis,
                                  index.vidx.data[k] + n_dvidx + 1);
        }
      }

      for (i64 vz = vmin.z; vz < vmax.z; ++vz) {
        for (i64 vy = vmin.y; vy < vmax.y; ++vy) {
          for (i64 vx = vmin.x; vx < vmax.x; ++vx) {
            const union vidx vidx = { vx, vy, vz };
            const i64 vi = vidx_to_vi(vidx, grid.n_voxels_per_axis);
            if (subgrid.ids[vi] != subgrid.id) {
              continue;
            }
            assert(subgrid.counts[vi] > 0);

            const union point q = subgrid.points[vi];
            const double dx = q.x - p.x;
            const double dy = q.y - p.y;
            const double dz = q.z - p.z;
            const double dist_squared = dx * dx + dy * dy + dz * dz;
            assert(dist_squared <= 12 * radius_squared);
            assert(dist_squared <=
                   3 * grid.voxel_width * (n_dvidx + 1) * (n_dvidx + 1));

            if (dist_squared > radius_squared) {
              continue;
            }

            const struct dist_ind elem = { dist_squared, vi, gi };
            insert_keep_sorted(di_neighbors.data(), k_nn, elem);
          }
        }
      }
    }
  }

  for (int i = 0; i < k_nn; ++i) {
    if (di_neighbors[i].dist_squared == INFINITY) {
      break;
    }

    const i64 vi = di_neighbors[i].vi;
    const int gi = di_neighbors[i].gi;
    const struct subgrid &subgrid = *grid.subgrids[gi];
    assert(subgrid.ids[vi] == subgrid.id && subgrid.counts[vi] > 0);

    const union point p = subgrid.points[vi];
    PointType ptp = {};
    ptp.x = p.x;
    ptp.y = p.y;
    ptp.z = p.z;
    ptp.intensity = p.i;
    neighbors.push_back(ptp);
  }
}

void grid_to_pc(const struct grid grid, PointCloudXYZI &pc) {
  pc.clear();
  pc.reserve(grid.n_voxels_per_subgrid); /* Heuristic estimate */
  for (int ci = 0; ci < 9; ++ci) {
    const struct subgrid &subgrid = *grid.subgrids[ci];
    for (i64 vi = 0; vi < grid.n_voxels_per_subgrid; ++vi) {
      if (subgrid.ids[vi] != subgrid.id) {
        continue;
      }
      assert(subgrid.counts[vi] > 0);

      const union point p = subgrid.points[vi];
      PointType ptp = {};
      ptp.x = p.x;
      ptp.y = p.y;
      ptp.z = p.z;
      ptp.intensity = p.i;
      pc.push_back(ptp);
    }
  }
}
} // namespace voxel_grid
