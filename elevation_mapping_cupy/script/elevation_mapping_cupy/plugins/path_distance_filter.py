import cupy as cp
import math
import string
from typing import List

from .plugin_manager import PluginBase


class PathDistanceFilter(PluginBase):
    def __init__(self, cell_n: int = 100, radius: float = 2.0, distance_cost_scaling: float = 1.0, resolution: float = 0.05, **kwargs):
        super().__init__()

        self.uses_path = True

        self.params["radius"] = radius
        self.params["distance_cost_scaling"] = distance_cost_scaling
        self.resolution = resolution

        self.width = cell_n
        self.height = cell_n

        self.distances = cp.zeros((self.width, self.height))
        self.path_map = cp.zeros((self.width, self.height))

        self.path_distance_kernel = cp.ElementwiseKernel(
            in_params="raw U path_map, int32 radius, float32 max_distance, float32 distance_cost_scaling",
            out_params="raw U resultmap",
            preamble=string.Template(
                """
                __device__ int get_map_idx(int idx, int layer_n)
                {
                    const int layer = ${width} * ${height};
                    return layer * layer_n + idx;
                }

                __device__ int get_relative_map_idx(int idx, int dx, int dy, int layer_n)
                {
                    const int layer = ${width} * ${height};
                    const int relative_idx = idx + ${width} * dy + dx;
                    return layer * layer_n + relative_idx;
                }

                __device__ bool is_inside(int idx, int dx, int dy)
                {
                    int idx_x = (idx % ${width}) + dx;
                    int idx_y = (idx / ${width}) + dy;
                    if (idx_x <= 0 || idx_x >= ${width} - 1)
                    {
                        return false;
                    }
                    if (idx_y <= 0 || idx_y >= ${height} - 1)
                    {
                        return false;
                    }
                    return true;
                }
                """
            ).substitute(width=self.width, height=self.height),
            operation=string.Template(
                """
                U& center_value = resultmap[get_map_idx(i, 0)];

                for (int dy = -radius; dy <= radius; ++dy)
                {
                  for (int dx = -radius; dx <= radius; ++dx)
                  {
                    if (!is_inside(i, dx, dy))
                    {
                      continue;
                    }

                    const int idx = get_relative_map_idx(i, dx, dy, 0);
                    const float map_path_value = path_map[idx];
                    if (isnan(map_path_value))
                    {
                      continue;
                    }

                    const float distance = sqrt((float)(dy*dy) + (float)(dx*dx)) * ${resolution};
                    if (distance > max_distance)
                    {
                      continue;
                    }
                    const float value = (distance * distance_cost_scaling) + map_path_value;

                    if (isnan(center_value) || value < center_value)
                    {
                      center_value = value;
                    }
                  }
                }

                if (isnan(center_value))
                {
                  center_value = 1.0F / 0.0F; // Results in inf. Including math.h doesn't work
                }
                """
            ).substitute(resolution=self.resolution),
            name="path_distance_kernel",
        )

    def __call__(self, map: cp.ndarray, layer_names: List[str],
                 plugin_layers: cp.ndarray, plugin_layer_names: List[str]) -> cp.ndarray:

        if self.path is None or len(self.path) == 0:
            return self.distances.copy()

        self.distances = cp.full((self.width, self.height), float('nan'))

        self.path_map = cp.full((self.width, self.height), float('nan'))

        sum_length = 0.0
        last_point = self.path[-1]
        for path_point_index in reversed(self.path):
            dx = (last_point[0] - path_point_index[0]) * self.resolution
            dy = (last_point[1] - path_point_index[1]) * self.resolution
            sum_length = sum_length + math.sqrt((dx*dx) + (dy*dy))
            self.path_map[path_point_index[0],
                          path_point_index[1]] = sum_length
            last_point = path_point_index

        cell_radius = math.ceil(self.params["radius"] / self.resolution)
        self.path_distance_kernel(
            self.path_map,
            cp.int32(cell_radius),
            cp.float32(self.params["radius"]),
            cp.float32(self.params["distance_cost_scaling"]),
            self.distances,
            size=(self.width * self.height),
        )

        return self.distances.copy()
