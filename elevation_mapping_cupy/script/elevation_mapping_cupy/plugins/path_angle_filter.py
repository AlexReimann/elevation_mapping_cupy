import cupy as cp
import math
import string
from typing import List

from .plugin_manager import PluginBase


class PathAngleFilter(PluginBase):
    def __init__(self, cell_n: int = 100, radius: float = 2.0, distance_cost_scaling: float = 1.0, resolution: float = 0.05, midpoint: float = 0.5, steepness: float = 10.0, **kwargs):
        super().__init__()

        self.uses_path = True

        self.params["radius"] = radius
        self.params["distance_cost_scaling"] = distance_cost_scaling
        self.resolution = resolution
        self.params["midpoint"] = midpoint
        self.params["steepness"] = steepness

        self.width = cell_n
        self.height = cell_n

        self.angles = cp.zeros((self.width, self.height))
        self.path_map = cp.zeros((self.width, self.height))

        self.path_angle_kernel = cp.ElementwiseKernel(
            in_params="raw U path_map, int32 radius, float32 max_distance, float32 distance_cost_scaling, float32 midpoint, float32 steepness",
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

                float min_valid_distance = 1.0F / 0.0F; // Results in inf. Including math.h doesn't work

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
                    if (distance >= max_distance)
                    {
                      continue;
                    }

                    if (isinf(map_path_value))
                    {
                      center_value = atan2((float)dx, (float)dy);
                      continue;
                    }

                    if (distance >= min_valid_distance)
                    {
                      continue;
                    }

                    min_valid_distance = distance;


                    const float distance_normalized = distance / max_distance;

                    const float scaling = 1.0F / ( 1.0F + exp(-steepness*(distance_normalized - midpoint)) );

                    const float scaled_dx = dx * distance_cost_scaling * scaling;
                    const float scaled_dy = dy * distance_cost_scaling * scaling;

                    const float path_x = cos(map_path_value);
                    const float path_y = sin(map_path_value);
                    const float average_x = 0.5F * (path_x + scaled_dy);
                    const float average_y = 0.5F * (path_y + scaled_dx);

                    center_value = atan2(average_y, average_x);
                  }
                }

                if (isnan(center_value))
                {
                  center_value = 1.0F / 0.0F; // Results in inf. Including math.h doesn't work
                }
                """
            ).substitute(resolution=self.resolution),
            name="path_angle_kernel",
        )

    def __call__(self, map: cp.ndarray, layer_names: List[str],
                 plugin_layers: cp.ndarray, plugin_layer_names: List[str]) -> cp.ndarray:

        if self.path is None or len(self.path) == 0:
            return self.angles.copy()


        self.angles = cp.full((self.width, self.height), float('nan'))
        self.path_map = cp.full((self.width, self.height), float('nan'))

        if len(self.path) == 1:
            self.path_map[self.path[0][0], self.path[0][1]] = cp.inf

        else:
            last_point = self.path[0]
            for path_point_index in self.path[1:-1]:
                dx = (path_point_index[0] - last_point[0])
                dy = (path_point_index[1] - last_point[1])

                if dx == 0 and dy == 0:
                    continue

                self.path_map[path_point_index[0],
                              path_point_index[1]] = math.atan2(float(dy), float(dx))
                last_point = path_point_index


        cell_radius = math.ceil(self.params["radius"] / self.resolution)
        self.path_angle_kernel(
            self.path_map,
            cp.int32(cell_radius),
            cp.float32(self.params["radius"]),
            cp.float32(self.params["distance_cost_scaling"]),
            cp.float32(self.params["midpoint"]),
            cp.float32(self.params["steepness"]),
            self.angles,
            size=(self.width * self.height),
        )

        return self.angles.copy()
