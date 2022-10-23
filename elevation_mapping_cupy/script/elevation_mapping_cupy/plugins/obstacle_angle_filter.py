import cupy as cp
import math
import string
from typing import List

from .plugin_manager import PluginBase


class ObstacleAnglefilter(PluginBase):
    def __init__(self, cell_n: int = 100, radius: float = 0.5, resolution: float = 0.05, step_threshold: float = 0.0,
                 min_distance: float = 0.34, input_layer_name: str = "elevation", **kwargs):
        super().__init__()

        self.params["radius"] = radius
        self.resolution = resolution
        self.params["step_threshold"] = step_threshold
        self.params["min_distance"] = min_distance

        self.width = cell_n
        self.height = cell_n
        self.input_layer_name = input_layer_name

        self.angles = cp.zeros((self.width, self.height))
        self.obstacle_angle_kernel = cp.ElementwiseKernel(
            in_params="raw U map, int32 radius, float32 max_distance, float32 step_threshold, float32 min_distance",
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
                    if (map[idx] < step_threshold)
                    {
                      continue;
                    }

                    const float distance = sqrt((float)(dy*dy) + (float)(dx*dx)) * ${resolution};

                    if (distance < min_distance)
                    {
                      resultmap[get_map_idx(i, 0)] = 1.0F / 0.0F; // Results in inf. Including math.h doesn't work
                      return;
                    }

                    if (distance > min_valid_distance)
                    {
                      continue;
                    }

                    min_valid_distance = distance;
                    // grid axis and coordinates are flipped
                    resultmap[get_map_idx(i, 0)] = atan2(-float(dx), -float(dy));
                  }
                }

                """
            ).substitute(resolution=self.resolution),
            name="obstacle_angle_kernel",
        )

    def __call__(self, map: cp.ndarray, layer_names: List[str],
                 plugin_layers: cp.ndarray, plugin_layer_names: List[str]) -> cp.ndarray:

        if self.input_layer_name in layer_names:
            input_layer_idx = layer_names.index(self.input_layer_name)
            input_layer = map[input_layer_idx]
        elif self.input_layer_name in plugin_layer_names:
            input_layer_idx = plugin_layer_names.index(self.input_layer_name)
            input_layer = plugin_layers[input_layer_idx]
        else:
            print("layer name {} was not found in neither layers nor plugin layers. Returning zerod layer".format(
                self.input_layer_name))
            return self.angles.copy()

        self.angles = cp.full((self.width, self.height), float('nan'))

        cell_radius = math.ceil(self.params["radius"] / self.resolution)
        self.obstacle_angle_kernel(
            input_layer,
            cp.int32(cell_radius),
            cp.float32(self.params["radius"]),
            cp.float32(self.params["step_threshold"]),
            cp.float32(self.params["min_distance"]),
            self.angles,
            size=(self.width * self.height),
        )

        return self.angles.copy()
