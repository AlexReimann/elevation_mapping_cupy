import cupy as cp
import math
import string
from typing import List

from .plugin_manager import PluginBase


class DistanceFilter(PluginBase):
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

        self.distances = cp.zeros((self.width, self.height))
        self.distance_kernel = cp.ElementwiseKernel(
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
                    if (map[idx] < step_threshold)
                    {
                      continue;
                    }

                    const float distance = sqrt((float)(dy*dy) + (float)(dx*dx)) * ${resolution};
                    if (distance > max_distance)
                    {
                      continue;
                    }

                    if (isnan(center_value) || center_value > distance)
                    {
                      center_value = distance;
                    }
                  }
                }

                if (center_value < min_distance)
                {
                  center_value = 1.0F / 0.0F; // Results in inf. Including math.h doesn't work
                }
                else if (isnan(center_value))
                {
                  center_value = 0.0F;
                }
                else
                {
                  const float distance_range = max_distance - min_distance;
                  const float normalized_distance = (center_value - min_distance) / distance_range;

                  const float midpoint = 0.4F;
                  const float steepness = 10.0F;
                  center_value = 1.0F - (1.0F / ( 1.0F + exp(-steepness*(normalized_distance - midpoint)) ));
                }
                """
            ).substitute(resolution=self.resolution),
            name="distance_kernel",
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
            return self.distances.copy()

        self.distances = cp.full((self.width, self.height), float('nan'))

        cell_radius = math.ceil(self.params["radius"] / self.resolution)
        self.distance_kernel(
            input_layer,
            cp.int32(cell_radius),
            cp.float32(self.params["radius"]),
            cp.float32(self.params["step_threshold"]),
            cp.float32(self.params["min_distance"]),
            self.distances,
            size=(self.width * self.height),
        )

        return self.distances.copy()
