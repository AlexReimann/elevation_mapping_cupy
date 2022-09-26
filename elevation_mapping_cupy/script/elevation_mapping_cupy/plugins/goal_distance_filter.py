import cupy as cp
import math
import string
from typing import List

from .plugin_manager import PluginBase


class GoalDistanceFilter(PluginBase):
    def __init__(self, cell_n: int = 100, resolution: float = 0.05, **kwargs):
        super().__init__()

        self.uses_goal = True
        self.resolution = resolution

        self.width = cell_n
        self.height = cell_n

        self.distances = cp.zeros((self.width, self.height))

        self.goal_distance_kernel = cp.ElementwiseKernel(
            in_params="float32 goal_x, float32 goal_y",
            out_params="raw U resultmap",
            operation=string.Template(
                """
                const float x = (0.5 * ${width} - (i % ${width})) * ${resolution};
                const float y = (0.5 * ${height} - (i / ${width})) * ${resolution};

                const float dx = goal_x - x;
                const float dy = goal_y - y;

                resultmap[i] = sqrt((dx*dx) + (dy*dy));
                """
            ).substitute(width=self.width, height=self.height, resolution=self.resolution),
            name="goal_distance_kernel",
        )

    def __call__(self, map: cp.ndarray, layer_names: List[str],
                 plugin_layers: cp.ndarray, plugin_layer_names: List[str]) -> cp.ndarray:

        if self.goal is None:
            return self.distances.copy()

        self.distances = cp.full((self.width, self.height), float('nan'))

        self.goal_distance_kernel(
            cp.float32(self.goal[0]),
            cp.float32(self.goal[1]),
            self.distances,
            size=(self.width * self.height),
        )

        return self.distances.copy()
