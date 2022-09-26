import cupy as cp
import math
import string
from typing import List

from .plugin_manager import PluginBase


class GoalDistanceFilter(PluginBase):
    def __init__(self, cell_n: int = 100, **kwargs):
        super().__init__()

        # Set by dynamic reconfigure
        self.params["obstacle_cost_scaling"] = 1.0
        self.params["path_distance_cost_scaling"] = 1.0
        self.params["goal_distance_cost_scaling"] = 1.0

        self.costs = cp.zeros((cell_n, cell_n))

    def __call__(self, map: cp.ndarray, layer_names: List[str],
                 plugin_layers: cp.ndarray, plugin_layer_names: List[str]) -> cp.ndarray:

        if "distance_filter_layer" not in plugin_layer_names \
          or "path_distance_filter_layer" not in plugin_layer_names \
          or "goal_distance_filter_layer" not in plugin_layer_names:
            return self.costs

        get_layer = lambda layer_name: plugin_layers[plugin_layer_names.index(layer_name)]

        distance_layer = get_layer("distance_filter_layer")
        path_distance_layer = get_layer("path_distance_filter_layer")
        goal_distance_layer = get_layer("goal_distance_filter_layer")

        self.costs = (distance_layer * self.params["obstacle_cost_scaling"]) \
                   + (path_distance_layer * self.params["path_distance_cost_scaling"]) \
                   + (goal_distance_layer * self.params["goal_distance_cost_scaling"])

        return self.costs.copy()
