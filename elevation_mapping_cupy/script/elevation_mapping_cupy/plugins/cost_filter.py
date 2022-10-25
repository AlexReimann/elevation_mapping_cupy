import cupy as cp
import math
import string
from typing import List

from .plugin_manager import PluginBase


class GoalDistanceFilter(PluginBase):
    def __init__(self, cell_n: int = 100, obstacle_cost_scaling: float = 1.0,
      path_distance_cost_scaling: float = 1.0, goal_distance_cost_scaling: float = 1.0, **kwargs):
        super().__init__()

        # Set by dynamic reconfigure
        self.params["obstacle_cost_scaling"] = float(obstacle_cost_scaling)
        self.params["path_distance_cost_scaling"] = float(path_distance_cost_scaling)
        self.params["goal_distance_cost_scaling"] = float(goal_distance_cost_scaling)

        self.costs = cp.zeros((cell_n, cell_n))

    def __call__(self, map: cp.ndarray, layer_names: List[str],
                 plugin_layers: cp.ndarray, plugin_layer_names: List[str]) -> cp.ndarray:

        if "distance_filter_layer" not in plugin_layer_names \
          or "path_distance_filter_layer" not in plugin_layer_names \
          or "goal_distance_filter_layer" not in plugin_layer_names:
            return self.costs

        get_layer = lambda layer_name: plugin_layers[plugin_layer_names.index(layer_name)]

        distance_layer = get_layer("distance_filter_layer")
        obstacle_angle_layer = get_layer("obstacle_angle_layer")
        path_angle_layer = get_layer("path_angle_filter_layer")

        obstacle_angle_nans = cp.isnan(obstacle_angle_layer)
        mask = obstacle_angle_nans * path_angle_layer
        masked_obstacle_angles = obstacle_angle_layer
        masked_obstacle_angles[obstacle_angle_nans] = 0.0
        masked_obstacle_angles = masked_obstacle_angles + mask

        scaled_distance_costs = distance_layer * self.params["obstacle_cost_scaling"]
        obstacle_angles_x = scaled_distance_costs * cp.cos(masked_obstacle_angles)
        obstacle_angles_y = scaled_distance_costs * cp.sin(masked_obstacle_angles)

        path_angles_x = cp.cos(path_angle_layer)
        path_angles_y = cp.sin(path_angle_layer)

        x_angles = ((scaled_distance_costs * obstacle_angles_x) + path_angles_x) / 2.0
        y_angles = ((scaled_distance_costs * obstacle_angles_y) + path_angles_y) / 2.0

        self.costs = cp.arctan2(y_angles, x_angles)

        return self.costs.copy()
