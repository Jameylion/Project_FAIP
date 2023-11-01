from typing import Dict, Text

import numpy as np
from gymnasium.envs.registration import register
from gymnasium import spaces

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class MergeInEnv(AbstractEnv):

    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "action": {
                "type": "DiscreteAction"
            },
            "collision_reward": -1,
            # "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "merging_speed_reward": -0.5,
            "lane_change_reward": -0.05,
        })
        return cfg
    

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(self.config.get(name, 0) * reward for name, reward in self._rewards(action).items())
        return utils.lmap(reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])

    def _rewards(self, action: int) -> Dict[Text, float]:
        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward":self._right_lane_reward(),
            "high_speed_reward": self.vehicle.speed / 40,
            "lane_change_reward": action in [0, 2],
            "merging_reward": self._calculate_merging_reward(),
        }
    
    def _right_lane_reward(self):
        if self.vehicle.position[0] <= 300 and self.vehicle.lane_index[2] == 0:
            return 0.1
        elif self.vehicle.position[0] > 300 and self.vehicle.lane_index[2] == 2:
            return 0.1
        else:
            return -0.05
            


    def _calculate_merging_reward(self) -> float:
        merging_lane = ("a", "b", 2)  # Define the lane where the car merges into
        merging_vehicle_speeds = [
            vehicle.speed for vehicle in self.road.vehicles if vehicle.lane_index == merging_lane
        ]

        # Calculate the merging reward based on the speed of merging vehicles
        if merging_vehicle_speeds:
            average_merging_speed = sum(merging_vehicle_speeds) / len(merging_vehicle_speeds)
            merging_reward = 1 - (self.vehicle.speed / average_merging_speed)
            # Clamp the reward to a reasonable range
            merging_reward = np.clip(merging_reward, -1, 1)
        elif self.vehicle.position[0] > 310:
            # No merging vehicles, assign a positive reward for staying in the merging lane
            merging_reward = 0.1

        return merging_reward
    
    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 400)

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 60,20 , 150]  # Before, converging, merge,conv merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH,2*StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s],[n, s], [n, c]]
        line_type_merge = [[c, s],[n, s], [n, n]]
        
        for i in range(3):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:4]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:4]), y[i]], [sum(ends), y[i]], line_types=line_type[i],forbidden=False))

        # Merging lane
        amplitude = 3.25
        amplitude2 = 6.25
        ljk = StraightLane([0, 6.5 + 4 + 4+ 4], [ends[0], 6.5 + 4 + 4+ 4], line_types=[c, c], forbidden=False)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=False)
        # lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
        #                    line_types=[n, c],forbidden=False)
        lbm = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1]+ends[2], -amplitude2), line_types=[s, c],forbidden=False)
        lmc = SineLane(lbm.position(ends[2], -amplitude2), lbm.position(ends[2]+ends[3], -amplitude2),
                        amplitude2,  2*np.pi / (2 * ends[2]), np.pi / 2, line_types=[n, c], forbidden=False)
        
        # print(f'lbm begin =  {lkb.position(ends[1], 0)}')
        # print(f'lbm eind =  {lkb.position(ends[1]+ends[2], 0)}')
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "m", lbm)
        net.add_lane("m", "c", lmc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        # road.objects.append(Obstacle(road, lmc.position(ends[2] +30, 0)))
        # print(net.get_lane(("j","k",0)))
        self.road = road

    def _make_vehicles(self, speed_factor: int =1) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle =  self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20)
        
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), speed=29))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(90, 0), speed=31.5))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 2)).position(110, 0), speed=20))

        # merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20)
       
        # merging_v.target_speed = 30
        # road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle
        # self.merger = self.vehicle 

class MergeInEnvReward2(MergeInEnv):
    def _rewards(self, action: int) -> Dict[Text, float]:
        return {
            "collision_reward": self.vehicle.crashed,
            "merging_reward": self._calculate_merging_reward(),
        }

    def _calculate_merging_reward(self) -> float:
        merging_lane = ("a", "b", 2)  # Define the merging lane
        merging_vehicle_speeds = [
            vehicle.speed for vehicle in self.road.vehicles if vehicle.lane_index == merging_lane
        ]

        # Calculate the merging reward based on the speed of merging vehicles
        if merging_vehicle_speeds:
            average_merging_speed = sum(merging_vehicle_speeds) / len(merging_vehicle_speeds)
            merging_reward = 1 - (self.vehicle.speed / average_merging_speed)
            # Clamp the reward to a reasonable range
            merging_reward = np.clip(merging_reward, -1, 1)
        else:
            # No merging vehicles, assign a positive reward for staying in the merging lane
            merging_reward = 0.1

        return merging_reward

    def _reward(self, action: int) -> float:
        rewards = self._rewards(action)
        reward = rewards["collision_reward"] + rewards["merging_reward"]

        # Additional penalties/rewards can be added here based on your task requirements

        # Scaling the reward
        reward /= 20
        
        # Crash penalty
        if rewards["collision_reward"]:
            reward = -10

        return reward

class DiscreteMergeInEnvReward2(MergeInEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                #"features": ["presence", "x", "y", "vx", "vy", "long_off", "lat_off", "ang_off"],
            },
            "action": {
                "type": "DiscreteAction",
                "longitudinal": True,
                "lateral": True,
                "actions_per_axis": (3, 5)
            },
        })
        
        return config
    
    def __init__(self, config=None):
        super().__init__(config)
        self.action_space = spaces.Discrete(15)  # Total 15 discrete actions (3 longitudinal actions × 5 lateral actions)

    def _rewards(self, action: int) -> float:
        return 0
    
    def _reward(self, action: int) -> float:
        reward = 0
        
        merging_lane = ("a", "b", 2)  # Define the lane where to merge
        merging_vehicle_speeds = [
            vehicle.speed for vehicle in self.road.vehicles if vehicle.lane_index == merging_lane
        ]

        # Calculate the merging reward based on the speed of merging vehicles
        if merging_vehicle_speeds:
            average_merging_speed = sum(merging_vehicle_speeds) / len(merging_vehicle_speeds)
            reward += 1 - (self.vehicle.speed / average_merging_speed)
            # Clamp the reward to a reasonable range
            reward += np.clip(reward, -1, 1)
            reward *= 20
        else:
            # No merging vehicles, assign a positive reward for staying in the merging lane
            reward += 2

     
        #punishment for driving on the left lane
        # lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
        #     else self.vehicle.lane_index[2]
        # reward -= lane
        #
        #take the difference in radians of the heading of the car and the heading of the road
        anglediff = min(abs(self.vehicle.heading-self.vehicle.lane.lane_heading(self.vehicle.position)),abs(self.vehicle.lane.lane_heading(self.vehicle.position))+abs(self.vehicle.heading))

        #effective speed
        if self.vehicle.speed > 0:
            reward += np.cos(anglediff)*self.vehicle.speed/max(1,abs(self.vehicle.lane_distance))
            
        #punishment for distance to the lane
        # print(f'self. lane = {self.vehicle.lane_index[2]}')
        #reward += 30/abs(self.vehicle.lane_distance)
        if self.vehicle.lane_index[2] == 0 and self.vehicle.position[0] < 150:
            reward *= self.vehicle.lane_index[2]
        else:
            reward *= 0.5
        # if self.vehicle.crashed:
        #     return -10
        #print(self.vehicle.lane_index)
        #scaling
        reward = reward/20
        
        #crash punishment
        if self.vehicle.crashed:
            return -10
        
        
        
        return reward

class DiscreteMergeInEnvReward1(MergeInEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                #"features": ["presence", "x", "y", "vx", "vy", "long_off", "lat_off", "ang_off"],
            },
            "action": {
                "type": "DiscreteAction",
                "longitudinal": True,
                "lateral": True,
                "actions_per_axis": (3, 5)
            },
        })
        
        return config
    
    def __init__(self, config=None):
        super().__init__(config)
        self.action_space = spaces.Discrete(15)  # Total 15 discrete actions (3 longitudinal actions × 5 lateral actions)

    def _rewards(self, action: int) -> float:
        return 0
    
    def _reward(self, action: int) -> float:
        reward = 0
        
        merging_lane = ("a", "b", 2)  # Define the lane where to merge
        merging_vehicle_speeds = [
            vehicle.speed for vehicle in self.road.vehicles if vehicle.lane_index == merging_lane
        ]

        # Calculate the merging reward based on the speed of merging vehicles
        if merging_vehicle_speeds:
            average_merging_speed = sum(merging_vehicle_speeds) / len(merging_vehicle_speeds)
            reward += 1 - (self.vehicle.speed / average_merging_speed)
            # Clamp the reward to a reasonable range
            reward += np.clip(reward, -1, 1)
            reward *= 20


        if self.vehicle.position[0] <= 300 and self.vehicle.lane_index[2] == 0:
            reward *= 2
        elif self.vehicle.position[0] > 300 and self.vehicle.lane_index[2] == 2:
            reward *= 2
        else:
            reward /= 2
        #punishment for driving on the left lane
        # lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
        #     else self.vehicle.lane_index[2]
        # reward -= lane
        #
        #take the difference in radians of the heading of the car and the heading of the road
        anglediff = min(abs(self.vehicle.heading-self.vehicle.lane.lane_heading(self.vehicle.position)),abs(self.vehicle.lane.lane_heading(self.vehicle.position))+abs(self.vehicle.heading))

        #effective speed
        if self.vehicle.speed > 0:
            reward += np.cos(anglediff)*self.vehicle.speed/max(1,abs(self.vehicle.lane_distance))
            
        #punishment for distance to the lane
        # print(f'self. lane = {self.vehicle.lane_index[2]}')
        #reward += 30/abs(self.vehicle.lane_distance)
        if self.vehicle.lane_index[2] == 0 and self.vehicle.position[0] < 150:
            reward *= self.vehicle.lane_index[2]
        else:
            reward *= 0.5
        # if self.vehicle.crashed:
        #     return -10
        #print(self.vehicle.lane_index)
        #scaling
        reward = reward/20
        
        #crash punishment
        if self.vehicle.crashed:
            return -10
        
        
        
        return reward
# register(
#     id='merge_in-v0',
#     entry_point='highway_env.envs:MergeInEnv',
# )

# register(
#     id='merge_in-v1',
#     entry_point='highway_env.envs:MergeInEnvReward2',
# )

# register(
#     id='merge_in-v2',
#     entry_point='highway_env.envs:DiscreteMergeInEnvReward2',
# )

# register(
#     id='merge_in-v3',
#     entry_point='highway_env.envs:DiscreteMergeInEnvReward1',
# )
