# manager that adds items (currently pedestrian) on the sidewalk.
# Note: currently you need to change path in the init function.
import math
import os
from collections import defaultdict
from random import sample
from asset.read_config import configReader
from metaurban.component.lane.abs_lane import AbstractLane
from metaurban.component.pgblock.curve import Curve
from metaurban.component.pgblock.first_block import FirstPGBlock
from metaurban.component.pgblock.ramp import InRampOnStraight, OutRampOnStraight
from metaurban.component.pgblock.straight import Straight
from metaurban.component.road_network import Road
from metaurban.component.static_object.test_new_object import TestObject, TestGLTFObject
from metaurban.component.static_object.traffic_object import TrafficCone, TrafficWarning, TrafficBarrier
from metaurban.engine.engine_utils import get_engine
from metaurban.manager.base_manager import BaseManager
from metaurban.constants import PGDrivableAreaProperty as DrivableAreaProperty
import numpy as np
import cv2
import json
import random
class GridCell:
    """
    Represents a single cell in a grid, which can be occupied by an object.
    """
    def __init__(self, position, occupied=False):
        """
        Args:
            position (tuple): The position of the cell in the grid, as a tuple (i, j).
            occupied (bool): Whether the cell is occupied by an object.
        """
        self.position = position
        self.occupied = occupied
        self.object = None  # Optional, to reference the object occupying the cell

    def is_occupied(self):
        """
        Check if the cell is occupied by an object.
        returns:
            bool: True if the cell is occupied, False otherwise.
        """
        return self.occupied

    def occupy(self, obj):
        """
        Mark the cell as occupied by an object.
        returns:
            bool: True if the cell is occupied, False otherwise.
        """
        self.occupied = True
        self.object = obj

    def release(self):
        """
        Mark the cell as unoccupied.
        """
        self.occupied = False
        self.object = None
class ObjectPlacer:
    """
    This class is used to place objects on a grid, ensuring that they do not overlap.
    """
    def __init__(self, grid):
        """
        Args:
            grid (list): A 2D list of GridCell objects representing the placement grid.
            
            Example:
            grid = [
                [GridCell(), GridCell(), GridCell()],
                [GridCell(), GridCell(), GridCell()],
                [GridCell(), GridCell(), GridCell()]
            ]
        """
        self.grid = grid
        # Dictionary to store the objects that have been placed, with their positions
        # Keys are tuples (object_id, position), values are tuples (position, object)
        self.placed_objects = {}

    def place_object(self, obj, last_long=None):
        """
        Attempts to place a single object on the grid.

        Args:
            obj (dict): The object to be placed, with properties like length and width.

        Returns:
            bool: True if the object was successfully placed, False otherwise.
        """
        # Find a position where the object can be placed
        if last_long is not None:
            assert 'spawn_long_gap' in obj
            position = self.find_placement_position(obj, last_long)
            if position is not None:
                # Mark the cells as occupied
                self.mark_occupied_cells(position, obj)
                obj_id = (obj['CLASS_NAME'], position)  # each object has a unique 'id'
                self.placed_objects[obj_id] = (position, obj)
                return True, position[0]
            return False, last_long
        if last_long is None:
            position = self.find_placement_position(obj)
            # Mark the cells as occupied
            if position is not None:
                self.mark_occupied_cells(position, obj)
                obj_id = (obj['CLASS_NAME'], position)  # each object has a unique 'id'
                self.placed_objects[obj_id] = (position, obj)
                return True
            return False

    def find_placement_position(self, obj, last_long=None):
        """
        Find a position on the grid where the object can be placed.
        For now, we just return the top-left position where the object can be placed.
        Args:
            obj (dict): The object to be placed, with properties like length and width.
        Returns:
            tuple: The top-left position where the object can be placed, or None if no position is found.
        """
        if 'obj_generation_mode' in obj:
            if obj['obj_generation_mode'] == 'parallel_only':
                if last_long is not None:
                    assert 'spawn_long_gap' in obj
                    for i in range(last_long + obj['spawn_long_gap'], len(self.grid)):
                        for j in range(1):
                            if self.can_place(i+1, j+1, obj):
                                return (i+1, j+1)  # Top-left position where the object can be placed
                    return None
                
                for i in range(len(self.grid)):
                        for j in range(1):
                            if self.can_place(i+1, j+1, obj):
                                return (i+1, j+1)  # Top-left position where the object can be placed
                            
                return None
            
            if obj['obj_generation_mode'] == 'normal':
                if last_long is not None:
                    assert 'spawn_long_gap' in obj
                    for i in range(last_long + obj['spawn_long_gap'], len(self.grid)):
                        for j in range(len(self.grid[i])):
                            if self.can_place(i+1, j+1, obj):
                                return (i+1, j+1)  # Top-left position where the object can be placed
                    return None
                
                for i in range(len(self.grid)):
                        for j in range(len(self.grid[i])):
                            if self.can_place(i+1, j+1, obj):
                                return (i+1, j+1)  # Top-left position where the object can be placed
            
            if obj['obj_generation_mode'] == 'random_start':
                if last_long is not None:
                    assert 'spawn_long_gap' in obj
                    if last_long + obj['spawn_long_gap'] >= len(self.grid) - 5:
                        return None
                    start_long = np.random.randint(last_long + obj['spawn_long_gap'], min(len(self.grid) - 5, last_long + obj['spawn_long_gap'] + 1), 1)[0]
                    start_lat = np.random.randint(0, max(len(self.grid[0]) - 10, 1), 1)[0]
                    for i in range(start_long, len(self.grid)):
                        for j in range(start_lat, len(self.grid[i])):
                            if self.can_place(i+1, j+1, obj):
                                return (i+1, j+1)  # Top-left position where the object can be placed
                    return None
                
                start_long = np.random.randint(0, max(len(self.grid) - 10, 1), 1)[0]
                start_lat = np.random.randint(0, max(len(self.grid[0]) - 10, 1), 1)[0]
                for i in range(start_long, len(self.grid)):
                    for j in range(start_lat, len(self.grid[i])):
                            return (i+1, j+1)  # Top-left position where the object can be placed
                            
                return None
            
            if obj['obj_generation_mode'] == 'inverse':
                if last_long is not None:
                    assert 'spawn_long_gap' in obj
                    for i in range(len(self.grid) - 1, last_long + obj['spawn_long_gap'], -1):
                        for j in range(len(self.grid[i]) - 1, 0, -1):
                            if self.can_place(i+1, j+1, obj):
                                return (i+1, j+1)  # Top-left position where the object can be placed
                    return None
                
                for i in range(len(self.grid)):
                        for j in range(len(self.grid[i])):
                            if self.can_place(i+1, j+1, obj):
                                return (i+1, j+1)  # Top-left position where the object can be placed
                            
                return None
        else:
            if last_long is not None:
                assert 'spawn_long_gap' in obj
                for i in range(last_long + obj['spawn_long_gap'], len(self.grid)):
                    for j in range(len(self.grid[i])):
                        if self.can_place(i+1, j+1, obj):
                            return (i+1, j+1)  # Top-left position where the object can be placed
                return None
            
            for i in range(len(self.grid)):
                    for j in range(len(self.grid[i])):
                        if self.can_place(i+1, j+1, obj):
                            return (i+1, j+1)  # Top-left position where the object can be placed
                        
            return None

    def can_place(self, start_i, start_j, obj):
        """
        Check if the object can be placed starting from the given position.
        The object is placed with additional 2-cell buffer around it.
        Args:
            start_i (int): The starting row index of the grid.
            start_j (int): The starting column index of the grid.
            obj (dict): The object to be placed, with properties like length and width.
        Returns:
            bool: True if the object can be placed, False otherwise.
        """
        # Define the size of each cell (in meters, for example)
        cell_length = 1  # Length of each cell in meters
        cell_width = 1  # Width of each cell in meters

        # Calculate the number of cells the object spans, rounding up
        # Note we add 2 to the span to create a buffer around the object
        span_length = math.ceil(obj['general']['length'] / cell_length) + self.buffer
        span_width = math.ceil(obj['general']['width'] / cell_width) + self.buffer
        # span_length = math.ceil(obj['general']['width'] / cell_width) + 2
        # span_width = math.ceil(obj['general']['length'] / cell_length) + 2

        # Check if the object fits within the grid bounds
        if start_i + span_length > len(self.grid) or start_j + span_width > len(self.grid[0]):
            return False

        # Check for any overlaps with existing objects
        for i in range(start_i, start_i + span_length):
            for j in range(start_j, start_j + span_width):
                if self.grid[i][j].is_occupied():
                    return False

        return True

    def mark_occupied_cells(self, start_position, obj):
        """
        Mark the cells occupied by the object, with a buffer of 2 cells around it.
        Args:
            start_position (tuple): The top-left position where the object is placed.
            obj (dict): The object to be placed, with properties like length and width.
        Returns:
            Nothing, directly marks the cells as occupied.
        """
        start_i, start_j = start_position
        cell_length = 1  # Length of each cell in meters
        cell_width = 1  # Width of each cell in meters

        # Calculate the number of cells the object spans, rounding up
        span_length = math.ceil(obj['general']['length'] / cell_length) + self.buffer
        span_width = math.ceil(obj['general']['width'] / cell_width) + self.buffer

        # Mark the occupied cells
        for i in range(start_i, start_i + span_length):
            for j in range(start_j, start_j + span_width):
                self.grid[i][j].occupy(obj)

    def is_placement_possible(self):
        """
        Optional: Checks if there is any space left on the grid to place any object.

        Returns:
            bool: True if there is space available, False otherwise.
        """
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if not self.grid[i][j].is_occupied():
                    return True
        return False

class AssetManager(BaseManager):
    """
    This class is used to spawn static objects on the sidewalk
    The main idea is to create a grid for each region of the sidewalk (e.g., onsidewalk, outsidewalk, nearsidewalk)
    The main entry point is the reset method, which is called at the beginning of each episode.
    """
    PRIORITY = 9


    def __init__(self):
        super(AssetManager, self).__init__()
        self.debug = True
        self.density = self.engine.global_config['object_density']
        
        self.config = configReader()
        self.path_config = self.config.loadPath()
        self.init_static_adj_list() # Load the metainfo for all static objects
        self.get_attr() # Get the number and position of objects to spawn
        
        self.init_regular_objects()
        
        self.all_object_polygons = []
        
    def init_regular_objects(self):
        
        # regular objects
        self.regular_objects = {
            'Tree': [['nearroad_buffer_sidewalk', 'nearroad_sidewalk'], 0, 'parallel_only', []],
            'Lamp_post': [['nearroad_buffer_sidewalk', 'nearroad_sidewalk'], 1, 'parallel_only', []],
            'TrashCan': [['nearroad_buffer_sidewalk', 'nearroad_sidewalk'], 2, 'parallel_only', []],
            'Mailbox': [['main_sidewalk'], 3, 'random_start', []],
            'Telephone_booth': [['main_sidewalk'], 4, 'parallel_only', []],
            'FireHydrant': [['nearroad_buffer_sidewalk', 'nearroad_sidewalk'], 5, 'parallel_only', []],
            'Building': [['valid_region'], 6, 'normal', []],
            'Wall': [[''], 7, 'parallel_only', []],
            # 'Chair': [['farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'], 8, 'parallel_only', []],
            # 'Vegetation': [['farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'], 9, 'normal', []],
            # 'Advertising_board': [['farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'], 10, 'parallel_only', []],
            # 'Bench': [['farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'], 11, 'parallel_only', []],
            # 'Traffic_sign': [['nearroad_buffer_sidewalk', 'nearroad_sidewalk'], 12, 'parallel_only', []],
            # 'Bollard': [['nearroad_buffer_sidewalk', 'nearroad_sidewalk'], 13, 'parallel_only', []],
            # 'dog': [['main_sidewalk'], 14, 'random_start', []],
            # 'Vending_machine': [['main_sidewalk'], 15, 'random_start', []],
            # 'Bag': [['main_sidewalk'], 16, 'random_start', []],
            # 'Table': [['main_sidewalk'], 17, 'random_start', []],
            # 'Bonsai': [['farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'], 18, 'random_start', []],
            # 'Cone': [['main_sidewalk'], 19, 'random_start', []],
            # 'FoodTruck': [['valid_region'], 20, 'random_start', []],
            # 'Bike': [['valid_region'], 21, 'random_start', []],
            # 'Motorcycle': [['valid_region'], 22, 'random_start', []],
            # 'Scooter': [['valid_region'], 23, 'random_start', []],
            # 'Wheelchair': [['valid_region'], 24, 'random_start', []],
            'Chair': [['farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'], 8, 'parallel_only', []],
            'Vegetation': [['farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'], 9, 'normal', []],
            'Advertising_board': [['farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'], 10, 'parallel_only', []],
            'Bench': [['farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'], 11, 'parallel_only', []],
            'Traffic_sign': [['nearroad_buffer_sidewalk', 'nearroad_sidewalk'], 12, 'parallel_only', []],
            'Bollard': [['nearroad_buffer_sidewalk', 'nearroad_sidewalk'], 13, 'parallel_only', []],
            'dog': [['main_sidewalk'], 14, 'random_start', []],
            'Vending_machine': [['main_sidewalk'], 15, 'random_start', []],
            'Bag': [['main_sidewalk'], 16, 'random_start', []],
            'Table': [['main_sidewalk'], 17, 'random_start', []],
            'Bonsai': [['farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'], 18, 'random_start', []],
            'Cone': [['main_sidewalk'], 19, 'random_start', []],
            'FoodTruck': [['valid_region'], 20, 'random_start', []],
            'Bike': [['valid_region'], 21, 'random_start', []],
            'Motorcycle': [['valid_region'], 22, 'random_start', []],
            'Scooter': [['valid_region'], 23, 'random_start', []],
            'Wheelchair': [['valid_region'], 24, 'random_start', []],
        }
        self.regular_object_type_list = list(self.regular_objects.keys())
        
        # rank
        self.regular_object_by_rank = ['' for _ in range(len(list(self.regular_objects.keys())))]
        rank_list = [v[1] for v in self.regular_objects.values()]
        assert np.max(rank_list) == len(rank_list) - 1
        assert np.min(rank_list) == 0
        
        # object by rank
        for k, v in self.regular_objects.items():
            self.regular_object_by_rank[v[1]] = k
            
        # padding objects
        self.padding_objects = {
            'Tree': [['valid_region'], 0, 'normal', []],
        }
        self.padding_object_type_list = list(self.padding_objects.keys())
        self.padding_object_by_rank = ['' for _ in range(len(list(self.padding_objects.keys())))]
        rank_list = [v[1] for v in self.padding_objects.values()]
        assert np.max(rank_list) == len(rank_list) - 1
        assert np.min(rank_list) == 0
        
        # object by rank
        for k, v in self.padding_objects.items():
            self.padding_object_by_rank[v[1]] = k
            
        # intersection specific objects
        self.intersection_objects = {
            'Traffic_light': [['main_sidewalk'], 0, 'normal', []],
        }
        self.intersection_object_type_list = list(self.intersection_objects.keys())
        self.intersection_object_by_rank = ['' for _ in range(len(list(self.intersection_objects.keys())))]
        rank_list = [v[1] for v in self.intersection_objects.values()]
        assert np.max(rank_list) == len(rank_list) - 1
        assert np.min(rank_list) == 0
        
        # object by rank
        for k, v in self.intersection_objects.items():
            self.intersection_object_by_rank[v[1]] = k

    def init_static_adj_list(self):
        """
        Load the metainfo for all static objects
        """
        # The dictionary to store the metainfo for each object type
        # The key is the detail type, the value is a list of metainfo dictionaries
        # For example, key is bicycle, value is a list of metainfo dictionaries for all bicycle objects
        self.type_metainfo_dict = defaultdict(list)
        for root, dirs, files in os.walk(self.path_config["adj_parameter_folder"]):
            for file in files:
                # We only load the metainfo for static objects, skip cars
                if not file.lower().startswith("car"):
                    with open(os.path.join(root, file), 'r') as f:
                        loaded_metainfo = json.load(f)
                        self.type_metainfo_dict[loaded_metainfo['general']['detail_type']].append(loaded_metainfo)
        number = 0
        for k, v in self.type_metainfo_dict.items():
            number += len(v)

    def get_attr(self):
        """
        Get the number and position of objects to spawn
        Get the minimal gap between objects
        Get the rank of each objects
        """
        # The dictionary to store the number of objects to spawn for each type
        self.num_dict = dict()
        # The dictionary to store the position of objects to spawn for each type
        self.pos_dict = dict()
        
        self.interval_long = dict()
        self.interval_lat = dict()
        self.random_gap = dict()
        self.rank_dict = dict()
        # The dictionary to store the heading (rotation) of objects to spawn for each type
        self.heading_dict = dict()
        for detail_type in self.type_metainfo_dict.keys():
            self.num_dict[detail_type] = max(int(self.config.getSpawnNum(detail_type) * self.density), 1)
            self.pos_dict[detail_type]  = self.config.getSpawnPos(detail_type)
            self.interval_long[detail_type]  = max(min(int(self.config.getSpawnInterval(detail_type) * 1 / self.density), 40), 1)
            self.interval_lat[detail_type]  = self.config.getSpawnLatInterval(detail_type)
            self.random_gap[detail_type]  = self.config.getrandom_gap(detail_type)
            self.rank_dict[detail_type]  = self.config.get_rank(detail_type)
            if self.config.getSpawnHeading(detail_type):
                self.heading_dict[detail_type] = [heading * math.pi for heading in self.config.getSpawnHeading(detail_type)]
            else:
                self.heading_dict[detail_type] = [0, math.pi]
        
    @staticmethod
    def load_json_file(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    def before_reset(self):
        """
        Update episode level config to this manager and clean element or detach element
        """
        items = self.clear_objects([object_id for object_id in self.spawned_objects.keys()])
        self.spawned_objects = {}
    def reset(self):
        """
        Reset the manager and spawn objects on the sidewalk.
        Main entry point for the manager.
        """
        super(AssetManager, self).reset()
        
        seed = self.engine.global_random_seed
        import os, random
        import numpy as np
        import torch
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        self.generated_lane = []
        
        self.count = 0
        self.all_object_polygons = []
        engine = get_engine()
        assert len(self.spawned_objects.keys()) == 0
        # Iterate over all blocks in the current map (The blocks are the straight road segments in the map)
        # TODO: block by block
        for block in engine.current_map.blocks:
            if isinstance(block, FirstPGBlock):
                continue
            
            # Iterate over both lanes in the block (Each block has a positive and negative lane, representing the two directions of traffic)
            if block.ID == 'S':
                self.block_type = 'S'
                valid_lane = []
                # from metaurban.component.lane.circular_lane import CircularLane
                # graph = block.block_network.graph
                # for _from, to_dict in graph.items():
                #     for _to, lanes in to_dict.items():
                #         for _id, lane in enumerate(lanes):
                #             if isinstance(lane, CircularLane):
                #                 for lane_ in [block.positive_basic_lane, block.negative_basic_lane]:
                #                     if lane_.is_previous_lane_of(lane) or lane.is_previous_lane_of(lane_):
                #                         valid_lane.append(lane)
                # for _from, to_dict in graph.items():
                #     for _to, lanes in to_dict.items():
                #         for _id, lane in enumerate(lanes):
                #             for lane_ in [block.positive_basic_lane, block.negative_basic_lane] + valid_lane:
                #                 if lane_.is_previous_lane_of(lane) or lane.is_previous_lane_of(lane_):
                #                     valid_lane.append(lane)

                from metaurban.component.lane.circular_lane import CircularLane
                for lane in [block.positive_basic_lane, block.negative_basic_lane] + valid_lane:
                    if lane in self.generated_lane:
                        continue
                    self.generated_lane.append(lane)
                    # Create grids for each region
                    near_road_width = block.near_road_width
                    near_road_buffer_width = block.near_road_buffer_width
                    main_width = block.main_width
                    far_from_buffer_width = block.far_from_buffer_width
                    far_from_width = block.far_from_width
                    valid_house_width = block.valid_house_width
                    
                    width_list = [near_road_buffer_width, near_road_width, main_width, far_from_buffer_width, far_from_width, valid_house_width]
                    
                    self.sidewalk_type = block.sidewalk_type
                    if self.sidewalk_type == 'Narrow Sidewalk':
                        nearroad_buffer_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_buffer_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_buffer_sidewalk', nearroad_buffer_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Narrow Sidewalk with Trees':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Ribbon Sidewalk':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Neighborhood 1':
                        nearroad_buffer_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_buffer_sidewalk", lane, width_list, self.sidewalk_type))
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('nearroad_buffer_sidewalk', nearroad_buffer_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Neighborhood 2':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Medium Commercial':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Wide Commercial':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_buffer_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_buffer_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('farfromroad_buffer_sidewalk', farfromroad_buffer_sidewalk_grid), ('valid_region', valid_region_grid)]
                    else:
                        raise NotImplementedError
                    
                    # for region, grid in name_grid_list:                    
                    #     for i in range(5):
                    #         for j in range(len(grid[0])):
                    #             grid[i][j].occupied = True
                    #             grid[-i][j].occupied = True
                    
                    # init placers
                    object_placer_dict = {}
                    for region, grid in name_grid_list:                    
                        object_placer = ObjectPlacer(grid)
                        object_placer_dict.update({region: object_placer})
                        
                    # regular generation by rank
                    regular_object_by_rank = self.regular_object_by_rank
                    for obj_detail_type in regular_object_by_rank:
                        if obj_detail_type.lower() == 'wall' and self.sidewalk_type != 'Wide Commercial':
                            continue
                        generated_type = False
                        for region, grid in name_grid_list:
                            if region not in self.regular_objects[obj_detail_type][0]:
                                continue
                            else:
                                generated_type = False
                            object_placer = object_placer_dict[region]
                            obj_generation_mode = self.regular_objects[obj_detail_type][2]
                            self.retrieve_target_object_for_region(region, object_placer, obj_detail_type, obj_generation_mode)
                            
                            if generated_type:
                                break
                          
                    delta_scale = None  
                    regular_object_by_rank = self.padding_object_by_rank
                    for obj_detail_type in regular_object_by_rank:
                        if obj_detail_type.lower() == 'wall' and self.sidewalk_type != 'Wide Commercial':
                            continue
                        generated_type = False
                        for region, grid in [('valid_region', valid_region_grid)]:
                            if region not in self.padding_objects[obj_detail_type][0]:
                                continue
                            else:
                                generated_type = False
                            object_placer = object_placer_dict[region]
                            obj_generation_mode = self.padding_objects[obj_detail_type][2]
                            self.retrieve_target_object_for_region(region, object_placer, obj_detail_type, obj_generation_mode, delta_scale)
                            
                            if generated_type:
                                break
                                            
                    # detach to world
                    for region, grid in name_grid_list:
                        object_placer = object_placer_dict[region]
                        for obj_name, (grid_position, obj) in object_placer.placed_objects.items():
                            # Convert the grid position to a lane position
                            if 'region' in region:
                                coeff = 1
                            else:
                                coeff = 0
                            lane_position = self.convert_grid_to_lane_position([grid_position[0], grid_position[1] + (math.ceil(obj['general']['width']) + self.buffer) // 2 * coeff], lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                            span_length = math.ceil(obj['general']['length']) + self.buffer
                            span_width = math.ceil(obj['general']['width']) + self.buffer
                            start_lane_position = self.convert_grid_to_longitudelateral(grid_position, lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                            end_lane_position = self.convert_grid_to_longitudelateral((grid_position[0] + span_length, grid_position[1] + span_width), lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                            self.count += 1
                            
                            self.spawn_object(
                                TestObject,
                                force_spawn = True,
                                lane=lane,
                                position=lane_position,
                                static=self.engine.global_config["static_traffic_object"],
                                heading_theta=lane.heading_theta_at(lane_position[0]) + obj['general'].get(
                                    'heading', 0),
                                asset_metainfo=obj
                            )
                            
                            polygon = []
                            start_lat = start_lane_position[1]
                            side_lat = end_lane_position[1]
                            longs = []
                            for i in range(span_length):
                                lane_long = self.convert_grid_to_longitudelateral((grid_position[0] + i, grid_position[1] + (math.ceil(obj['general']['width'])) // 2), lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))[0]
                                longs.append(lane_long)
                            for k, lateral in enumerate([start_lat, side_lat]):
                                if k == 1:
                                    longs = longs[::-1]
                                for longitude in longs:
                                    longitude = min(lane.length + 0.1, longitude)
                                    point = lane.position(longitude, lateral)
                                    polygon.append([point[0], point[1]])
                            self.all_object_polygons.append(polygon)
            
            if block.ID == 'O':
                walkable_map = self.walkable_region_for_roundabout(self.current_map)
                self.block_type = 'O'
                valid_lane = []
                from metaurban.component.lane.circular_lane import CircularLane
                graph = block.block_network.graph
                for _from, to_dict in graph.items():
                    for _to, lanes in to_dict.items():
                        for _id, lane in enumerate(lanes):
                            if isinstance(lane, CircularLane):
                                for lane_ in [block.positive_basic_lane, block.negative_basic_lane]:
                                    if lane_.is_previous_lane_of(lane) or lane.is_previous_lane_of(lane_):
                                        valid_lane.append(lane)
                for _from, to_dict in graph.items():
                    for _to, lanes in to_dict.items():
                        for _id, lane in enumerate(lanes):
                            for lane_ in [block.positive_basic_lane, block.negative_basic_lane] + valid_lane:
                                if lane_.is_previous_lane_of(lane) or lane.is_previous_lane_of(lane_):
                                    if True:
                                        valid_lane.append(lane)
                valid_lane = list(set(valid_lane))
                if block.positive_basic_lane in valid_lane:
                    valid_lane.remove(block.positive_basic_lane)
                if block.negative_basic_lane in valid_lane:
                    valid_lane.remove(block.negative_basic_lane)
                
                pos_lane_list = []
                neg_lane_list = []
                
                for k, v in block._sockets.items():
                    pos_lane = v.get_positive_lanes(block._global_network)[-1]
                    # ray_localization
                    neg_lane = v.get_negative_lanes(block._global_network)[-1]
                    pos_lane_list.append(pos_lane)
                    neg_lane_list.append(neg_lane)
                    
                # valid_lane = []
                # from metaurban.component.lane.circular_lane import CircularLane
                # graph = block.block_network.graph
                # for _from, to_dict in graph.items():
                #     for _to, lanes in to_dict.items():
                #         for _id, lane in enumerate(lanes):
                #             if isinstance(lane, CircularLane):
                #                 for lane_ in [block.positive_basic_lane, block.negative_basic_lane] + pos_lane_list + neg_lane_list:
                #                     if lane_ in [block.positive_basic_lane] + pos_lane_list:
                #                         if lane_.is_previous_lane_of(lane):
                #                             valid_lane.append(lane)
                #                     else:
                #                         if lane.is_previous_lane_of(lane_):
                #                             valid_lane.append(lane)
                # for _from, to_dict in graph.items():
                #     for _to, lanes in to_dict.items():
                #         for _id, lane in enumerate(lanes):
                #             for lane_ in [block.positive_basic_lane, block.negative_basic_lane] + pos_lane_list + neg_lane_list + valid_lane:
                #                 if lane_ in [block.positive_basic_lane] + pos_lane_list:
                #                     if lane_ in [block.positive_basic_lane] + pos_lane_list:
                #                         if lane_.is_previous_lane_of(lane):
                #                             valid_lane.append(lane)
                #                     else:
                #                         if lane.is_previous_lane_of(lane_):
                #                             valid_lane.append(lane)
                # for lane in block.right_lanes:
                #     valid_lane.append(lane)  
                # valid_lane = list(set(valid_lane))
                # for lane in [block.positive_basic_lane, block.negative_basic_lane] + pos_lane_list + neg_lane_list:
                #     if lane in valid_lane:
                #         valid_lane.remove(lane)
                # for lane in valid_lane:
                #     flag = False
                #     for _lane in pos_lane_list + neg_lane_list:
                #         if lane_.is_previous_lane_of(lane) or lane.is_previous_lane_of(lane_):
                #             flag = True
                #     if not flag:
                #         valid_lane.remove(lane)

                from metaurban.component.lane.circular_lane import CircularLane
                for lane in [block.positive_basic_lane, block.negative_basic_lane] + pos_lane_list + neg_lane_list + valid_lane:
                    if lane in self.generated_lane:
                        continue
                    self.generated_lane.append(lane)
                    # Create grids for each region
                    near_road_width = block.near_road_width
                    near_road_buffer_width = block.near_road_buffer_width
                    main_width = block.main_width
                    far_from_buffer_width = block.far_from_buffer_width
                    far_from_width = block.far_from_width
                    valid_house_width = block.valid_house_width
                    
                    width_list = [near_road_buffer_width, near_road_width, main_width, far_from_buffer_width, far_from_width, valid_house_width]
                    
                    self.sidewalk_type = block.sidewalk_type
                    if self.sidewalk_type == 'Narrow Sidewalk':
                        nearroad_buffer_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_buffer_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_buffer_sidewalk', nearroad_buffer_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Narrow Sidewalk with Trees':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Ribbon Sidewalk':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Neighborhood 1':
                        nearroad_buffer_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_buffer_sidewalk", lane, width_list, self.sidewalk_type))
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('nearroad_buffer_sidewalk', nearroad_buffer_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Neighborhood 2':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Medium Commercial':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Wide Commercial':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_buffer_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_buffer_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('farfromroad_buffer_sidewalk', farfromroad_buffer_sidewalk_grid), ('valid_region', valid_region_grid)]
                    else:
                        raise NotImplementedError
                    
                    # for region, grid in name_grid_list:                    
                    #     for i in range(5):
                    #         for j in range(len(grid[0])):
                    #             grid[i][j].occupied = True
                    #             grid[-i][j].occupied = True
                    
                    # init placers
                    object_placer_dict = {}
                    for region, grid in name_grid_list:                    
                        object_placer = ObjectPlacer(grid)
                        object_placer_dict.update({region: object_placer})
                        
                    # regular generation by rank
                    regular_object_by_rank = self.regular_object_by_rank
                    for obj_detail_type in regular_object_by_rank:
                        if obj_detail_type.lower() == 'wall' and self.sidewalk_type != 'Wide Commercial':
                            continue
                        generated_type = False
                        for region, grid in name_grid_list:
                            if region not in self.regular_objects[obj_detail_type][0]:
                                continue
                            else:
                                generated_type = False
                            object_placer = object_placer_dict[region]
                            obj_generation_mode = self.regular_objects[obj_detail_type][2]
                            self.retrieve_target_object_for_region(region, object_placer, obj_detail_type, obj_generation_mode)
                            
                            if generated_type:
                                break
                          
                    delta_scale = None  
                    regular_object_by_rank = self.padding_object_by_rank
                    for obj_detail_type in regular_object_by_rank:
                        if obj_detail_type.lower() == 'wall' and self.sidewalk_type != 'Wide Commercial':
                            continue
                        generated_type = False
                        for region, grid in [('valid_region', valid_region_grid)]:
                            if region not in self.padding_objects[obj_detail_type][0]:
                                continue
                            else:
                                generated_type = False
                            object_placer = object_placer_dict[region]
                            obj_generation_mode = self.padding_objects[obj_detail_type][2]
                            self.retrieve_target_object_for_region(region, object_placer, obj_detail_type, obj_generation_mode, delta_scale)
                            
                            if generated_type:
                                break
                                            
                    # detach to world
                    for region, grid in name_grid_list:
                        object_placer = object_placer_dict[region]
                        for obj_name, (grid_position, obj) in object_placer.placed_objects.items():
                            # Convert the grid position to a lane position
                            if 'region' in region:
                                coeff = 1
                            else:
                                coeff = 0
                            lane_position = self.convert_grid_to_lane_position([grid_position[0], grid_position[1] + (math.ceil(obj['general']['width']) + self.buffer) // 2 * coeff], lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                            span_length = math.ceil(obj['general']['length']) + self.buffer
                            span_width = math.ceil(obj['general']['width']) + self.buffer
                            start_lane_position = self.convert_grid_to_longitudelateral(grid_position, lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                            end_lane_position = self.convert_grid_to_longitudelateral((grid_position[0] + span_length, grid_position[1] + span_width), lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                            
                            # check on the sidewalk
                            polygon = []
                            start_lat = start_lane_position[1]
                            side_lat = end_lane_position[1]
                            longs = []
                            for i in range(span_length):
                                lane_long = self.convert_grid_to_longitudelateral((grid_position[0] + i, grid_position[1] + (math.ceil(obj['general']['width'])) // 2), lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))[0]
                                longs.append(lane_long)
                            for k, lateral in enumerate([start_lat, side_lat]):
                                if k == 1:
                                    longs = longs[::-1]
                                for longitude in longs:
                                    longitude = min(lane.length + 0.1, longitude)
                                    point = lane.position(longitude, lateral)
                                    polygon.append([point[0], point[1]])
                            polygon_array = np.array(polygon)
                            polygon_array += self.mask_translate
                            polygon_array = np.floor(polygon_array).astype(int)
                            polygon_array = polygon_array.reshape((-1, 1, 2))
                            from shapely.geometry import Polygon
                            import cv2, copy
                            walkable_regions_mask = copy.deepcopy(walkable_map)
                            cv2.fillPoly(walkable_regions_mask, [polygon_array], [0, 0, 0])
                            # cv2.imwrite('./1112.png', walkable_regions_mask)
                            if ((walkable_regions_mask - walkable_map) ** 2).sum() == 0.:
                                continue
                                            
                            self.count += 1
                            
                            self.spawn_object(
                                TestObject,
                                force_spawn = True,
                                lane=lane,
                                position=lane_position,
                                static=self.engine.global_config["static_traffic_object"],
                                heading_theta=lane.heading_theta_at(lane_position[0]) + obj['general'].get(
                                    'heading', 0),
                                asset_metainfo=obj
                            )
                            
                            polygon = []
                            start_lat = start_lane_position[1]
                            side_lat = end_lane_position[1]
                            longs = []
                            for i in range(span_length):
                                lane_long = self.convert_grid_to_longitudelateral((grid_position[0] + i, grid_position[1] + (math.ceil(obj['general']['width'])) // 2), lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))[0]
                                longs.append(lane_long)
                            for k, lateral in enumerate([start_lat, side_lat]):
                                if k == 1:
                                    longs = longs[::-1]
                                for longitude in longs:
                                    longitude = min(lane.length + 0.1, longitude)
                                    point = lane.position(longitude, lateral)
                                    polygon.append([point[0], point[1]])
                            self.all_object_polygons.append(polygon)
            
            if block.ID == 'X':
                self.block_type = 'X'
                
                pos_lane_list = []
                neg_lane_list = []
                
                for k, v in block._sockets.items():
                    pos_lane = v.get_positive_lanes(block._global_network)[-1]
                    # ray_localization
                    neg_lane = v.get_negative_lanes(block._global_network)[-1]
                    pos_lane_list.append(pos_lane)
                    neg_lane_list.append(neg_lane)
                    
                valid_lane = []
                from metaurban.component.lane.circular_lane import CircularLane
                # graph = block.block_network.graph
                # for _from, to_dict in graph.items():
                #     for _to, lanes in to_dict.items():
                #         for _id, lane in enumerate(lanes):
                #             if isinstance(lane, CircularLane):
                #                 for lane_ in [block.positive_basic_lane, block.negative_basic_lane] + pos_lane_list + neg_lane_list:
                #                     if lane_ in [block.positive_basic_lane] + pos_lane_list:
                #                         if lane_.is_previous_lane_of(lane):
                #                             valid_lane.append(lane)
                #                     else:
                #                         if lane.is_previous_lane_of(lane_):
                #                             valid_lane.append(lane)
                # for _from, to_dict in graph.items():
                #     for _to, lanes in to_dict.items():
                #         for _id, lane in enumerate(lanes):
                #             for lane_ in [block.positive_basic_lane, block.negative_basic_lane] + pos_lane_list + neg_lane_list + valid_lane:
                #                 if lane_ in [block.positive_basic_lane] + pos_lane_list:
                #                     if lane_ in [block.positive_basic_lane] + pos_lane_list:
                #                         if lane_.is_previous_lane_of(lane):
                #                             valid_lane.append(lane)
                #                     else:
                #                         if lane.is_previous_lane_of(lane_):
                #                             valid_lane.append(lane)
                for lane in block.right_lanes:
                    valid_lane.append(lane)  
                valid_lane = list(set(valid_lane))
                for lane in [block.positive_basic_lane, block.negative_basic_lane] + pos_lane_list + neg_lane_list:
                    if lane in valid_lane:
                        valid_lane.remove(lane)

                from metaurban.component.lane.circular_lane import CircularLane
                for lane in [block.positive_basic_lane, block.negative_basic_lane] + pos_lane_list + neg_lane_list + valid_lane:
                    if lane in self.generated_lane:
                        continue
                    self.generated_lane.append(lane)
                    if isinstance(lane, CircularLane):
                        delta_scale = 1.5
                    else:
                        delta_scale = None
                    # Create grids for each region
                    near_road_width = block.near_road_width
                    near_road_buffer_width = block.near_road_buffer_width
                    main_width = block.main_width
                    far_from_buffer_width = block.far_from_buffer_width
                    far_from_width = block.far_from_width
                    valid_house_width = block.valid_house_width
                    
                    width_list = [near_road_buffer_width, near_road_width, main_width, far_from_buffer_width, far_from_width, valid_house_width]
                    
                    self.sidewalk_type = block.sidewalk_type
                    if self.sidewalk_type == 'Narrow Sidewalk':
                        nearroad_buffer_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_buffer_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_buffer_sidewalk', nearroad_buffer_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Narrow Sidewalk with Trees':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Ribbon Sidewalk':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Neighborhood 1':
                        nearroad_buffer_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_buffer_sidewalk", lane, width_list, self.sidewalk_type))
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('nearroad_buffer_sidewalk', nearroad_buffer_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Neighborhood 2':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Medium Commercial':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Wide Commercial':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_buffer_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_buffer_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('farfromroad_buffer_sidewalk', farfromroad_buffer_sidewalk_grid), ('valid_region', valid_region_grid)]
                    else:
                        raise NotImplementedError
                    
                    # init placers
                    object_placer_dict = {}
                    # for region, grid in name_grid_list:                    
                    #     for i in range(5):
                    #         for j in range(len(grid[0])):
                    #             grid[i][j].occupied = True
                    #             grid[-i][j].occupied = True
                    if lane == block.positive_basic_lane:
                        for region, grid in name_grid_list:                    
                            for i in range(10):
                                for j in range(len(grid[0])):
                                    grid[i][j].occupied = True

                    for region, grid in name_grid_list:                    
                        object_placer = ObjectPlacer(grid)
                        object_placer_dict.update({region: object_placer})
                        
                    # regular generation by rank
                    regular_object_by_rank = self.regular_object_by_rank
                    for obj_detail_type in regular_object_by_rank:
                        if obj_detail_type.lower() == 'wall' and self.sidewalk_type != 'Wide Commercial':
                            continue
                        generated_type = False
                        for region, grid in name_grid_list:
                            if region not in self.regular_objects[obj_detail_type][0]:
                                continue
                            else:
                                generated_type = False
                            object_placer = object_placer_dict[region]
                            obj_generation_mode = self.regular_objects[obj_detail_type][2]
                            self.retrieve_target_object_for_region(region, object_placer, obj_detail_type, obj_generation_mode, delta_scale)
                            
                            if generated_type:
                                break
                            
                    # padding valid region
                    # regular generation by rank
                    regular_object_by_rank = self.padding_object_by_rank
                    for obj_detail_type in regular_object_by_rank:
                        if obj_detail_type.lower() == 'wall' and self.sidewalk_type != 'Wide Commercial':
                            continue
                        generated_type = False
                        for region, grid in [('valid_region', valid_region_grid)]:
                            if region not in self.padding_objects[obj_detail_type][0]:
                                continue
                            else:
                                generated_type = False
                            object_placer = object_placer_dict[region]
                            obj_generation_mode = self.padding_objects[obj_detail_type][2]
                            self.retrieve_target_object_for_region(region, object_placer, obj_detail_type, obj_generation_mode, delta_scale)
                            
                            if generated_type:
                                break
                            
                    # specific objects
                    if lane == block.positive_basic_lane or lane in neg_lane_list:
                        regular_object_by_rank = self.intersection_object_by_rank
                        for obj_detail_type in regular_object_by_rank:
                            if obj_detail_type.lower() == 'wall' and self.sidewalk_type != 'Wide Commercial':
                                continue
                            self.intersection_objects[obj_detail_type][2] = 'inverse'
                            generated_type = False
                            for region, grid in name_grid_list:
                                if region not in self.intersection_objects[obj_detail_type][0]:
                                    continue
                                else:
                                    generated_type = False
                                object_placer = object_placer_dict[region]
                                obj_generation_mode = self.intersection_objects[obj_detail_type][2]
                                self.retrieve_target_object_for_region(region, object_placer, obj_detail_type, obj_generation_mode, delta_scale)
                                
                                if generated_type:
                                    break
                    else:
                        regular_object_by_rank = self.intersection_object_by_rank
                        for obj_detail_type in regular_object_by_rank:
                            if obj_detail_type.lower() == 'wall' and self.sidewalk_type != 'Wide Commercial':
                                continue
                            self.intersection_objects[obj_detail_type][2] = 'normal'
                            generated_type = False
                            for region, grid in name_grid_list:
                                if region not in self.intersection_objects[obj_detail_type][0]:
                                    continue
                                else:
                                    generated_type = False
                                object_placer = object_placer_dict[region]
                                obj_generation_mode = self.intersection_objects[obj_detail_type][2]
                                self.retrieve_target_object_for_region(region, object_placer, obj_detail_type, obj_generation_mode, delta_scale)
                                
                                if generated_type:
                                    break
                                            
                    # detach to world
                    for region, grid in name_grid_list:
                        object_placer = object_placer_dict[region]
                        for obj_name, (grid_position, obj) in object_placer.placed_objects.items():
                        # Convert the grid position to a lane position
                            if 'region' in region:
                                coeff = 1
                            else:
                                coeff = 0
                            lane_position = self.convert_grid_to_lane_position([grid_position[0], grid_position[1] + (math.ceil(obj['general']['width']) + self.buffer) // 2 * coeff], lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                            span_length = math.ceil(obj['general']['length']) + self.buffer
                            span_width = math.ceil(obj['general']['width']) + self.buffer
                            start_lane_position = self.convert_grid_to_longitudelateral(grid_position, lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                            end_lane_position = self.convert_grid_to_longitudelateral((grid_position[0] + span_length, grid_position[1] + span_width), lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                            self.count += 1
                            
                            self.spawn_object(
                                TestObject,
                                force_spawn = True,
                                lane=lane,
                                position=lane_position,
                                static=self.engine.global_config["static_traffic_object"],
                                heading_theta=lane.heading_theta_at(lane_position[0]) + obj['general'].get(
                                    'heading', 0),
                                asset_metainfo=obj
                            )
                            
                            polygon = []
                            start_lat = start_lane_position[1]
                            side_lat = end_lane_position[1]
                            longs = []
                            for i in range(span_length):
                                lane_long = self.convert_grid_to_longitudelateral((grid_position[0] + i, grid_position[1] + (math.ceil(obj['general']['width'])) // 2), lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))[0]
                                longs.append(lane_long)
                            for k, lateral in enumerate([start_lat, side_lat]):
                                if k == 1:
                                    longs = longs[::-1]
                                for longitude in longs:
                                    longitude = min(lane.length + 0.1, longitude)
                                    point = lane.position(longitude, lateral)
                                    polygon.append([point[0], point[1]])
                            self.all_object_polygons.append(polygon)     
                
            if block.ID == 'C':
                
                self.block_type = 'C'
                valid_lane = []
                from metaurban.component.lane.circular_lane import CircularLane
                graph = block.block_network.graph
                for _from, to_dict in graph.items():
                    for _to, lanes in to_dict.items():
                        for _id, lane in enumerate(lanes):
                            if isinstance(lane, CircularLane):
                                for lane_ in [block.positive_basic_lane, block.negative_basic_lane]:
                                    if lane_.is_previous_lane_of(lane) or lane.is_previous_lane_of(lane_):
                                        valid_lane.append(lane)
                for _from, to_dict in graph.items():
                    for _to, lanes in to_dict.items():
                        for _id, lane in enumerate(lanes):
                            for lane_ in [block.positive_basic_lane, block.negative_basic_lane] + valid_lane:
                                if lane_.is_previous_lane_of(lane) or lane.is_previous_lane_of(lane_):
                                    valid_lane.append(lane)
                valid_lane = list(set(valid_lane))
                if block.positive_basic_lane in valid_lane:
                    valid_lane.remove(block.positive_basic_lane)
                if block.negative_basic_lane in valid_lane:
                    valid_lane.remove(block.negative_basic_lane)
                
                for lane in [block.positive_basic_lane, block.negative_basic_lane] + valid_lane:
                    if lane in self.generated_lane:
                        continue
                    self.generated_lane.append(lane)
                    if isinstance(lane, CircularLane):
                        delta_scale = 2.
                    else:
                        delta_scale = None
                    # Create grids for each region
                    near_road_width = block.near_road_width
                    near_road_buffer_width = block.near_road_buffer_width
                    main_width = block.main_width
                    far_from_buffer_width = block.far_from_buffer_width
                    far_from_width = block.far_from_width
                    valid_house_width = block.valid_house_width
                    
                    width_list = [near_road_buffer_width, near_road_width, main_width, far_from_buffer_width, far_from_width, valid_house_width]
                    
                    self.sidewalk_type = block.sidewalk_type
                    if self.sidewalk_type == 'Narrow Sidewalk':
                        nearroad_buffer_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_buffer_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_buffer_sidewalk', nearroad_buffer_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Narrow Sidewalk with Trees':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Ribbon Sidewalk':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Neighborhood 1':
                        nearroad_buffer_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_buffer_sidewalk", lane, width_list, self.sidewalk_type))
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('nearroad_buffer_sidewalk', nearroad_buffer_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Neighborhood 2':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Medium Commercial':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('valid_region', valid_region_grid)]
                    elif self.sidewalk_type == 'Wide Commercial':
                        nearroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("nearroad_sidewalk", lane, width_list, self.sidewalk_type))
                        main_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("main_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_sidewalk", lane, width_list, self.sidewalk_type))
                        farfromroad_buffer_sidewalk_grid = self.create_grid(lane, self.calculate_lateral_range("farfromroad_buffer_sidewalk", lane, width_list, self.sidewalk_type))
                        valid_region_grid = self.create_grid(lane, self.calculate_lateral_range("valid_region", lane, width_list, self.sidewalk_type))
                        name_grid_list = [('nearroad_sidewalk', nearroad_sidewalk_grid), ('main_sidewalk', main_sidewalk_grid), ('farfromroad_sidewalk', farfromroad_sidewalk_grid), ('farfromroad_buffer_sidewalk', farfromroad_buffer_sidewalk_grid), ('valid_region', valid_region_grid)]
                    else:
                        raise NotImplementedError
                    
                    # for region, grid in name_grid_list:                    
                    #     for i in range(5):
                    #         for j in range(len(grid[0])):
                    #             grid[i][j].occupied = True
                    #             grid[-i][j].occupied = True
                    
                    # init placers
                    object_placer_dict = {}
                    for region, grid in name_grid_list:                    
                        object_placer = ObjectPlacer(grid)
                        object_placer_dict.update({region: object_placer})
                        
                    # regular generation by rank
                    regular_object_by_rank = self.regular_object_by_rank
                    for obj_detail_type in regular_object_by_rank:
                        if obj_detail_type.lower() == 'wall' and self.sidewalk_type != 'Wide Commercial':
                            continue
                        generated_type = False
                        # if obj_detail_type.lower() == 'building':
                        #     if  isinstance(lane, CircularLane):
                        #         continue
                        for region, grid in name_grid_list:
                            if region not in self.regular_objects[obj_detail_type][0]:
                                continue
                            else:
                                generated_type = False
                            object_placer = object_placer_dict[region]
                            obj_generation_mode = self.regular_objects[obj_detail_type][2]
                            self.retrieve_target_object_for_region(region, object_placer, obj_detail_type, obj_generation_mode, delta_scale)
                            
                            if generated_type:
                                break
                            
                    # padding valid region
                    # regular generation by rank
                    regular_object_by_rank = self.padding_object_by_rank
                    for obj_detail_type in regular_object_by_rank:
                        if obj_detail_type.lower() == 'wall' and self.sidewalk_type != 'Wide Commercial':
                            continue
                        generated_type = False
                        for region, grid in [('valid_region', valid_region_grid)]:
                            if region not in self.padding_objects[obj_detail_type][0]:
                                continue
                            else:
                                generated_type = False
                            object_placer = object_placer_dict[region]
                            obj_generation_mode = self.padding_objects[obj_detail_type][2]
                            self.retrieve_target_object_for_region(region, object_placer, obj_detail_type, obj_generation_mode, delta_scale)
                            
                            if generated_type:
                                break
                                            
                    # detach to world
                    for region, grid in name_grid_list:
                        object_placer = object_placer_dict[region]
                        for obj_name, (grid_position, obj) in object_placer.placed_objects.items():
                        # Convert the grid position to a lane position
                            if 'region' in region:
                                coeff = 1
                            else:
                                coeff = 0
                            lane_position = self.convert_grid_to_lane_position([grid_position[0], grid_position[1] + (math.ceil(obj['general']['width']) + self.buffer) // 2 * coeff], lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                            span_length = math.ceil(obj['general']['length']) + self.buffer
                            span_width = math.ceil(obj['general']['width']) + self.buffer
                            start_lane_position = self.convert_grid_to_longitudelateral(grid_position, lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                            end_lane_position = self.convert_grid_to_longitudelateral((grid_position[0] + span_length, grid_position[1] + span_width), lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                            self.count += 1
                            
                            self.spawn_object(
                                TestObject,
                                force_spawn = True,
                                lane=lane,
                                position=lane_position,
                                static=self.engine.global_config["static_traffic_object"],
                                heading_theta=lane.heading_theta_at(lane_position[0]) + obj['general'].get(
                                    'heading', 0),
                                asset_metainfo=obj
                            )
                            
                            polygon = []
                            start_lat = start_lane_position[1]
                            side_lat = end_lane_position[1]
                            longs = []
                            for i in range(span_length):
                                lane_long = self.convert_grid_to_longitudelateral((grid_position[0] + i, grid_position[1] + (math.ceil(obj['general']['width'])) // 2), lane,
                                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))[0]
                                longs.append(lane_long)
                            for k, lateral in enumerate([start_lat, side_lat]):
                                if k == 1:
                                    longs = longs[::-1]
                                for longitude in longs:
                                    longitude = min(lane.length + 0.1, longitude)
                                    point = lane.position(longitude, lateral)
                                    polygon.append([point[0], point[1]])
                            self.all_object_polygons.append(polygon)
                            
                # for lane in  [block.positive_basic_lane, block.negative_basic_lane] +  valid_lane:
                #     if isinstance(lane, CircularLane):
                #         delta_scale = 2.
                #     else:
                #         delta_scale = None
                        
                #     object_placer_dict = {}
                #     for region, grid in name_grid_list:                    
                #         object_placer = ObjectPlacer(grid)
                #         object_placer_dict.update({region: object_placer})
                        
                #     # regular generation by rank
                #     regular_object_by_rank = self.padding_object_by_rank
                #     for obj_detail_type in regular_object_by_rank:
                #         generated_type = False
                #         for region, grid in [('valid_region', valid_region_grid)]:
                #             if region not in self.padding_objects[obj_detail_type][0]:
                #                 continue
                #             else:
                #                 generated_type = False
                #             object_placer = object_placer_dict[region]
                #             obj_generation_mode = self.padding_objects[obj_detail_type][2]
                #             self.retrieve_target_object_for_region(region, object_placer, obj_detail_type, obj_generation_mode, delta_scale)
                            
                #             if generated_type:
                #                 break
                                            
                #     # detach to world
                #     for region, grid in [('valid_region', valid_region_grid)]:
                #         object_placer = object_placer_dict[region]
                #         for obj_name, (grid_position, obj) in object_placer.placed_objects.items():
                #         # Convert the grid position to a lane position
                #             if 'region' in region:
                #                 coeff = 1
                #             else:
                #                 coeff = 0
                #             lane_position = self.convert_grid_to_lane_position([grid_position[0], grid_position[1] + (math.ceil(obj['general']['width']) + self.buffer) // 2 * coeff], lane,
                #                                                             self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                #             span_length = math.ceil(obj['general']['length']) + self.buffer
                #             span_width = math.ceil(obj['general']['width']) + self.buffer
                #             start_lane_position = self.convert_grid_to_longitudelateral(grid_position, lane,
                #                                                             self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                #             end_lane_position = self.convert_grid_to_longitudelateral((grid_position[0] + span_length, grid_position[1] + span_width), lane,
                #                                                             self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                #             self.count += 1
                            
                #             self.spawn_object(
                #                 TestObject,
                #                 force_spawn = True,
                #                 lane=lane,
                #                 position=lane_position,
                #                 static=self.engine.global_config["static_traffic_object"],
                #                 heading_theta=lane.heading_theta_at(lane_position[0]) + obj['general'].get(
                #                     'heading', 0),
                #                 asset_metainfo=obj
                #             )
                            
                #             polygon = []
                #             start_lat = start_lane_position[1]
                #             side_lat = end_lane_position[1]
                #             longs = []
                #             for i in range(span_length):
                #                 lane_long = self.convert_grid_to_longitudelateral((grid_position[0] + i, grid_position[1] + (math.ceil(obj['general']['width'])) // 2), lane,
                #                                                             self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))[0]
                #                 longs.append(lane_long)
                #             for k, lateral in enumerate([start_lat, side_lat]):
                #                 if k == 1:
                #                     longs = longs[::-1]
                #                 for longitude in longs:
                #                     longitude = min(lane.length + 0.1, longitude)
                #                     point = lane.position(longitude, lateral)
                #                     polygon.append([point[0], point[1]])
                #             self.all_object_polygons.append(polygon)                    
                # Retrieve and place objects for each region
                # for region, grid in name_grid_list:
                #     # Create an ObjectPlacer object to place objects on the grid
                    
                #     object_placer = ObjectPlacer(grid)
                #     count_tmp = 0
                #     # Place objects on the virtual grid, but not actually spawned yet
                #     self.retrieve_objects_for_region(region, object_placer)
                #     # self.visualize_grid(grid)
                #     # Iterate over the placed objects and spawn them in the simulation
                #     for obj_name, (grid_position, obj) in object_placer.placed_objects.items():
                #         # Convert the grid position to a lane position
                #         if 'region' not in region:
                #             coeff = 0
                #         else:
                #             coeff = 1
                #         lane_position = self.convert_grid_to_lane_position([grid_position[0], grid_position[1] + (math.ceil(obj['general']['width']) + self.buffer) // 2 * coeff], lane,
                #                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                #         span_length = math.ceil(obj['general']['length']) + self.buffer
                #         span_width = math.ceil(obj['general']['width']) + self.buffer
                #         start_lane_position = self.convert_grid_to_longitudelateral(grid_position, lane,
                #                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                #         end_lane_position = self.convert_grid_to_longitudelateral((grid_position[0] + span_length, grid_position[1] + span_width), lane,
                #                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))
                #         self.count += 1
                #         count_tmp += 1
                        
                #         self.spawn_object(
                #             TestObject,
                #             force_spawn = True,
                #             lane=lane,
                #             position=lane_position,
                #             static=self.engine.global_config["static_traffic_object"],
                #             heading_theta=lane.heading_theta_at(lane_position[0]) + obj['general'].get(
                #                 'heading', 0),
                #             asset_metainfo=obj
                #         )
                        
                #         polygon = []
                #         start_lat = start_lane_position[1]
                #         side_lat = end_lane_position[1]
                #         longs = []
                #         for i in range(span_length):
                #             lane_long = self.convert_grid_to_longitudelateral((grid_position[0] + i, grid_position[1] + span_width), lane,
                #                                                            self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type))[0]
                #             longs.append(lane_long)
                #         for k, lateral in enumerate([start_lat, side_lat]):
                #             if k == 1:
                #                 longs = longs[::-1]
                #             for longitude in longs:
                #                 longitude = min(lane.length + 0.1, longitude)
                #                 point = lane.position(longitude, lateral)
                #                 polygon.append([point[0], point[1]])
                #         self.all_object_polygons.append(polygon)

                #     print("======For region:{} Spawned {} objects=======".format(region, count_tmp))
        # print('==================')
        # print(self.count)
        # print(self.count)
        # print('==================')
        self.engine.objects_counts = self.count
        self._get_walkable_regions(self.current_map)

    def create_grid(self, lane, lateral_range):
        """
        Create a grid for a given lane and lateral range.
        Args:
            lane (Lane): The lane object.
            lateral_range (tuple): The start and end of the lateral range for the grid.
        Returns:
            list: A 2D list of GridCell objects representing the grid."""
        # Define the size of each cell (in meters, for example)
        cell_length = 1  # Length of a cell along the lane
        cell_width = 1  # Width of a cell across the lane

        # Calculate the number of cells along the lane and across its width
        from metaurban.constants import PGDrivableAreaProperty
        if self.block_type == 'X':
            num_cells_long = int((lane.length) / cell_length)
        else:
            num_cells_long = int((lane.length + PGDrivableAreaProperty.SIDEWALK_LENGTH) / cell_length)
        num_cells_lat = int((lateral_range[1] - lateral_range[0]) / cell_width)

        # Create the grid as a 2D array of GridCell objects
        grid = [[GridCell(position=(i * cell_length, j * cell_width + lateral_range[0]))
                 for j in range(num_cells_lat)] for i in range(num_cells_long)]

        return grid
    
    def retrieve_target_object_for_region(self, region, object_placer, obj_detail_type=None, obj_generation_mode=None, delta_scale=None):
        
        seed = self.engine.global_seed
        import os, random
        import numpy as np
        import torch
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        detail_type_groups = defaultdict(list)
        object_counts = defaultdict(int)  # Track how many objects of each type have been tried
        # last generated position
        last_object_postion_long = defaultdict()
        for detail_type, objects in self.type_metainfo_dict.items():
            if detail_type != obj_detail_type:
                continue
            
            # change information about the tree
            if obj_detail_type.lower() == 'tree':
                new_list = []
                for obj in objects:
                    obj['general']['width'] = 2.
                    obj['general']['length'] = 2.
                    obj['general']['bounding_box'] = [[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]]
                    new_list.append(obj)
                self.type_metainfo_dict[detail_type] = new_list
            
            # Note, we only place objects in the specified region
            self.pos_dict[detail_type] = region
            if self.pos_dict[detail_type] == region:
                for idx, obj in enumerate(objects):
                    unique_id = (detail_type, idx)
                    detail_type_groups[detail_type].append(unique_id)
                    last_object_postion_long[detail_type] = 0
        self.buffer = 2 if 'near' not in region else 0
        if obj_detail_type.lower() == 'building':
            self.buffer = 10
        # Set to keep track of tried objects
        any_object_placed = True
        
        iteration_time = 0
            
        # Continue round-robin placement until all objects are tried or count limit is reached
        if not hasattr(self, 'placed_types'):
            self.placed_types = {}
        self.placed_types[region] = []
        while any_object_placed or iteration_time < 400:
            any_object_placed = False
            # Iterate over all detail types and try to place objects
            for detail_type, object_ids in detail_type_groups.items():
                # If we have already placed the required number of objects, skip
                if object_counts[detail_type] < self.num_dict[detail_type]:
                    # Randomly select an object to place
                    obj_id = random.sample(object_ids, 1)[0]
                    obj = self.type_metainfo_dict[detail_type][obj_id[1]]  # Retrieve the actual object
                    interval_long = self.interval_long[detail_type]
                    interval_lat = self.interval_lat[detail_type]
                    random_start = self.random_gap[detail_type]
                    if random_start:
                        offset = np.random.randint(0, 5, 1)[0]
                    else:
                        offset = 0
                    if region == 'valid_region':
                        if object_counts[detail_type] < 1:
                            offset = 10
                    else:
                        if object_counts[detail_type] < 1:
                            offset = 10 + len(self.placed_types[region]) * 8
                    obj['spawn_long_gap'] = interval_long
                    
                    if delta_scale is not None and region == 'valid_region':
                        obj['spawn_long_gap'] = int(interval_long * delta_scale)
                    
                    if obj_detail_type.lower() == 'tree' and region == 'valid_region':
                        interval_long = int(2 * 1 / self.density)
                        obj['spawn_long_gap'] = interval_long
                        offset = 0
                        
                    object_placer.buffer = self.buffer
                    obj['obj_generation_mode'] = obj_generation_mode
                    if 'spawn_long_gap' in obj:
                        # print('Spawning Longititude Constraint is added')
                        if object_counts[detail_type] > 0:
                            generated_, last_long = object_placer.place_object(obj, last_object_postion_long[detail_type] + offset)
                            if generated_:
                                last_object_postion_long[detail_type] = last_long
                                object_counts[detail_type] += 1
                                any_object_placed = True
                        else:
                            generated_, last_long = object_placer.place_object(obj, last_object_postion_long[detail_type] + offset)
                            if generated_:
                                last_object_postion_long[detail_type] = last_long
                                object_counts[detail_type] += 1
                                any_object_placed = True
                    elif object_placer.place_object(obj):
                        object_counts[detail_type] += 1
                        any_object_placed = True
                    
                    if detail_type not in self.placed_types[region]:
                        self.placed_types[region].append(detail_type)
                
                iteration_time += 1
                
                if not any_object_placed and all(object_counts[dt] >= self.num_dict[dt] for dt in detail_type_groups):
                    break

    def retrieve_objects_for_region(self, region, object_placer, obj_detail_type=None):
        """
        Retrieve objects for a given region and place them on the grid.
        Args:
            region (str): The region (e.g., 'onsidewalk', 'nearsidewalk', 'outsidewalk').
            object_placer (ObjectPlacer): The ObjectPlacer object to place objects on the grid.
        Returns:
            Nothing, directly places objects on the grid.
        """
        # Group objects by detail type and initialize counters for each type
        detail_type_groups = defaultdict(list)
        object_counts = defaultdict(int)  # Track how many objects of each type have been tried
        # last generated position
        last_object_postion_long = defaultdict()
        # iterate over all objects and group them by detail type
        for detail_type, objects in self.type_metainfo_dict.items():
            # print(detail_type, objects)
            # import pdb; pdb.set_trace()
            # Note, we only place objects in the specified region
            if self.pos_dict[detail_type] == region:
                for idx, obj in enumerate(objects):
                    unique_id = (detail_type, idx)
                    detail_type_groups[detail_type].append(unique_id)
                    last_object_postion_long[detail_type] = 0
        
        self.buffer = 2 if 'near' not in region else 0
        # Set to keep track of tried objects
        any_object_placed = True
        # Continue round-robin placement until all objects are tried or count limit is reached
        while any_object_placed:
            any_object_placed = False
            # Iterate over all detail types and try to place objects
            for detail_type, object_ids in detail_type_groups.items():
                # If we have already placed the required number of objects, skip
                if object_counts[detail_type] < self.num_dict[detail_type]:
                    # Randomly select an object to place
                    obj_id = random.sample(object_ids, 1)[0]
                    obj = self.type_metainfo_dict[detail_type][obj_id[1]]  # Retrieve the actual object
                    interval_long = self.interval_long[detail_type]
                    interval_lat = self.interval_lat[detail_type]
                    random_start = self.random_gap[detail_type]
                    if random_start:
                        offset = np.random.randint(0, 5, 1)[0]
                    else:
                        offset = 0
                    obj['spawn_long_gap'] = interval_long
                    object_placer.buffer = self.buffer
                    if 'spawn_long_gap' in obj:
                        # print('Spawning Longititude Constraint is added')
                        if object_counts[detail_type] > 0:
                            generated_, last_long = object_placer.place_object(obj, last_object_postion_long[detail_type] + offset)
                            if generated_:
                                last_object_postion_long[detail_type] = last_long
                                object_counts[detail_type] += 1
                                any_object_placed = True
                        else:
                            generated_ = object_placer.place_object(obj, last_object_postion_long[detail_type] + offset)
                            if generated_:
                                last_object_postion_long[detail_type] = 0
                                object_counts[detail_type] += 1
                                any_object_placed = True
                    elif object_placer.place_object(obj):
                        object_counts[detail_type] += 1
                        any_object_placed = True
                if not any_object_placed and all(object_counts[dt] >= self.num_dict[dt] for dt in detail_type_groups):
                    break

    def convert_grid_to_lane_position(self, grid_position, lane, lateral_range):
        """
        Convert a grid position to a lane position.
        Args:
            grid_position (tuple): The grid position as a tuple (i, j).
            lane (Lane): The lane object.
            lateral_range (tuple): The start and end of the lateral range for the grid.
        Returns:
            tuple: The lane position as a tuple (longitude, lateral).
        """
        grid_i, grid_j = grid_position
        cell_length = 1  # Length of a cell along the lane, should be consistent with create_grid method
        cell_width = 1   # Width of a cell across the lane, should be consistent with create_grid method

        # Convert grid position to longitudinal and lateral position relative to the lane
        longitude = grid_i * cell_length
        lateral = lateral_range[0] + grid_j * cell_width

        return lane.position(longitude, lateral)
    
    def convert_grid_to_longitudelateral(self, grid_position, lane, lateral_range):
        """
        Convert a grid position to a lane position.
        Args:
            grid_position (tuple): The grid position as a tuple (i, j).
            lane (Lane): The lane object.
            lateral_range (tuple): The start and end of the lateral range for the grid.
        Returns:
            tuple: The lane position as a tuple (longitude, lateral).
        """
        grid_i, grid_j = grid_position
        cell_length = 1  # Length of a cell along the lane, should be consistent with create_grid method
        cell_width = 1   # Width of a cell across the lane, should be consistent with create_grid method

        # Convert grid position to longitudinal and lateral position relative to the lane
        longitude = grid_i * cell_length
        lateral = lateral_range[0] + grid_j * cell_width

        return (longitude, lateral)
    
    def visualize_grid(self, grid):
        """
        Visualize the grid by printing it to the console.
        Args:
            grid (list): A 2D list of GridCell objects representing the grid.
        Returns:
            Nothing, directly prints the grid to the console.
        """
        for row in grid:
            for cell in row:
                # Assuming each cell has a method 'is_occupied' to check if it's occupied
                char = 'X' if cell.is_occupied() else '.'
                print(char, end=' ')
            print()  # Newline after each row
    def calculate_lateral_range(self, region, lane, width_list, sidewalk_type):
        """
        Calculate the lateral range for a given region of a lane.

        Args:
            region (str): The region (e.g., 'sidewalk', 'nearsidewalk', 'outsidewalk').
            lane (Lane): The lane object.

        Returns:
            tuple: A tuple representing the start and end of the lateral range.
        """
        if sidewalk_type == 'Narrow Sidewalk':
            assert width_list[0] is not None
            assert width_list[2] is not None
        elif sidewalk_type == 'Narrow Sidewalk with Trees':
            assert width_list[1] is not None
            assert width_list[2] is not None
        elif sidewalk_type == 'Ribbon Sidewalk':
            assert width_list[1] is not None
            assert width_list[2] is not None
            assert width_list[4] is not None
        elif sidewalk_type == 'Neighborhood 1':
            assert width_list[0] is not None
            assert width_list[1] is not None
            assert width_list[2] is not None
        elif sidewalk_type == 'Neighborhood 2':
            assert width_list[1] is not None
            assert width_list[2] is not None
            assert width_list[4] is not None
        elif sidewalk_type == 'Medium Commercial':
            assert width_list[1] is not None
            assert width_list[2] is not None
            assert width_list[4] is not None
        elif sidewalk_type == 'Wide Commercial':
            assert width_list[1] is not None
            assert width_list[2] is not None
            assert width_list[3] is not None
            assert width_list[4] is not None
        else:
            raise NotImplementedError 
        
        if sidewalk_type == 'Narrow Sidewalk':
            if region == 'nearroad_buffer_sidewalk':
                return (lane.width_at(0) / 2, lane.width_at(0) / 2 + width_list[0])
            elif region == 'main_sidewalk':
                return (lane.width_at(0) / 2 + width_list[0], lane.width_at(0) / 2 + width_list[0] + width_list[2])
            elif region == 'valid_region':
                return (lane.width_at(0) / 2 + width_list[0] + width_list[2], lane.width_at(0) / 2 + width_list[0] + width_list[2] + width_list[-1])
            else:
                raise ValueError("Incorrect region type")
        elif sidewalk_type == 'Narrow Sidewalk with Trees':
            if region == 'nearroad_sidewalk':
                return (lane.width_at(0) / 2, lane.width_at(0) / 2 + width_list[1])
            elif region == 'main_sidewalk':
                return (lane.width_at(0) / 2 + width_list[1], lane.width_at(0) / 2 + width_list[1] + width_list[2])
            elif region == 'valid_region':
                return (lane.width_at(0) / 2 + width_list[1] + width_list[2], lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[-1])
            else:
                raise ValueError("Incorrect region type")
        elif sidewalk_type == 'Ribbon Sidewalk':
            if region == 'nearroad_sidewalk':
                return (lane.width_at(0) / 2, lane.width_at(0) / 2 + width_list[1])
            elif region == 'main_sidewalk':
                return (lane.width_at(0) / 2 + width_list[1], lane.width_at(0) / 2 + width_list[1] + width_list[2])
            elif region == 'farfromroad_sidewalk':
                return (lane.width_at(0) / 2 + width_list[1] + width_list[2], lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[4])
            elif region == 'valid_region':
                return (lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[4], lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[4] + width_list[-1])
            else:
                raise ValueError("Incorrect region type")
        elif sidewalk_type == 'Neighborhood 1':
            if region == 'nearroad_buffer_sidewalk':
                return (lane.width_at(0) / 2, lane.width_at(0) / 2 + width_list[0])
            elif region == 'nearroad_sidewalk':
                return (lane.width_at(0) / 2 + width_list[0], lane.width_at(0) / 2 + width_list[0] + width_list[1])
            elif region == 'main_sidewalk':
                return (lane.width_at(0) / 2 + width_list[0] + width_list[1], lane.width_at(0) / 2 + width_list[0] + width_list[1] + width_list[2])
            elif region == 'valid_region':
                return (lane.width_at(0) / 2 + width_list[0] + width_list[1] + width_list[2], lane.width_at(0) / 2 + width_list[0] + width_list[1] + width_list[2] + width_list[-1])
            else:
                raise ValueError("Incorrect region type")
        elif sidewalk_type == 'Neighborhood 2':
            if region == 'nearroad_sidewalk':
                return (lane.width_at(0) / 2, lane.width_at(0) / 2 + width_list[1])
            elif region == 'main_sidewalk':
                return (lane.width_at(0) / 2 + width_list[1], lane.width_at(0) / 2 + width_list[1] + width_list[2])
            elif region == 'farfromroad_sidewalk':
                return (lane.width_at(0) / 2 + width_list[1] + width_list[2], lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[4])
            elif region == 'valid_region':
                return (lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[4], lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[4] + width_list[-1])
            else:
                raise ValueError("Incorrect region type")
        elif sidewalk_type == 'Medium Commercial':
            if region == 'nearroad_sidewalk':
                return (lane.width_at(0) / 2, lane.width_at(0) / 2 + width_list[1])
            elif region == 'main_sidewalk':
                return (lane.width_at(0) / 2 + width_list[1], lane.width_at(0) / 2 + width_list[1] + width_list[2])
            elif region == 'farfromroad_sidewalk':
                return (lane.width_at(0) / 2 + width_list[1] + width_list[2], lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[4])
            elif region == 'valid_region':
                return (lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[4], lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[4] + width_list[-1])
            else:
                raise ValueError("Incorrect region type")
        elif sidewalk_type == 'Wide Commercial':
            if region == 'nearroad_sidewalk':
                return (lane.width_at(0) / 2, lane.width_at(0) / 2 + width_list[1])
            elif region == 'main_sidewalk':
                return (lane.width_at(0) / 2 + width_list[1], lane.width_at(0) / 2 + width_list[1] + width_list[2])
            elif region == 'farfromroad_buffer_sidewalk':
                return (lane.width_at(0) / 2 + width_list[1] + width_list[2], lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[3])
            elif region == 'farfromroad_sidewalk':
                return (lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[3], lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[3] + width_list[4])
            elif region == 'valid_region':
                return (lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[3] + width_list[4], lane.width_at(0) / 2 + width_list[1] + width_list[2] + width_list[3] + width_list[4] + width_list[-1])
            else:
                raise ValueError("Incorrect region type")
        else:
            raise NotImplementedError
        
    @property
    def current_map(self) -> object:
        return self.engine.map_manager.current_map
    
    def walkable_region_for_roundabout(self, current_map):
        self.crosswalks = current_map.crosswalks
        self.sidewalks = current_map.sidewalks
        self.sidewalks_near_road = current_map.sidewalks_near_road
        self.sidewalks_farfrom_road = current_map.sidewalks_farfrom_road
        self.sidewalks_near_road_buffer = current_map.sidewalks_near_road_buffer
        self.sidewalks_farfrom_road_buffer = current_map.sidewalks_farfrom_road_buffer
        self.valid_region = current_map.valid_region

        polygons = []
        for sidewalk in self.sidewalks.keys():
            # if "SDW_I_" in sidewalk: continue
            polygon = self.sidewalks[sidewalk]['polygon']
            polygons += polygon
        for crosswalk in self.crosswalks.keys():
            # if "CRS_I_" in crosswalk: continue
            polygon = self.crosswalks[crosswalk]['polygon']
            polygons += polygon
            
        for sidewalk in self.sidewalks_near_road_buffer.keys():
            polygon = self.sidewalks_near_road_buffer[sidewalk]['polygon']
            polygons += polygon
        for sidewalk in self.sidewalks_near_road.keys():
            polygon = self.sidewalks_near_road[sidewalk]['polygon']
            polygons += polygon
        for sidewalk in self.sidewalks_farfrom_road.keys():
            polygon = self.sidewalks_farfrom_road[sidewalk]['polygon']
            polygons += polygon
        for sidewalk in self.sidewalks_farfrom_road_buffer.keys():
            polygon = self.sidewalks_farfrom_road_buffer[sidewalk]['polygon']
            polygons += polygon
        # for sidewalk in self.valid_region.keys():
        #     polygon = self.valid_region[sidewalk]['polygon']
        #     polygons += polygon
            
        # for polygon in self.all_object_polygons:
        #     polygons += polygon

        polygon_array = np.array(polygons)
        min_x = np.min(polygon_array[:, 0])
        max_x = np.max(polygon_array[:, 0])
        min_y = np.min(polygon_array[:, 1])
        max_y = np.max(polygon_array[:, 1])
        self.mask_delta = 2
        rows = math.ceil(max_y - min_y) + 2*self.mask_delta
        columns = math.ceil(max_x - min_x) + 2*self.mask_delta

        self.mask_translate = np.array([-min_x+self.mask_delta, -min_y+self.mask_delta])
        walkable_regions_mask = np.zeros((rows, columns, 3), np.uint8)
        from shapely.geometry import Polygon
        all_area = 0
        for sidewalk in self.sidewalks.keys():
            # if "SDW_I_" in sidewalk: continue
            polygon_array = np.array(self.sidewalks[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            area = Polygon(polygon).area
            all_area += area
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])

        for crosswalk in self.crosswalks.keys():
            # if "SDW_I_" in sidewalk: continue
            polygon_array = np.array(self.crosswalks[crosswalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            area = Polygon(polygon).area
            all_area += area
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
            
        for sidewalk in self.sidewalks_near_road_buffer.keys():
            polygon_array = np.array(self.sidewalks_near_road_buffer[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            area = Polygon(polygon).area
            all_area += area
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
        for sidewalk in self.sidewalks_near_road.keys():
            polygon_array = np.array(self.sidewalks_near_road[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            area = Polygon(polygon).area
            all_area += area
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
        for sidewalk in self.sidewalks_farfrom_road.keys():
            polygon_array = np.array(self.sidewalks_farfrom_road[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            area = Polygon(polygon).area
            all_area += area
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
        for sidewalk in self.sidewalks_farfrom_road_buffer.keys():
            polygon_array = np.array(self.sidewalks_farfrom_road_buffer[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            area = Polygon(polygon).area
            all_area += area
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
            
        return walkable_regions_mask
    
    def _get_walkable_regions(self, current_map):
        self.crosswalks = current_map.crosswalks
        self.sidewalks = current_map.sidewalks
        self.sidewalks_near_road = current_map.sidewalks_near_road
        self.sidewalks_farfrom_road = current_map.sidewalks_farfrom_road
        self.sidewalks_near_road_buffer = current_map.sidewalks_near_road_buffer
        self.sidewalks_farfrom_road_buffer = current_map.sidewalks_farfrom_road_buffer
        self.valid_region = current_map.valid_region

        polygons = []
        for sidewalk in self.sidewalks.keys():
            # if "SDW_I_" in sidewalk: continue
            polygon = self.sidewalks[sidewalk]['polygon']
            polygons += polygon
        for crosswalk in self.crosswalks.keys():
            # if "CRS_I_" in crosswalk: continue
            polygon = self.crosswalks[crosswalk]['polygon']
            polygons += polygon
            
        for sidewalk in self.sidewalks_near_road_buffer.keys():
            polygon = self.sidewalks_near_road_buffer[sidewalk]['polygon']
            polygons += polygon
        for sidewalk in self.sidewalks_near_road.keys():
            polygon = self.sidewalks_near_road[sidewalk]['polygon']
            polygons += polygon
        for sidewalk in self.sidewalks_farfrom_road.keys():
            polygon = self.sidewalks_farfrom_road[sidewalk]['polygon']
            polygons += polygon
        for sidewalk in self.sidewalks_farfrom_road_buffer.keys():
            polygon = self.sidewalks_farfrom_road_buffer[sidewalk]['polygon']
            polygons += polygon
        # for sidewalk in self.valid_region.keys():
        #     polygon = self.valid_region[sidewalk]['polygon']
        #     polygons += polygon
            
        # for polygon in self.all_object_polygons:
        #     polygons += polygon

        polygon_array = np.array(polygons)
        min_x = np.min(polygon_array[:, 0])
        max_x = np.max(polygon_array[:, 0])
        min_y = np.min(polygon_array[:, 1])
        max_y = np.max(polygon_array[:, 1])
        self.mask_delta = 2
        rows = math.ceil(max_y - min_y) + 2*self.mask_delta
        columns = math.ceil(max_x - min_x) + 2*self.mask_delta

        self.mask_translate = np.array([-min_x+self.mask_delta, -min_y+self.mask_delta])
        walkable_regions_mask = np.zeros((rows, columns, 3), np.uint8)
        from shapely.geometry import Polygon
        all_area = 0
        for sidewalk in self.sidewalks.keys():
            # if "SDW_I_" in sidewalk: continue
            polygon_array = np.array(self.sidewalks[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            area = Polygon(polygon).area
            all_area += area
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])

        for crosswalk in self.crosswalks.keys():
            # if "SDW_I_" in sidewalk: continue
            polygon_array = np.array(self.crosswalks[crosswalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            area = Polygon(polygon).area
            all_area += area
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
            
        for sidewalk in self.sidewalks_near_road_buffer.keys():
            polygon_array = np.array(self.sidewalks_near_road_buffer[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            area = Polygon(polygon).area
            all_area += area
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
        for sidewalk in self.sidewalks_near_road.keys():
            polygon_array = np.array(self.sidewalks_near_road[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            area = Polygon(polygon).area
            all_area += area
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
        for sidewalk in self.sidewalks_farfrom_road.keys():
            polygon_array = np.array(self.sidewalks_farfrom_road[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            area = Polygon(polygon).area
            all_area += area
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
        for sidewalk in self.sidewalks_farfrom_road_buffer.keys():
            polygon_array = np.array(self.sidewalks_farfrom_road_buffer[sidewalk]['polygon'])
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            area = Polygon(polygon).area
            all_area += area
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
        # cv2.imwrite('1111.png', walkable_regions_mask)
        # for sidewalk in self.valid_region.keys():
        #     polygon_array = np.array(self.valid_region[sidewalk]['polygon'])
        #     polygon_array += self.mask_translate
        #     polygon_array = np.floor(polygon_array).astype(int)
        #     polygon_array = polygon_array.reshape((-1, 1, 2))
        #     cv2.fillPoly(walkable_regions_mask, [polygon_array], [255, 255, 255])
        
        obj_area_total = 0
        for polygon in self.all_object_polygons:
            polygon_array = np.array(polygon)
            polygon_array += self.mask_translate
            polygon_array = np.floor(polygon_array).astype(int)
            polygon_array = polygon_array.reshape((-1, 1, 2))
            obj_area = Polygon(polygon).area
            obj_area_total += obj_area
            cv2.fillPoly(walkable_regions_mask, [polygon_array], [0, 0, 0])
        self.engine.area = all_area
        self.engine.obj_area = obj_area_total
        walkable_regions_mask = cv2.flip(walkable_regions_mask, 0)   ### flip for orca   ###### 
        # cv2.imwrite('1111.png', walkable_regions_mask)
        # import sys
        # sys.exit(0)
        self.engine.walkable_regions_mask = walkable_regions_mask
        self.engine.mask_translate = self.mask_translate

        return walkable_regions_mask
