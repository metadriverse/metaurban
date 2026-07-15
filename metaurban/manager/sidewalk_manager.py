# Manager that spawns static objects (trees, benches, buildings, ...) on the
# sidewalk regions of every road block.
#
# A block's cross-section is decomposed into parallel strips ("regions")
# stacked outward from the road edge: near-road buffer / near-road sidewalk /
# main sidewalk / far-from-road (buffer) sidewalk / valid (house) region.
# Each region gets a 1 m occupancy grid along the lane; a declarative catalog
# routes object types to regions; placed grid cells are converted back to lane
# coordinates for spawning.
import math
import os
import random
from collections import defaultdict

import cv2
import numpy as np

from metaurban.component.lane.circular_lane import CircularLane
from metaurban.component.lane.straight_lane import StraightLane
from metaurban.component.pgblock.first_block import FirstPGBlock
from metaurban.component.static_object.test_new_object import TestObject
from metaurban.engine.engine_utils import get_engine
from metaurban.manager.base_manager import BaseManager
from metaurban.manager.read_config import configReader

# Grid cells are 1 m x 1 m; all grid<->lane conversions assume this.
CELL_SIZE = 1

# Cross-section layout per sidewalk type: ordered (region, width index) bands
# stacked outward from the road edge. The width index selects from
# [near_road_buffer, near_road, main, far_from_buffer, far_from, valid_house].
_SIDEWALK_BANDS = {
    'Narrow Sidewalk': [('nearroad_buffer_sidewalk', 0), ('main_sidewalk', 2), ('valid_region', 5)],
    'Narrow Sidewalk with Trees': [('nearroad_sidewalk', 1), ('main_sidewalk', 2), ('valid_region', 5)],
    'Ribbon Sidewalk': [('nearroad_sidewalk', 1), ('main_sidewalk', 2), ('farfromroad_sidewalk', 4),
                        ('valid_region', 5)],
    'Neighborhood 1': [('nearroad_buffer_sidewalk', 0), ('nearroad_sidewalk', 1), ('main_sidewalk', 2),
                       ('valid_region', 5)],
    'Neighborhood 2': [('nearroad_sidewalk', 1), ('main_sidewalk', 2), ('farfromroad_sidewalk', 4),
                       ('valid_region', 5)],
    'Medium Commercial': [('nearroad_sidewalk', 1), ('main_sidewalk', 2), ('farfromroad_sidewalk', 4),
                          ('valid_region', 5)],
    'Wide Commercial': [('nearroad_sidewalk', 1), ('main_sidewalk', 2), ('farfromroad_buffer_sidewalk', 3),
                        ('farfromroad_sidewalk', 4), ('valid_region', 5)],
}

# Region processing order (grid creation, placement and detach). Kept separate
# from the band order because it historically differs from it for
# 'Neighborhood 1' and 'Wide Commercial'.
_REGION_ORDER = {
    'Narrow Sidewalk': ['nearroad_buffer_sidewalk', 'main_sidewalk', 'valid_region'],
    'Narrow Sidewalk with Trees': ['nearroad_sidewalk', 'main_sidewalk', 'valid_region'],
    'Ribbon Sidewalk': ['nearroad_sidewalk', 'main_sidewalk', 'farfromroad_sidewalk', 'valid_region'],
    'Neighborhood 1': ['nearroad_sidewalk', 'main_sidewalk', 'nearroad_buffer_sidewalk', 'valid_region'],
    'Neighborhood 2': ['nearroad_sidewalk', 'main_sidewalk', 'farfromroad_sidewalk', 'valid_region'],
    'Medium Commercial': ['nearroad_sidewalk', 'main_sidewalk', 'farfromroad_sidewalk', 'valid_region'],
    'Wide Commercial': ['nearroad_sidewalk', 'main_sidewalk', 'farfromroad_sidewalk',
                        'farfromroad_buffer_sidewalk', 'valid_region'],
}


class ObjectPlacer:
    """Places objects on a boolean occupancy grid (rows of 1 m cells) without overlap.

    ``buffer`` (cells of clearance added to each object's span) is set by the
    manager per region before placement.
    """
    def __init__(self, grid):
        self.grid = grid
        self.buffer = 0
        # (CLASS_NAME, position) -> (position, metainfo) in placement order
        self.placed_objects = {}

    def place_object(self, obj, last_long):
        """Try to place ``obj`` at least ``spawn_long_gap`` rows past ``last_long``.

        Returns (placed, last_longitudinal_row).
        """
        position = self.find_placement_position(obj, last_long)
        if position is None:
            return False, last_long
        self.mark_occupied_cells(position, obj)
        self.placed_objects[(obj['CLASS_NAME'], position)] = (position, obj)
        return True, position[0]

    def find_placement_position(self, obj, last_long):
        """First grid position where ``obj`` fits, or None.

        The scan pattern depends on ``obj['obj_generation_mode']``:
        parallel_only scans the first lateral column, random_start starts from
        a random cell, inverse scans backwards, normal (or no mode) scans
        everything front-to-back.
        """
        mode = obj.get('obj_generation_mode')
        n_long, n_lat = len(self.grid), len(self.grid[0]) if self.grid else 0
        start_long = last_long + obj['spawn_long_gap']

        if mode == 'parallel_only':
            for i in range(start_long, n_long):
                if self.can_place(i + 1, 1, obj):
                    return (i + 1, 1)
            return None

        if mode == 'random_start':
            if start_long >= n_long - 5:
                return None
            # this draw always yields start_long; kept to preserve the RNG stream
            start_long = np.random.randint(start_long, start_long + 1, 1)[0]
            start_lat = np.random.randint(0, max(n_lat - 10, 1), 1)[0]
            for i in range(start_long, n_long):
                for j in range(start_lat, n_lat):
                    if self.can_place(i + 1, j + 1, obj):
                        return (i + 1, j + 1)
            return None

        if mode == 'inverse':
            for i in range(n_long - 1, start_long, -1):
                for j in range(n_lat - 1, 0, -1):
                    if self.can_place(i + 1, j + 1, obj):
                        return (i + 1, j + 1)
            return None

        # 'normal' or no mode
        for i in range(start_long, n_long):
            for j in range(len(self.grid[i])):
                if self.can_place(i + 1, j + 1, obj):
                    return (i + 1, j + 1)
        return None

    def _span(self, obj):
        """Cell footprint of ``obj`` including the clearance buffer."""
        return (
            math.ceil(obj['general']['length'] / CELL_SIZE) + self.buffer,
            math.ceil(obj['general']['width'] / CELL_SIZE) + self.buffer,
        )

    def can_place(self, start_i, start_j, obj):
        span_length, span_width = self._span(obj)
        if start_i + span_length > len(self.grid) or start_j + span_width > len(self.grid[0]):
            return False
        return not any(
            self.grid[i][j] for i in range(start_i, start_i + span_length)
            for j in range(start_j, start_j + span_width)
        )

    def mark_occupied_cells(self, start_position, obj):
        start_i, start_j = start_position
        span_length, span_width = self._span(obj)
        for i in range(start_i, start_i + span_length):
            for j in range(start_j, start_j + span_width):
                self.grid[i][j] = True


class AssetManager(BaseManager):
    """Spawns static objects on the sidewalk regions of every map block.

    The main entry point is :meth:`reset`, called at the beginning of each
    episode: for every block it collects the lanes to decorate, builds one
    occupancy grid per sidewalk region, places the object catalogs onto the
    grids, and finally spawns each placed object in the world.
    """
    PRIORITY = 9

    # {detail_type: (regions, generation mode)}; insertion order is the
    # placement priority.
    REGULAR_OBJECTS = {
        'Tree': (('nearroad_buffer_sidewalk', 'nearroad_sidewalk'), 'parallel_only'),
        'Lamp_post': (('nearroad_buffer_sidewalk', 'nearroad_sidewalk'), 'parallel_only'),
        'TrashCan': (('nearroad_buffer_sidewalk', 'nearroad_sidewalk'), 'parallel_only'),
        'Mailbox': (('main_sidewalk', ), 'random_start'),
        'Telephone_booth': (('main_sidewalk', ), 'parallel_only'),
        'FireHydrant': (('nearroad_buffer_sidewalk', 'nearroad_sidewalk'), 'parallel_only'),
        'Building': (('valid_region', ), 'normal'),
        'Chair': (('farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'), 'parallel_only'),
        'Vegetation': (('farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'), 'normal'),
        'Advertising_board': (('farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'), 'parallel_only'),
        'Bench': (('farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'), 'parallel_only'),
        'Traffic_sign': (('nearroad_buffer_sidewalk', 'nearroad_sidewalk'), 'parallel_only'),
        'Bollard': (('nearroad_buffer_sidewalk', 'nearroad_sidewalk'), 'parallel_only'),
        'dog': (('main_sidewalk', ), 'random_start'),
        'Vending_machine': (('main_sidewalk', ), 'random_start'),
        'Bag': (('main_sidewalk', ), 'random_start'),
        'Table': (('main_sidewalk', ), 'random_start'),
        'Bonsai': (('farfromroad_sidewalk', 'farfromroad_buffer_sidewalk'), 'random_start'),
        'Cone': (('main_sidewalk', ), 'random_start'),
        'FoodTruck': (('valid_region', ), 'random_start'),
        'Bike': (('valid_region', ), 'random_start'),
        'Motorcycle': (('valid_region', ), 'random_start'),
        'Scooter': (('valid_region', ), 'random_start'),
        'Wheelchair': (('valid_region', ), 'random_start'),
    }
    # Fills the leftover valid (house) region after the regular pass.
    PADDING_OBJECTS = {
        'Tree': (('valid_region', ), 'normal'),
    }
    # Placed only on intersection blocks; the mode is chosen per lane.
    INTERSECTION_OBJECTS = {
        'Traffic_light': (('main_sidewalk', ), None),
    }

    def __init__(self):
        super(AssetManager, self).__init__()
        self.density = self.engine.global_config['object_density']

        self.config = configReader()
        self.path_config = self.config.loadPath()
        self.init_static_adj_list()  # Load the metainfo for all static objects
        self.get_attr()  # Get the spawn policy for each object type

        self.all_object_polygons = []

    def init_static_adj_list(self):
        """
        Load the metainfo for all static objects
        """
        # The dictionary to store the metainfo for each object type
        # The key is the detail type, the value is a list of metainfo dictionaries
        # For example, key is bicycle, value is a list of metainfo dictionaries for all bicycle objects
        # Metainfo is derived from the GLBs themselves (cached) plus curated
        # semantics, instead of the per-asset adj_parameter_folder JSONs.
        from metaurban.asset_metainfo import load_asset_metainfo
        self.config.getReverseType()
        self.type_metainfo_dict = defaultdict(list)
        skipped_types = set()
        for metainfo in load_asset_metainfo(self.path_config["metaurbanasset"]):
            detail_type = metainfo['general']['detail_type']
            # Types absent from asset_config.yaml have no spawn policy and would
            # crash get_attr; skip them so new GLBs can be dropped in freely.
            if detail_type not in self.config.reverseType:
                skipped_types.add(detail_type)
                continue
            self.type_metainfo_dict[detail_type].append(metainfo)
        if skipped_types:
            print(f"[asset_metainfo] no spawn config for types {sorted(skipped_types)}, not spawning them")

    def get_attr(self):
        """Per-type spawn policy from asset_config.yaml, scaled by density."""
        self.num_dict = {}
        self.interval_long = {}
        self.random_gap = {}
        for detail_type in self.type_metainfo_dict.keys():
            self.num_dict[detail_type] = max(int(self.config.getSpawnNum(detail_type) * self.density), 1)
            self.interval_long[detail_type] = max(
                min(int(self.config.getSpawnInterval(detail_type) * 1 / self.density), 40), 1
            )
            self.random_gap[detail_type] = self.config.getrandom_gap(detail_type)

    @staticmethod
    def _seed_everything(seed):
        import torch
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def reset(self):
        """
        Reset the manager and spawn objects on the sidewalk.
        Main entry point for the manager.
        """
        super(AssetManager, self).reset()
        self._seed_everything(self.engine.global_seed)

        self.generated_lane = []
        self.all_object_polygons = []
        engine = get_engine()
        assert len(self.spawned_objects.keys()) == 0

        for block in engine.current_map.blocks:
            if isinstance(block, FirstPGBlock):
                continue
            if block.ID == 'S':
                self.block_type = 'S'
                for lane in [block.positive_basic_lane, block.negative_basic_lane]:
                    self._populate_lane(block, lane)
            elif block.ID == 'O':
                walkable_map = self.walkable_region_for_roundabout(self.current_map)
                self.block_type = 'O'
                for lane in self._roundabout_lanes(block):
                    # objects outside the roundabout's walkable ring are dropped
                    self._populate_lane(block, lane, walkable_map=walkable_map)
            elif block.ID in ('X', 'T'):
                self.block_type = 'X'
                lanes, neg_socket_lanes = self._intersection_lanes(block)
                for lane in lanes:
                    entry_lane = lane == block.positive_basic_lane
                    self._populate_lane(
                        block,
                        lane,
                        delta_scale=1.5 if isinstance(lane, CircularLane) else None,
                        reserve_head=entry_lane,
                        intersection_mode='inverse' if entry_lane or lane in neg_socket_lanes else 'normal',
                    )
            elif block.ID == 'C':
                self.block_type = 'C'
                for lane in self._curve_lanes(block):
                    self._populate_lane(block, lane, delta_scale=2. if isinstance(lane, CircularLane) else None)

        self._get_walkable_regions(self.current_map)

    # --- lane collection per block type ---------------------------------

    @staticmethod
    def _socket_lanes(block):
        pos_lanes, neg_lanes = [], []
        for socket in block._sockets.values():
            pos_lanes.append(socket.get_positive_lanes(block._global_network)[-1])
            neg_lanes.append(socket.get_negative_lanes(block._global_network)[-1])
        return pos_lanes, neg_lanes

    @staticmethod
    def _adjacent(lane_a, lane_b):
        return lane_a.is_previous_lane_of(lane_b) or lane_b.is_previous_lane_of(lane_a)

    def _adjacent_lanes(self, block, circular_pairs_only):
        """Two scans of the block graph: circular lanes touching a basic lane,
        then lanes touching any lane collected so far. With
        ``circular_pairs_only`` the second scan skips pairs involving a
        straight lane. Returns (basic lanes, collected lanes, with repeats)."""
        basics = [block.positive_basic_lane, block.negative_basic_lane]
        graph = block.block_network.graph
        valid_lane = []
        for to_dict in graph.values():
            for lanes in to_dict.values():
                for lane in lanes:
                    if isinstance(lane, CircularLane):
                        for basic in basics:
                            if self._adjacent(basic, lane):
                                valid_lane.append(lane)
        for to_dict in graph.values():
            for lanes in to_dict.values():
                for lane in lanes:
                    for lane_ in basics + valid_lane:
                        if self._adjacent(lane_, lane):
                            if not circular_pairs_only or (not isinstance(lane, StraightLane)
                                                           and not isinstance(lane_, StraightLane)):
                                valid_lane.append(lane)
        return basics, valid_lane

    def _roundabout_lanes(self, block):
        basics, valid_lane = self._adjacent_lanes(block, circular_pairs_only=True)
        valid_lane = [
            lane for lane in set(valid_lane) if lane not in basics and 'ROAD_EDGE_BOUNDARY' in lane.line_types
        ]
        pos_lanes, neg_lanes = self._socket_lanes(block)
        return basics + pos_lanes + neg_lanes + valid_lane

    def _intersection_lanes(self, block):
        basics = [block.positive_basic_lane, block.negative_basic_lane]
        pos_lanes, neg_lanes = self._socket_lanes(block)
        valid_lane = [lane for lane in set(block.right_lanes) if lane not in basics + pos_lanes + neg_lanes]
        return basics + pos_lanes + neg_lanes + valid_lane, neg_lanes

    def _curve_lanes(self, block):
        basics, valid_lane = self._adjacent_lanes(block, circular_pairs_only=False)
        return basics + [lane for lane in set(valid_lane) if lane not in basics]

    # --- per-lane population ---------------------------------------------

    def _populate_lane(self, block, lane, delta_scale=None, walkable_map=None, reserve_head=False,
                       intersection_mode=None):
        """Build the region grids of ``lane``, place the catalogs, spawn the result."""
        if lane in self.generated_lane:
            return
        self.generated_lane.append(lane)

        self.sidewalk_type = block.sidewalk_type
        if self.sidewalk_type not in _REGION_ORDER:
            raise NotImplementedError(self.sidewalk_type)
        width_list = [
            block.near_road_buffer_width, block.near_road_width, block.main_width, block.far_from_buffer_width,
            block.far_from_width, block.valid_house_width
        ]

        name_grid_list = [
            (region, self.create_grid(lane, self.calculate_lateral_range(region, lane, width_list,
                                                                         self.sidewalk_type)))
            for region in _REGION_ORDER[self.sidewalk_type]
        ]
        if reserve_head:
            # keep the first meters after the intersection entrance clear
            for _, grid in name_grid_list:
                for row in grid[:10]:
                    row[:] = [True] * len(row)
        placers = {region: ObjectPlacer(grid) for region, grid in name_grid_list}

        self._place_catalog(self.REGULAR_OBJECTS, name_grid_list, placers, delta_scale)
        self._place_catalog(self.PADDING_OBJECTS, name_grid_list, placers, delta_scale)
        if intersection_mode is not None:
            catalog = {t: (regions, intersection_mode) for t, (regions, _) in self.INTERSECTION_OBJECTS.items()}
            self._place_catalog(catalog, name_grid_list, placers, delta_scale)

        self._detach_to_world(lane, name_grid_list, placers, width_list, walkable_map)

    def _place_catalog(self, catalog, name_grid_list, placers, delta_scale=None):
        for detail_type, (regions, mode) in catalog.items():
            for region, _ in name_grid_list:
                if region in regions:
                    self.retrieve_target_object_for_region(region, placers[region], detail_type, mode, delta_scale)

    def retrieve_target_object_for_region(
        self, region, object_placer, obj_detail_type, obj_generation_mode=None, delta_scale=None
    ):
        """Place up to the configured number of ``obj_detail_type`` assets on the region grid."""
        self._seed_everything(self.engine.global_seed + 21931)

        objects = self.type_metainfo_dict.get(obj_detail_type, [])
        if obj_detail_type.lower() == 'tree':
            # trees are placed on a fixed 2 x 2 m footprint regardless of the mesh
            for obj in objects:
                obj['general']['width'] = 2.
                obj['general']['length'] = 2.

        self.buffer = 0 if 'near' in region else 2
        if obj_detail_type.lower() == 'building':
            self.buffer = 10
        if not objects:
            return

        placed, last_long = 0, 0
        while placed < self.num_dict[obj_detail_type]:
            obj = objects[random.sample(range(len(objects)), 1)[0]]
            interval_long = self.interval_long[obj_detail_type]
            if self.random_gap[obj_detail_type]:
                offset = np.random.randint(0, 5, 1)[0]
            else:
                offset = 0
            if placed < 1:
                offset = 10  # push the first instance away from the lane start
            obj['spawn_long_gap'] = interval_long
            if delta_scale is not None and region == 'valid_region':
                obj['spawn_long_gap'] = int(interval_long * delta_scale)
            if obj_detail_type.lower() == 'tree' and region == 'valid_region':
                obj['spawn_long_gap'] = int(2 * 1 / self.density)
                offset = 0
            object_placer.buffer = self.buffer
            obj['obj_generation_mode'] = obj_generation_mode

            generated, last_long = object_placer.place_object(obj, last_long + offset)
            if not generated:
                break
            placed += 1

    # --- world spawning ----------------------------------------------------

    def _detach_to_world(self, lane, name_grid_list, placers, width_list, walkable_map=None):
        """Spawn every placed object; with ``walkable_map``, drop objects outside it."""
        for region, _ in name_grid_list:
            lat_range = self.calculate_lateral_range(region, lane, width_list, self.sidewalk_type)
            # valid-region objects anchor at their footprint center, sidewalk ones at the edge
            coeff = 1 if 'region' in region else 0
            # self.buffer is always 2 here — the padding/intersection pass ran last
            # and set it — NOT each object's own placement buffer; anchors and
            # polygons intentionally keep that historical quirk.
            for grid_position, obj in placers[region].placed_objects.values():
                span_length = math.ceil(obj['general']['length']) + self.buffer
                span_width = math.ceil(obj['general']['width']) + self.buffer
                polygon = self._object_polygon(grid_position, span_length, span_width, obj, lane, lat_range)
                if walkable_map is not None and not self._overlaps_walkable(polygon, walkable_map):
                    continue
                lane_position = self.convert_grid_to_lane_position(
                    [grid_position[0], grid_position[1] + span_width // 2 * coeff], lane, lat_range
                )
                self.spawn_object(
                    TestObject,
                    force_spawn=True,
                    lane=lane,
                    position=lane_position,
                    static=self.engine.global_config["static_traffic_object"],
                    heading_theta=lane.heading_theta_at(lane_position[0]),
                    asset_metainfo=obj
                )
                self.all_object_polygons.append(polygon)

    def _object_polygon(self, grid_position, span_length, span_width, obj, lane, lat_range):
        """World-frame footprint polygon of an object placed at ``grid_position``."""
        start_lat = self.convert_grid_to_longitudelateral(grid_position, lat_range)[1]
        side_lat = self.convert_grid_to_longitudelateral(
            (grid_position[0] + span_length, grid_position[1] + span_width), lat_range
        )[1]
        mid_j = grid_position[1] + math.ceil(obj['general']['width']) // 2
        longs = [
            self.convert_grid_to_longitudelateral((grid_position[0] + i, mid_j), lat_range)[0]
            for i in range(span_length)
        ]
        polygon = []
        for lateral, row in ((start_lat, longs), (side_lat, longs[::-1])):
            for longitude in row:
                point = lane.position(min(lane.length + 0.1, longitude), lateral)
                polygon.append([point[0], point[1]])
        return polygon

    def _overlaps_walkable(self, polygon, walkable_map):
        """True if the polygon covers at least one walkable pixel of the map."""
        arr = np.floor(np.array(polygon) + self.mask_translate).astype(int).reshape((-1, 1, 2))
        masked = walkable_map.copy()
        cv2.fillPoly(masked, [arr], [0, 0, 0])
        return ((masked - walkable_map)**2).sum() != 0.

    # --- grid geometry -----------------------------------------------------

    def create_grid(self, lane, lateral_range):
        """1 m boolean occupancy grid covering ``lane`` over ``lateral_range``."""
        from metaurban.constants import PGDrivableAreaProperty
        if self.block_type == 'X':
            num_cells_long = int(lane.length / CELL_SIZE)
        else:
            num_cells_long = int((lane.length + PGDrivableAreaProperty.SIDEWALK_LENGTH) / CELL_SIZE)
        num_cells_lat = int((lateral_range[1] - lateral_range[0]) / CELL_SIZE)
        return [[False] * num_cells_lat for _ in range(num_cells_long)]

    def convert_grid_to_longitudelateral(self, grid_position, lateral_range):
        grid_i, grid_j = grid_position
        return (grid_i * CELL_SIZE, lateral_range[0] + grid_j * CELL_SIZE)

    def convert_grid_to_lane_position(self, grid_position, lane, lateral_range):
        return lane.position(*self.convert_grid_to_longitudelateral(grid_position, lateral_range))

    def calculate_lateral_range(self, region, lane, width_list, sidewalk_type):
        """Lateral (start, end) of ``region``: cumulative band widths from the road edge."""
        if sidewalk_type not in _SIDEWALK_BANDS:
            raise NotImplementedError(sidewalk_type)
        start = lane.width_at(0) / 2
        for band_region, width_index in _SIDEWALK_BANDS[sidewalk_type]:
            width = width_list[width_index]
            assert width is not None
            if band_region == region:
                return (start, start + width)
            start += width
        raise ValueError("Incorrect region type")

    @property
    def current_map(self) -> object:
        return self.engine.map_manager.current_map

    # --- walkable region masks ----------------------------------------------

    def walkable_region_for_roundabout(self, current_map):
        return self._build_walkable_mask(current_map, with_valid_region_and_objects=False)

    def _get_walkable_regions(self, current_map):
        mask = self._build_walkable_mask(current_map, with_valid_region_and_objects=True)
        self.engine.walkable_regions_mask = mask
        self.engine.mask_translate = self.mask_translate
        return mask

    def _build_walkable_mask(self, current_map, with_valid_region_and_objects):
        """Rasterize the walkable regions to a mask; optionally also include the
        valid (house) region and carve out the spawned objects' footprints."""
        groups = [
            current_map.sidewalks, current_map.crosswalks, current_map.sidewalks_near_road_buffer,
            current_map.sidewalks_near_road, current_map.sidewalks_farfrom_road,
            current_map.sidewalks_farfrom_road_buffer
        ]
        if with_valid_region_and_objects:
            groups.append(current_map.valid_region)

        points = []
        for group in groups:
            for item in group.values():
                points += item['polygon']
        if with_valid_region_and_objects:
            for polygon in self.all_object_polygons:
                points += polygon

        points = np.array(points)
        min_x, max_x = points[:, 0].min(), points[:, 0].max()
        min_y, max_y = points[:, 1].min(), points[:, 1].max()
        mask_delta = 2
        rows = math.ceil(max_y - min_y) + 2 * mask_delta
        columns = math.ceil(max_x - min_x) + 2 * mask_delta
        self.mask_translate = np.array([-min_x + mask_delta, -min_y + mask_delta])

        mask = np.zeros((rows, columns, 3), np.uint8)
        for group in groups:
            for item in group.values():
                self._fill_polygon(mask, item['polygon'], [255, 255, 255])
        if with_valid_region_and_objects:
            for polygon in self.all_object_polygons:
                self._fill_polygon(mask, polygon, [0, 0, 0])
        return mask

    def _fill_polygon(self, mask, polygon, color):
        arr = np.floor(np.array(polygon) + self.mask_translate).astype(int).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [arr], color)
