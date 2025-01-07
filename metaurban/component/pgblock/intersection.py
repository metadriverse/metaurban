import copy
from metaurban.type import MetaUrbanType
from collections import deque
import numpy as np

from metaurban.component.lane.straight_lane import StraightLane
from metaurban.component.pgblock.create_pg_block_utils import CreateAdverseRoad, CreateRoadFrom, ExtendStraightLane, \
    create_bend_straight
from metaurban.component.pgblock.pg_block import PGBlock, PGBlockSocket
from metaurban.component.road_network import Road
from metaurban.constants import PGDrivableAreaProperty, PGLineType
from metaurban.utils.pg.utils import check_lane_on_road
from metaurban.component.pg_space import ParameterSpace, Parameter, BlockParameterSpace

import matplotlib.pyplot as plt


class InterSection(PGBlock):
    """
                                up(Goal:1)
                                   ||
                                   ||
                                   ||
                                   ||
                                   ||
                  _________________||_________________
    left(Goal:2)  -----------------||---------------- right(Goal:0)
                               __  ||
                              |    ||
             spawn_road    <--|    ||
                              |    ||
                              |__  ||
                                  down
    It's an Intersection with two lanes on same direction, 4 lanes on both roads
    """

    ID = "X"
    EXTRA_PART = "extra"
    PARAMETER_SPACE = ParameterSpace(BlockParameterSpace.INTERSECTION)
    SOCKET_NUM = 3
    ANGLE = 90  # may support other angle in the future
    EXIT_PART_LENGTH = 35

    _enable_u_turn_flag = False

    # LEFT_TURN_NUM = 1 now it is useless

    def __init__(self, *args, **kwargs):
        if "radius" in kwargs:
            self.radius = kwargs.pop("radius")
        else:
            self.radius = None
        super(InterSection, self).__init__(*args, **kwargs)
        if self.radius is None:
            self.radius = self.get_config(copy=False)[Parameter.radius]

    def _try_plug_into_previous_block(self) -> bool:
        para = self.get_config()
        decrease_increase = -1 if para[Parameter.decrease_increase] == 0 else 1
        # print(para[Parameter.decrease_increase])
        # print(decrease_increase)
        if self.positive_lane_num <= 1:
            decrease_increase = 1
        elif self.positive_lane_num >= 4:
            decrease_increase = -1
        # decrease_increase = 10
        self.lane_num_intersect = self.positive_lane_num + decrease_increase * para[Parameter.change_lane_num]
        no_cross = True
        attach_road = self.pre_block_socket.positive_road
        _attach_road = self.pre_block_socket.negative_road
        attach_lanes = attach_road.get_lanes(self._global_network)
        # current straight left node name, rotate it to fit different part
        intersect_nodes = deque(
            [self.road_node(0, 0),
             self.road_node(1, 0),
             self.road_node(2, 0), _attach_road.start_node]
        )

        # for i in range(3):
        #     self.draw_polygon(attach_lanes[i].polygon)
        #     input("Press Enter to continue...")

        self.right_lanes = []
        for i in range(4):
            right_lane, success = self._create_part(attach_lanes, attach_road, self.radius, intersect_nodes, i)

            # print(right_lane.polygon)
            # self.draw_polygon(right_lane.polygon, i)

            no_cross = no_cross and success
            if i != 3:
                lane_num = self.positive_lane_num if i == 1 else self.lane_num_intersect
                exit_road = Road(self.road_node(i, 0), self.road_node(i, 1))
                no_cross = CreateRoadFrom(
                    right_lane,
                    lane_num,
                    exit_road,
                    self.block_network,
                    self._global_network,
                    ignore_intersection_checking=self.ignore_intersection_checking
                ) and no_cross

                # self.draw_polygons_in_network_block(self.block_network)

                no_cross = CreateAdverseRoad(
                    exit_road,
                    self.block_network,
                    self._global_network,
                    ignore_intersection_checking=self.ignore_intersection_checking
                ) and no_cross

                # self.draw_polygons_in_network_block(self.block_network)

                socket = PGBlockSocket(exit_road, -exit_road)
                self.add_respawn_roads(socket.negative_road)
                self.add_sockets(socket)
                attach_road = -exit_road
                attach_lanes = attach_road.get_lanes(self.block_network)

        # self.draw_polygon(self.block_network.graph['>>>']['1X2_0_'][2].polygon)
        # input("Press Enter to continue...")
        # self.draw_polygon(self.block_network.graph['>>>']['1X1_0_'][2].polygon)
        # input("Press Enter to continue...")
        # self.draw_polygon(self.block_network.graph['>>>']['1X0_0_'][2].polygon)
        # input("Press Enter to continue...")

        # self.draw_polygons_in_network_block(self.block_network)

        return no_cross

    def _create_part(self, attach_lanes, attach_road: Road, radius: float, intersect_nodes: deque,
                     part_idx) -> (StraightLane, bool):
        lane_num = self.lane_num_intersect if part_idx == 0 or part_idx == 2 else self.positive_lane_num
        non_cross = True
        attach_left_lane = attach_lanes[0]
        # first left part
        assert isinstance(attach_left_lane, StraightLane), "Can't create a intersection following a circular lane"
        self._create_left_turn(radius, lane_num, attach_left_lane, attach_road, intersect_nodes, part_idx)

        # u-turn
        if self._enable_u_turn_flag:
            adverse_road = -attach_road
            self._create_u_turn(attach_road, part_idx)

        # go forward part
        lanes_on_road = copy.copy(attach_lanes)
        straight_lane_len = 2 * radius + (2 * lane_num - 1) * lanes_on_road[0].width_at(0)
        for l in lanes_on_road:
            next_lane = ExtendStraightLane(
                l,
                straight_lane_len, (PGLineType.NONE, PGLineType.NONE),
                metaurban_lane_type=MetaUrbanType.LANE_SURFACE_UNSTRUCTURE
            )
            self.block_network.add_lane(attach_road.end_node, intersect_nodes[1], next_lane)

        # right part
        length = self.EXIT_PART_LENGTH
        right_turn_lane = lanes_on_road[-1]
        assert isinstance(right_turn_lane, StraightLane), "Can't create a intersection following a circular lane"
        right_bend, right_straight = create_bend_straight(
            right_turn_lane, length, radius, np.deg2rad(self.ANGLE), True, right_turn_lane.width_at(0),
            (PGLineType.NONE, PGLineType.SIDE)
        )
        self.right_lanes.append(right_bend)
        non_cross = (
            not check_lane_on_road(
                self._global_network, right_bend, 1, ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and non_cross
        CreateRoadFrom(
            right_bend,
            min(self.positive_lane_num, self.lane_num_intersect),
            Road(attach_road.end_node, intersect_nodes[0]),
            self.block_network,
            self._global_network,
            toward_smaller_lane_index=True,
            side_lane_line_type=PGLineType.SIDE,
            inner_lane_line_type=PGLineType.NONE,
            center_line_type=PGLineType.NONE,
            ignore_intersection_checking=self.ignore_intersection_checking,
            metaurban_lane_type=MetaUrbanType.LANE_SURFACE_UNSTRUCTURE
        )

        intersect_nodes.rotate(-1)
        right_straight.line_types = [PGLineType.BROKEN, PGLineType.SIDE]
        return right_straight, non_cross

    def get_socket(self, index: int) -> PGBlockSocket:
        socket = super(InterSection, self).get_socket(index)
        if socket.negative_road in self.get_respawn_roads():
            self._respawn_roads.remove(socket.negative_road)
        return socket

    def _create_left_turn(self, radius, lane_num, attach_left_lane, attach_road, intersect_nodes, part_idx):
        left_turn_radius = radius + lane_num * attach_left_lane.width_at(0)
        diff = self.lane_num_intersect - self.positive_lane_num  # increase lane num
        if ((part_idx == 1 or part_idx == 3) and diff > 0) or ((part_idx == 0 or part_idx == 2) and diff < 0):
            diff = abs(diff)
            left_bend, extra_part = create_bend_straight(
                attach_left_lane, self.lane_width * diff, left_turn_radius, np.deg2rad(self.ANGLE), False,
                attach_left_lane.width_at(0), (PGLineType.NONE, PGLineType.NONE)
            )
            left_road_start = intersect_nodes[2]
            pre_left_road_start = left_road_start + self.EXTRA_PART
            CreateRoadFrom(
                left_bend,
                min(self.positive_lane_num, self.lane_num_intersect),
                Road(attach_road.end_node, pre_left_road_start),
                self.block_network,
                self._global_network,
                toward_smaller_lane_index=False,
                center_line_type=PGLineType.NONE,
                side_lane_line_type=PGLineType.NONE,
                inner_lane_line_type=PGLineType.NONE,
                ignore_intersection_checking=self.ignore_intersection_checking,
                metaurban_lane_type=MetaUrbanType.LANE_SURFACE_UNSTRUCTURE
            )

            CreateRoadFrom(
                extra_part,
                min(self.positive_lane_num, self.lane_num_intersect),
                Road(pre_left_road_start, left_road_start),
                self.block_network,
                self._global_network,
                toward_smaller_lane_index=False,
                center_line_type=PGLineType.NONE,
                side_lane_line_type=PGLineType.NONE,
                inner_lane_line_type=PGLineType.NONE,
                ignore_intersection_checking=self.ignore_intersection_checking
            )

        else:
            left_bend, _ = create_bend_straight(
                attach_left_lane, self.EXIT_PART_LENGTH, left_turn_radius, np.deg2rad(self.ANGLE), False,
                attach_left_lane.width_at(0), (PGLineType.NONE, PGLineType.NONE)
            )
            left_road_start = intersect_nodes[2]
            CreateRoadFrom(
                left_bend,
                min(self.positive_lane_num, self.lane_num_intersect),
                Road(attach_road.end_node, left_road_start),
                self.block_network,
                self._global_network,
                toward_smaller_lane_index=False,
                center_line_type=PGLineType.NONE,
                side_lane_line_type=PGLineType.NONE,
                inner_lane_line_type=PGLineType.NONE,
                ignore_intersection_checking=self.ignore_intersection_checking,
                metaurban_lane_type=MetaUrbanType.LANE_SURFACE_UNSTRUCTURE
            )

    def _create_u_turn(self, attach_road, part_idx):
        # set to CONTINUOUS to debug
        line_type = PGLineType.NONE
        lanes = attach_road.get_lanes(self.block_network) if part_idx != 0 else self.positive_lanes
        attach_left_lane = lanes[0]
        lane_num = len(lanes)
        left_turn_radius = self.lane_width / 2
        left_bend, _ = create_bend_straight(
            attach_left_lane, 0.1, left_turn_radius, np.deg2rad(180), False, attach_left_lane.width_at(0),
            (PGLineType.NONE, PGLineType.NONE)
        )
        left_road_start = (-attach_road).start_node
        CreateRoadFrom(
            left_bend,
            lane_num,
            Road(attach_road.end_node, left_road_start),
            self.block_network,
            self._global_network,
            toward_smaller_lane_index=False,
            center_line_type=line_type,
            side_lane_line_type=line_type,
            inner_lane_line_type=line_type,
            ignore_intersection_checking=self.ignore_intersection_checking,
            metaurban_lane_type=MetaUrbanType.LANE_SURFACE_UNSTRUCTURE
        )

    def enable_u_turn(self, enable_u_turn: bool):
        self._enable_u_turn_flag = enable_u_turn

    def get_intermediate_spawn_lanes(self):
        """Override this function for intersection so that we won't spawn vehicles in the center of intersection."""
        respawn_lanes = self.get_respawn_lanes()
        return respawn_lanes

    def draw_polygon(self, polygon):
        """
        Visualize the polygon with matplot lib
        Args:
            polygon: a list of 2D points

        Returns: None

        """
        import matplotlib.pyplot as plt

        # Create the rectangle
        rectangle_points = np.array(polygon)
        # Extract the points for easier plotting
        x_rect, y_rect = rectangle_points.T

        # Extract the original midpoints

        # Plot the rectangle
        plt.figure(figsize=(8, 8))
        plt.plot(
            *zip(*np.append(rectangle_points, [rectangle_points[0]], axis=0)), marker='o', label='Rectangle Vertices'
        )
        plt.fill(
            *zip(*np.append(rectangle_points, [rectangle_points[0]], axis=0)), alpha=0.3
        )  # Fill the rectangle with light opacity

        # Plot the original midpoints
        # plt.scatter(x_mid, y_mid, color='red', zorder=5, label='Midpoints')

        # Set equal scaling and labels
        plt.axis('equal')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.title('Visualization of the Rectangle and Input Points')
        plt.legend()
        plt.grid(True)

        plt.show()

        # input("Press Enter to continue...")

    def draw_polygons_in_network_block(self, network_block):
        """
        Visualize the polygons  with matplot lib
        Args:
            polygon: a list of 2D points

        Returns: None

        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(100, 100))

        for x in network_block.graph.keys():
            for y in network_block.graph[x].keys():
                for i in range(3):

                    polygon = network_block.graph[x][y][i].polygon

                    # Create the rectangle
                    rectangle_points = np.array(polygon)

                    # Plot the rectangle
                    plt.plot(
                        *zip(*np.append(rectangle_points, [rectangle_points[0]], axis=0)),
                        marker='o',
                        label='[' + x + ']' + '[' + y + ']' + '[' + str(i) + ']',
                        c=np.random.rand(1, 3)
                    )

                    # Fill the rectangle with light opacity
                    plt.fill(
                        *zip(*np.append(rectangle_points, [rectangle_points[0]], axis=0)),
                        alpha=0.3,
                        c=np.random.rand(1, 3)
                    )

        # Set equal scaling and labels
        plt.axis('equal')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.title('Visualization of the Rectangle and Input Points')
        plt.legend()
        plt.grid(True)

        plt.show()

        input("Press Enter to continue...")

    def _generate_crosswalk_from_line(self, lane, sidewalk_height=None, lateral_direction=1):
        crosswalk_width = lane.width * 4  #3 ## length
        start_lat = +lane.width_at(0) - crosswalk_width / 2 - 1.7  #0.5 #0.7
        side_lat = start_lat + crosswalk_width - 1.7  #0.5 #0.7

        # print('lane inside intersection: ')
        # print(lane.index)  #('4X0_0_', '4X0_1_', 0)

        build_at_start = True
        build_at_end = True
        ### change_on_0516 ###
        crs_part = -1  # init

        if build_at_end:
            longs = np.array(
                [
                    lane.length - PGDrivableAreaProperty.SIDEWALK_LENGTH, lane.length,
                    lane.length + PGDrivableAreaProperty.SIDEWALK_LENGTH
                ]
            )
            key = f"CRS_{self.ID}_" + str(lane.index)
            ## distribute crswalk part
            if f'{self.name}0_0_' == lane.index[0] and f'{self.name}0_1_' == lane.index[1]:
                crs_part = 1
            elif f'-{self.name}0_1_' == lane.index[0] and f'-{self.name}0_0_' == lane.index[1]:
                crs_part = 2
            elif f'{self.name}1_0_' == lane.index[0] and f'{self.name}1_1_' == lane.index[1]:
                crs_part = 3
            elif f'-{self.name}1_1_' == lane.index[0] and f'-{self.name}1_0_' == lane.index[1]:
                crs_part = 4
            elif f'{self.name}2_0_' == lane.index[0] and f'{self.name}2_1_' == lane.index[1]:
                crs_part = 5
            elif f'-{self.name}2_1_' == lane.index[0] and f'-{self.name}2_0_' == lane.index[1]:
                crs_part = 6
            if crs_part in self.valid_crswalk:
                self.build_crosswalk_block(key, lane, sidewalk_height, lateral_direction, longs, start_lat, side_lat)

        ### change_on_0516 ###
        crs_part = -1  # re_init
        if build_at_start:
            longs = np.array(
                [0 - PGDrivableAreaProperty.SIDEWALK_LENGTH, 0, 0 + PGDrivableAreaProperty.SIDEWALK_LENGTH]
            )
            key = f"CRS_{self.ID}_" + str(lane.index) + "_S"
            if f'-{self.name}0_1_' == lane.index[0] and f'-{self.name}0_0_' == lane.index[1]:
                crs_part = 1
            elif f'{self.name}0_0_' == lane.index[0] and f'{self.name}0_1_' == lane.index[1]:
                crs_part = 2
            elif f'-{self.name}1_1_' == lane.index[0] and f'-{self.name}1_0_' == lane.index[1]:
                crs_part = 3
            elif f'{self.name}1_0_' == lane.index[0] and f'{self.name}1_1_' == lane.index[1]:
                crs_part = 4
            elif f'-{self.name}2_1_' == lane.index[0] and f'-{self.name}2_0_' == lane.index[1]:
                crs_part = 5
            elif f'{self.name}2_0_' == lane.index[0] and f'{self.name}2_1_' == lane.index[1]:
                crs_part = 6
            if crs_part in self.valid_crswalk:
                self.build_crosswalk_block(key, lane, sidewalk_height, lateral_direction, longs, start_lat, side_lat)
