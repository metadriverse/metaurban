import copy
import math
from typing import Tuple, Union, List

import numpy as np

from metaurban.component.lane.abs_lane import AbstractLane
from metaurban.component.lane.pg_lane import PGLane
from metaurban.component.lane.circular_lane import CircularLane
from metaurban.component.lane.straight_lane import StraightLane
from metaurban.component.lane.extension_lane import ExtendingLane, ExtensionDirection
from metaurban.component.road_network import Road
from metaurban.component.road_network.node_road_network import NodeRoadNetwork
from metaurban.constants import PGLineType, PGLineColor, PGDrivableAreaProperty
from metaurban.type import MetaUrbanType
from metaurban.utils.math import get_vertical_vector
from metaurban.utils.pg.utils import check_lane_on_road


def create_extension(
    start_position: np.ndarray,
    end_position: np.ndarray,
    direction: ExtensionDirection,
    width: float = PGLane.DEFAULT_WIDTH,
    forbidden: bool = False,
    speed_limit: float = 20,
    priority: int = 0
):
    extension = ExtendingLane(
        direction, start_position, end_position, width, [PGLineType.NONE, PGLineType.CONTINUOUS], forbidden,
        speed_limit, priority
    )
    return extension


def create_bend_straight(
    previous_lane: "StraightLane",
    following_lane_length,
    radius: float,
    angle: float,
    clockwise: bool = True,
    width: float = PGLane.DEFAULT_WIDTH,
    line_types: Tuple[PGLineType, PGLineType] = None,
    forbidden: bool = False,
    speed_limit: float = 20,
    priority: int = 0
):
    bend_direction = 1 if clockwise else -1
    center = previous_lane.position(previous_lane.length, bend_direction * radius)
    p_lateral = previous_lane.direction_lateral
    x, y = p_lateral
    start_phase = np.arctan2(y, x) + (np.pi if clockwise else 0)
    bend = CircularLane(
        center, radius, start_phase, angle, clockwise, width, line_types, forbidden, speed_limit, priority
    )
    length = 2 * radius * angle / 2
    bend_end = bend.position(length, 0)
    next_lane_heading = get_vertical_vector(bend_end - center)
    # nxt_dir = next_lane_heading[0] if not clockwise else next_lane_heading[1]
    nxt_dir = next_lane_heading[0] if not clockwise else next_lane_heading[1]
    nxt_dir = np.asarray(nxt_dir)
    following_lane_end = nxt_dir * following_lane_length + bend_end
    following_lane = StraightLane(bend_end, following_lane_end, width, line_types, forbidden, speed_limit, priority)
    return bend, following_lane


def CreateRoadFrom(
    lane: Union["AbstractLane", "StraightLane", "CircularLane"],
    lane_num: int,
    road: "Road",
    roadnet_to_add_lanes: "NodeRoadNetwork",  # mostly, block network
    roadnet_to_check_cross: "NodeRoadNetwork",  # mostly, previous global_network
    toward_smaller_lane_index: bool = True,
    ignore_start: str = None,
    ignore_end: str = None,
    center_line_type=None,  # Identical to Block.CENTER_LINE_TYPE
    detect_one_side=True,
    side_lane_line_type=None,
    inner_lane_line_type=None,
    center_line_color=None,
    ignore_intersection_checking=None,
    metaurban_lane_type=None,
) -> bool:
    """
    Give the far left lane, then create lane_num lanes including the far left lane itself.
        | | | |
        | | | |
        | | | |
        | | | |

    Args:
        lane:
        lane_num:
        road:
        roadnet_to_add_lanes: The block network
        roadnet_to_check_cross: Existing global network
        toward_smaller_lane_index:
        ignore_start:
        ignore_end:
        center_line_type:
        detect_one_side:
        side_lane_line_type:
        inner_lane_line_type:
        center_line_color:
        ignore_intersection_checking:
        metaurban_lane_type:

    Return:
        A Boolean indicator saying whether the road is constructed successfully.
    """
    center_line_type = center_line_type or PGLineType.CONTINUOUS
    side_lane_line_type = side_lane_line_type or PGLineType.SIDE
    side_lane_line_type = side_lane_line_type or PGLineType.SIDE
    inner_lane_line_type = inner_lane_line_type or PGLineType.BROKEN
    center_line_color = center_line_color or PGLineColor.YELLOW

    lane_num -= 1  # include lane itself
    origin_lane = lane
    lanes = []
    lane_width = lane.width_at(0)
    for i in range(lane_num, 0, -1):
        side_lane = copy.copy(lane)
        if isinstance(lane, StraightLane):
            width = -lane_width if toward_smaller_lane_index else lane_width
            start = side_lane.position(0, width)
            end = side_lane.position(side_lane.length, width)
            side_lane.start = start
            side_lane.end = end
        elif isinstance(lane, CircularLane):
            new_lane_clockwise = True if lane.is_clockwise() else False
            radius1 = lane.radius
            if not toward_smaller_lane_index:
                radius2 = radius1 - lane_width if new_lane_clockwise else radius1 + lane_width
            else:
                radius2 = radius1 + lane_width if new_lane_clockwise else radius1 - lane_width
            side_lane.radius = radius2
            side_lane.update_properties()
        if i == 1:
            side_lane.line_types = [center_line_type, inner_lane_line_type] if toward_smaller_lane_index else \
                [inner_lane_line_type, side_lane_line_type]
        else:
            side_lane.line_types = [inner_lane_line_type, inner_lane_line_type]
        lanes.append(side_lane)
        lane = side_lane
    if toward_smaller_lane_index:
        lanes.reverse()
        lanes.append(origin_lane)
        origin_lane.line_types = [inner_lane_line_type if len(lanes) > 1 else center_line_type, side_lane_line_type]
    else:
        lanes.insert(0, origin_lane)
        if len(lanes) > 1:
            line_type = origin_lane.line_types[0], lanes[-1].line_types[0]
            origin_lane.line_types = line_type
    # check the left lane and right lane
    ignore = (ignore_start, ignore_end)
    factor = (
        PGDrivableAreaProperty.SIDEWALK_WIDTH + PGDrivableAreaProperty.SIDEWALK_LINE_DIST + lane_width / 2.0
    ) * 2.0 / lane_width
    if not detect_one_side:
        # Because of side walk, the width of side walk should be consider
        no_cross = not (
            check_lane_on_road(
                roadnet_to_check_cross,
                origin_lane,
                factor,
                ignore,
                ignore_intersection_checking=ignore_intersection_checking
            ) or check_lane_on_road(
                roadnet_to_check_cross,
                lanes[0],
                -0.95,
                ignore,
                ignore_intersection_checking=ignore_intersection_checking
            )
        )
    else:
        no_cross = not check_lane_on_road(
            roadnet_to_check_cross,
            origin_lane,
            factor,
            ignore,
            ignore_intersection_checking=ignore_intersection_checking
        )
    for l in lanes:
        roadnet_to_add_lanes.add_lane(road.start_node, road.end_node, l)
        if metaurban_lane_type is not None:
            assert MetaUrbanType.is_lane(metaurban_lane_type)
            l.set_metaurban_type(metaurban_lane_type)
    if lane_num == 0:
        lanes[-1].line_types = [center_line_type, side_lane_line_type]
    lanes[0].line_colors = [center_line_color, PGLineColor.GREY]
    return no_cross


def ExtendStraightLane(
    lane: "StraightLane",
    extend_length: float,
    line_types: (PGLineType, PGLineType),
    metaurban_lane_type=None
) -> "StraightLane":
    assert isinstance(lane, StraightLane)
    new_lane = copy.copy(lane)
    start_point = lane.end
    end_point = lane.position(lane.length + extend_length, 0)
    new_lane.start = start_point
    new_lane.end = end_point
    new_lane.line_types = line_types
    new_lane.update_properties()
    if metaurban_lane_type is not None:
        assert MetaUrbanType.is_lane(metaurban_lane_type)
        new_lane.set_metaurban_type(metaurban_lane_type)
    return new_lane


def get_lanes_on_road(road: "Road", roadnet: "NodeRoadNetwork") -> List[AbstractLane]:
    return roadnet.graph[road.start_node][road.end_node]


def CreateAdverseRoad(
    positive_road: "Road",
    roadnet_to_get_road: "NodeRoadNetwork",  # mostly, block network
    roadnet_to_check_cross: "NodeRoadNetwork",  # mostly, previous global network
    ignore_start: str = None,
    ignore_end: str = None,
    center_line_type=None,
    side_lane_line_type=None,
    inner_lane_line_type=None,
    center_line_color=None,
    ignore_intersection_checking=None,
    metaurban_lane_type=None
) -> bool:
    """
    Create the adverse road given the road network.

    Args:
        positive_road:
        roadnet_to_get_road: The block network
        roadnet_to_check_cross: The existing global network
        ignore_start:
        ignore_end:
        center_line_type:
        side_lane_line_type:
        inner_lane_line_type:
        center_line_color:
        ignore_intersection_checking:
        metaurban_lane_type:

    Returns:
        A Boolean indicator saying whether the road is constructed successfully.
    """
    center_line_type = center_line_type or PGLineType.CONTINUOUS  # Identical to Block.CENTER_LINE_TYPE
    side_lane_line_type = side_lane_line_type or PGLineType.SIDE
    inner_lane_line_type = inner_lane_line_type or PGLineType.BROKEN
    center_line_color = center_line_color or PGLineColor.YELLOW

    adverse_road = -positive_road
    lanes = get_lanes_on_road(positive_road, roadnet_to_get_road)
    reference_lane = lanes[-1]
    num = len(lanes) * 2
    width = reference_lane.width_at(0)
    if isinstance(reference_lane, StraightLane):
        start_point = reference_lane.position(lanes[-1].length, -(num - 1) * width)
        end_point = reference_lane.position(0, -(num - 1) * width)
        symmetric_lane = StraightLane(
            start_point, end_point, width, lanes[-1].line_types, reference_lane.forbidden, reference_lane.speed_limit,
            reference_lane.priority
        )
    elif isinstance(reference_lane, CircularLane):
        start_phase = reference_lane.end_phase
        angle = reference_lane.angle
        new_lane_clockwise = False if reference_lane.is_clockwise() else True
        if not new_lane_clockwise:
            radius = reference_lane.radius + (num - 1) * width
        else:
            radius = reference_lane.radius - (num - 1) * width
        symmetric_lane = CircularLane(
            reference_lane.center, radius, start_phase, angle, new_lane_clockwise, width, reference_lane.line_types,
            reference_lane.forbidden, reference_lane.speed_limit, reference_lane.priority
        )
    else:
        raise ValueError("Creating other lanes is not supported")
    success = CreateRoadFrom(
        symmetric_lane,
        int(num / 2),
        adverse_road,
        roadnet_to_get_road,
        roadnet_to_check_cross,
        ignore_start=ignore_start,
        ignore_end=ignore_end,
        side_lane_line_type=side_lane_line_type,
        inner_lane_line_type=inner_lane_line_type,
        center_line_type=center_line_type,
        center_line_color=center_line_color,
        ignore_intersection_checking=ignore_intersection_checking,
        metaurban_lane_type=metaurban_lane_type
    )
    positive_road.get_lanes(roadnet_to_get_road)[0].line_colors = [center_line_color, PGLineColor.GREY]
    return success


def CreateTwoWayRoad(
    road_to_change: "Road",
    roadnet_to_get_road: "NodeRoadNetwork",  # mostly, block network
    roadnet_to_check_cross: "NodeRoadNetwork",  # mostly, previous global network
    new_road_name: Road = None,
    ignore_start: str = None,
    ignore_end: str = None,
    center_line_type=PGLineType.CONTINUOUS,  # Identical to Block.CENTER_LINE_TYPE
    side_lane_line_type=PGLineType.SIDE,
    inner_lane_line_type=PGLineType.BROKEN,
    ignore_intersection_checking=None
) -> bool:
    """
    This function will add a new road in reverse direction to the road network
    Then the road will change from:
    ---------->
    ---------->
    to:
    <--------->
    <--------->
    As a result, vehicles can drive in both direction
    :return: cross or not
    """
    adverse_road = Road(road_to_change.end_node, road_to_change.start_node) if new_road_name is None else new_road_name
    lanes = get_lanes_on_road(road_to_change, roadnet_to_get_road)
    reference_lane = lanes[-1]
    num = len(lanes)
    width = reference_lane.width_at(0)
    if isinstance(reference_lane, StraightLane):
        start_point = reference_lane.position(lanes[-1].length, -(num - 1) * width)
        end_point = reference_lane.position(0, -(num - 1) * width)
        symmetric_lane = StraightLane(
            start_point, end_point, width, lanes[-1].line_types, reference_lane.forbidden, reference_lane.speed_limit,
            reference_lane.priority
        )
    elif isinstance(reference_lane, CircularLane):
        start_phase = reference_lane.end_phase
        angle = reference_lane.angle
        new_lane_clockwise = False if reference_lane.is_clockwise() else True
        if not new_lane_clockwise:
            radius = reference_lane.radius + (num - 1) * width
        else:
            radius = reference_lane.radius - (num - 1) * width
        symmetric_lane = CircularLane(
            reference_lane.center, radius, start_phase, angle, new_lane_clockwise, width, reference_lane.line_types,
            reference_lane.forbidden, reference_lane.speed_limit, reference_lane.priority
        )
    else:
        raise ValueError("Creating other lanes is not supported")
    success = CreateRoadFrom(
        symmetric_lane,
        num,
        adverse_road,
        roadnet_to_get_road,
        roadnet_to_check_cross,
        ignore_start=ignore_start,
        ignore_end=ignore_end,
        side_lane_line_type=side_lane_line_type,
        inner_lane_line_type=inner_lane_line_type,
        center_line_type=center_line_type,
        ignore_intersection_checking=ignore_intersection_checking
    )
    return success


def block_socket_merge(
    socket_1: "BlockSocket", socket_2: "BlockSocket", global_network: "NodeRoadNetwork", positive_merge: False
):
    global_network.graph[socket_1.positive_road.start_node][socket_2.negative_road.start_node] = \
        global_network.graph[socket_1.positive_road.start_node].pop(socket_1.positive_road.end_node)

    global_network.graph[socket_2.positive_road.start_node][socket_1.negative_road.start_node] = \
        global_network.graph[socket_2.positive_road.start_node].pop(socket_2.positive_road.end_node)


def create_wave_lanes(
    pre_lane, lateral_dist: float, wave_length: float, last_straight_length: float, lane_width, toward_left=True
):
    """
    Prodeuce two lanes in adverse direction
    :param pre_lane: Previous abstract lane
    :param lateral_dist: the dist moved in previous lane's lateral direction
    :param wave_length: the length of two circular lanes in the previous lane's longitude direction
    :param following_lane_length: the length of last straight lane
    :return: List[Circular lane]
    """
    angle = np.pi - 2 * np.arctan(wave_length / (2 * lateral_dist))
    radius = wave_length / (2 * math.sin(angle))
    circular_lane_1, pre_lane = create_bend_straight(
        pre_lane, 10, radius, angle, False if toward_left else True, lane_width, [PGLineType.NONE, PGLineType.NONE]
    )
    pre_lane.reset_start_end(pre_lane.position(-10, 0), pre_lane.position(pre_lane.length - 10, 0))
    circular_lane_2, straight_lane = create_bend_straight(
        pre_lane, last_straight_length, radius, angle, True if toward_left else False, lane_width,
        [PGLineType.NONE, PGLineType.NONE]
    )
    return circular_lane_1, circular_lane_2, straight_lane
