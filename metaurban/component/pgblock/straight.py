from metaurban.component.lane.straight_lane import StraightLane
from metaurban.component.pg_space import ParameterSpace, Parameter, BlockParameterSpace
from metaurban.component.pgblock.create_pg_block_utils import ExtendStraightLane, CreateRoadFrom, CreateAdverseRoad
from metaurban.component.pgblock.pg_block import PGBlock, PGBlockSocket
from metaurban.component.road_network import Road
from metaurban.constants import PGDrivableAreaProperty, PGLineType
import numpy as np


class Straight(PGBlock):
    """
    Straight Road
    ----------------------------------------
    ----------------------------------------
    ----------------------------------------
    """
    ID = "S"
    SOCKET_NUM = 1
    PARAMETER_SPACE = ParameterSpace(BlockParameterSpace.STRAIGHT)

    def _try_plug_into_previous_block(self) -> bool:
        self.set_part_idx(0)  # only one part in simple block like straight, and curve
        para = self.get_config()
        length = para[Parameter.length]
        basic_lane = self.positive_basic_lane
        assert isinstance(basic_lane, StraightLane), "Straight road can only connect straight type"
        new_lane = ExtendStraightLane(basic_lane, length, [PGLineType.BROKEN, PGLineType.SIDE])
        start = self.pre_block_socket.positive_road.end_node
        end = self.add_road_node()
        socket = Road(start, end)
        _socket = -socket

        # create positive road
        no_cross = CreateRoadFrom(
            new_lane,
            self.positive_lane_num,
            socket,
            self.block_network,
            self._global_network,
            ignore_intersection_checking=self.ignore_intersection_checking,
            side_lane_line_type=self.side_lane_line_type,
            center_line_type=self.center_line_type
        )
        if not self.remove_negative_lanes:
            # create negative road
            no_cross = CreateAdverseRoad(
                socket,
                self.block_network,
                self._global_network,
                ignore_intersection_checking=self.ignore_intersection_checking,
                side_lane_line_type=self.side_lane_line_type,
                center_line_type=self.center_line_type
            ) and no_cross

        self.add_sockets(PGBlockSocket(socket, _socket))
        return no_cross

    def _generate_crosswalk_from_line(self, lane, sidewalk_height=None, lateral_direction=1):
        """
        Construct the sidewalk for this lane
        Args:
            block:

        Returns:
        """
        crosswalk_width = lane.width * 4  # 3
        start_lat = +lane.width_at(0) - crosswalk_width / 2 - 1.7  #0.7
        side_lat = start_lat + crosswalk_width - 1.7  # 0.7

        build_at_start = True
        build_at_end = True
        if build_at_end:
            longs = np.array(
                [
                    lane.length - PGDrivableAreaProperty.SIDEWALK_LENGTH, lane.length,
                    lane.length + PGDrivableAreaProperty.SIDEWALK_LENGTH
                ]
            )
            key = f"CRS_{self.ID}_" + str(lane.index)
            self.build_crosswalk_block(key, lane, sidewalk_height, lateral_direction, longs, start_lat, side_lat)

        if build_at_start:
            longs = np.array(
                [0 - PGDrivableAreaProperty.SIDEWALK_LENGTH, 0, 0 + PGDrivableAreaProperty.SIDEWALK_LENGTH]
            )
            key = f"CRS_{self.ID}_" + str(lane.index) + "_S"
            self.build_crosswalk_block(key, lane, sidewalk_height, lateral_direction, longs, start_lat, side_lat)
