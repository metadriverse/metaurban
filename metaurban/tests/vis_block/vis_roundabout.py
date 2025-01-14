"""This file visualizes a Roundabout block."""
from metaurban.component.pgblock.curve import Curve
from metaurban.component.pgblock.first_block import FirstPGBlock
from metaurban.component.pgblock.roundabout import Roundabout
from metaurban.component.road_network.node_road_network import NodeRoadNetwork
from metaurban.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":
    test = TestBlock(False)
    from metaurban.engine.asset_loader import initialize_asset_loader

    initialize_asset_loader(test)
    global_network = NodeRoadNetwork()
    straight = FirstPGBlock(global_network, 3.0, 1, test.render, test.world, 1)
    rd = Roundabout(1, straight.get_socket(0), global_network, 1)
    rd.construct_block(test.render, test.world)
    id = 4
    for socket_idx in range(rd.SOCKET_NUM):
        block = Curve(id, rd.get_socket(socket_idx), global_network, id + 1)
        block.construct_block(test.render, test.world)
        id += 1
    test.show_bounding_box(global_network)
    test.run()
