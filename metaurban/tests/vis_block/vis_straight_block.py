"""This file visualizes a Straight block."""
from metaurban.component.pgblock.first_block import FirstPGBlock
from metaurban.component.pgblock.straight import Straight
from metaurban.component.road_network.node_road_network import NodeRoadNetwork
from metaurban.engine.asset_loader import initialize_asset_loader
from metaurban.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":
    test = TestBlock()
    initialize_asset_loader(test)
    global_network = NodeRoadNetwork()
    straight = FirstPGBlock(global_network, 3.0, 1, test.render, test.world, 1)
    for i in range(1, 3):
        straight = Straight(i, straight.get_socket(0), global_network, i)
        straight.construct_block(test.render, test.world)
    test.show_bounding_box(global_network)
    test.run()
