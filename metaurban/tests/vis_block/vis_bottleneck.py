"""This file visualizes a Bottleneck block."""
from metaurban.component.pgblock.bottleneck import Merge, Split
from metaurban.component.pgblock.first_block import FirstPGBlock
from metaurban.component.road_network.node_road_network import NodeRoadNetwork
from metaurban.engine.asset_loader import initialize_asset_loader
from metaurban.tests.vis_block.vis_block_base import TestBlock

if __name__ == "__main__":
    test = TestBlock()

    initialize_asset_loader(test)

    global_network = NodeRoadNetwork()
    b = FirstPGBlock(global_network, 3.0, 2, test.render, test.world, 1)
    for i in range(1, 13):
        tp = Merge if i % 3 == 0 else Split
        b = tp(i, b.get_socket(0), global_network, i)
        b.construct_block(test.render, test.world)
    test.show_bounding_box(global_network)
    test.run()
