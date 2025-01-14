"""This file visualizes the process of BIG algorithm, a procedural generation algorithm."""
from metaurban.component.algorithm.BIG import BIG
from metaurban.component.road_network.node_road_network import NodeRoadNetwork
from metaurban.engine.asset_loader import initialize_asset_loader
from metaurban.tests.vis_block.vis_block_base import TestBlock


def vis_big(debug: bool = False):
    test = TestBlock(debug=debug)

    test.cam.setPos(250, 100, 2000)

    initialize_asset_loader(engine=test)
    global_network = NodeRoadNetwork()

    big = BIG(2, 3.5, global_network, test.render, test.world, random_seed=5)
    test.vis_big(big)
    test.big.block_num = 45
    # big.generate(BigGenerateMethod.BLOCK_NUM, 10)
    test.run()


if __name__ == "__main__":
    vis_big()
