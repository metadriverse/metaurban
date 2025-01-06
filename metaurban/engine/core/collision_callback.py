from metaurban.constants import MetaUrbanType
from metaurban.utils.utils import get_object_from_node


def collision_callback(contact):
    """
    All collision callback should be here, and a notify() method can turn it on
    It may lower the performance if overdone
    """

    # now it only process BaseVehicle collision
    node0 = contact.getNode0()
    node1 = contact.getNode1()

    nodes = [node0, node1]
    another_nodes = [node1, node0]
    for i in range(2):
        if nodes[i].hasPythonTag(MetaUrbanType.VEHICLE):
            obj_type = another_node_name = another_nodes[i].getName()
            if obj_type in [MetaUrbanType.BOUNDARY_SIDEWALK, MetaUrbanType.CROSSWALK] \
                    or MetaUrbanType.is_road_line(obj_type):
                continue
            # print(obj_type)
            obj_1 = get_object_from_node(nodes[i])
            obj_2 = get_object_from_node(another_nodes[i])

            # crash vehicles
            if another_node_name == MetaUrbanType.VEHICLE:
                obj_1.crash_vehicle = True
            # crash objects
            elif MetaUrbanType.is_traffic_object(another_node_name):
                if not obj_2.crashed:
                    obj_1.crash_object = True
                    if obj_2.COST_ONCE:
                        obj_2.crashed = True
            # collision_human
            elif another_node_name in [MetaUrbanType.CYCLIST, MetaUrbanType.PEDESTRIAN]:
                obj_1.crash_human = True
            # crash invisible wall or building
            elif another_node_name in [MetaUrbanType.INVISIBLE_WALL, MetaUrbanType.BUILDING]:
                obj_1.crash_building = True
            # logging.debug("{} crash with {}".format(nodes[i].getName(), another_nodes[i].getName()))
