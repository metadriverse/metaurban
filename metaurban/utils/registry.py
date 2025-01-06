_metaurban_class_list = None
_metaurban_class_registry = None


def _initialize_registry():
    global _metaurban_class_list
    _metaurban_class_list = []

    # Register all PG blocks
    from metaurban.component.pgblock.bottleneck import Merge, Split
    from metaurban.component.pgblock.curve import Curve
    from metaurban.component.pgblock.fork import InFork, OutFork
    from metaurban.component.pgblock.parking_lot import ParkingLot
    from metaurban.component.pgblock.ramp import InRampOnStraight, OutRampOnStraight
    from metaurban.component.pgblock.roundabout import Roundabout
    from metaurban.component.pgblock.std_intersection import StdInterSection
    from metaurban.component.pgblock.std_t_intersection import StdTInterSection
    from metaurban.component.pgblock.straight import Straight
    from metaurban.component.pgblock.tollgate import TollGate
    from metaurban.component.pgblock.bidirection import Bidirection
    _metaurban_class_list.extend(
        [
            Merge, Split, Curve, InFork, OutFork, ParkingLot, InRampOnStraight, OutRampOnStraight, Roundabout,
            StdInterSection, StdTInterSection, Straight, TollGate, Bidirection
        ]
    )

    global _metaurban_class_registry
    _metaurban_class_registry = {k.__name__: k for k in _metaurban_class_list}


def get_metaurban_class(class_name):
    global _metaurban_class_registry
    if _metaurban_class_registry is None:
        _initialize_registry()

    assert class_name in _metaurban_class_registry, "{} is not in Registry: {}".format(
        class_name, _metaurban_class_registry.keys()
    )
    return _metaurban_class_registry[class_name]
