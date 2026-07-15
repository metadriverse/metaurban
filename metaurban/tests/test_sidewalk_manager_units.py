"""Engine-free unit checks for the sidewalk manager's placement primitives."""
import math

from metaurban.manager.sidewalk_manager import (
    _REGION_ORDER, _SIDEWALK_BANDS, AssetManager, ObjectPlacer
)


class _Lane:
    def width_at(self, s):
        return 3.5


def _grid(n_long, n_lat):
    return [[False] * n_lat for _ in range(n_long)]


def _obj(length=1.0, width=1.0, mode='normal', gap=3):
    return {
        'CLASS_NAME': 'obj', 'general': {'length': length, 'width': width},
        'obj_generation_mode': mode, 'spawn_long_gap': gap,
    }


def test_lateral_ranges_are_contiguous_bands():
    lane, widths = _Lane(), [1.5, 2.0, 3.0, 1.0, 2.5, 25.0]
    for sidewalk_type, bands in _SIDEWALK_BANDS.items():
        expected_start = lane.width_at(0) / 2
        for region, width_index in bands:
            lo, hi = AssetManager.calculate_lateral_range(None, region, lane, widths, sidewalk_type)
            assert math.isclose(lo, expected_start), (sidewalk_type, region)
            assert math.isclose(hi - lo, widths[width_index]), (sidewalk_type, region)
            expected_start = hi
        assert set(_REGION_ORDER[sidewalk_type]) == {r for r, _ in bands}


def test_placer_respects_occupancy_and_gap():
    placer = ObjectPlacer(_grid(30, 8))
    placer.buffer = 2
    ok, last = placer.place_object(_obj(), 0)
    assert ok and last >= 3  # first free row after the gap
    # the occupied cells must never be reused by the next placement
    ok2, _ = placer.place_object(_obj(), 0)
    assert ok2
    (pos1, pos2) = [pos for pos, _ in placer.placed_objects.values()]
    assert pos1 != pos2
    # an object wider than the grid can never be placed
    assert placer.place_object(_obj(width=20.0), 0) == (False, 0)


def test_parallel_only_sticks_to_first_column():
    placer = ObjectPlacer(_grid(30, 8))
    placer.buffer = 0
    ok, _ = placer.place_object(_obj(mode='parallel_only'), 0)
    assert ok
    assert all(pos[1] == 1 for pos, _ in placer.placed_objects.values())


def test_inverse_scans_backwards():
    placer = ObjectPlacer(_grid(30, 8))
    placer.buffer = 0
    ok, last = placer.place_object(_obj(mode='inverse'), 0)
    assert ok and last >= 25  # placed near the far end


if __name__ == '__main__':
    test_lateral_ranges_are_contiguous_bands()
    test_placer_respects_occupancy_and_gap()
    test_parallel_only_sticks_to_first_column()
    test_inverse_scans_backwards()
    print('sidewalk manager unit checks passed')
