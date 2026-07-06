from led_controller_nodes.group_utils import resolve_groups, validate_groups


def test_resolve_groups_preserves_requested_order():
    assert resolve_groups([2, 0, 1], 3) == [2, 0, 1]


def test_empty_groups_selects_all_groups():
    assert resolve_groups([], 3) == [0, 1, 2]


def test_validate_groups_rejects_invalid_indexes():
    assert validate_groups([-1, 0], 3) is not None
    assert validate_groups([0, 3], 3) is not None


def test_validate_groups_rejects_duplicates():
    assert validate_groups([1, 1], 3) is not None


def test_validate_groups_accepts_unsorted_unique_indexes():
    assert validate_groups([2, 0], 3) is None


def test_four_group_configuration_accepts_group_three():
    assert validate_groups([0, 1, 2, 3], 4) is None
    assert validate_groups([4], 4) is not None
