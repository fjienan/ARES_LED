from typing import List, Optional, Sequence


def validate_groups(requested_groups: Sequence[int], led_count: int) -> Optional[str]:
    groups = [int(group) for group in requested_groups]
    invalid = [group for group in groups if group < 0 or group >= led_count]
    if invalid:
        return f'invalid group indexes: {invalid}'
    if len(set(groups)) != len(groups):
        return f'duplicate group indexes are not allowed: {groups}'
    return None


def resolve_groups(requested_groups: Sequence[int], led_count: int) -> List[int]:
    if len(requested_groups) == 0:
        return list(range(led_count))
    return [int(group) for group in requested_groups]
