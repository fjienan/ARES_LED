"""Bridge three-segment detections to fixed-color protocol commands."""

from .protocol_decoder import ProtocolDetection


def protocol_detection_from_three_segment(protocol, item):
    command_id = protocol.decode(item.symbols)
    if command_id is None:
        return None
    return ProtocolDetection(
        command_id=int(command_id),
        symbols=item.symbols,
        segments=item.segments,
        score=item.score,
        confidence=item.confidence,
        geometry_quality=item.geometry_quality,
        angle_degrees=item.angle_degrees,
        cross_distance=item.cross_distance,
        center_distance_ratio=item.center_distance_ratio,
        gap_ratio=item.gap_ratio,
    )


def protocol_candidates_from_triples(protocol, triples):
    candidates = []
    for item in triples:
        if item.ambiguous:
            continue
        decoded = protocol_detection_from_three_segment(protocol, item)
        if decoded is not None:
            candidates.append(decoded)
    return candidates


def protocol_winner_from_triples(protocol, triples):
    if not triples:
        return None
    winner = triples[0]
    if winner.ambiguous:
        return None
    return protocol_detection_from_three_segment(protocol, winner)
