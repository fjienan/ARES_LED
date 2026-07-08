from rgb_comm_protocol import FixedColorProtocol
import pytest


def test_all_commands_round_trip():
    protocol = FixedColorProtocol()
    assert set(protocol.commands) == {0, 1, 2, 3, 4, 5, 6, 7, 8}
    assert protocol.code_length == 3
    assert protocol.decode_reversed is False
    assert protocol.encode_symbols(0) == ('BLUE', 'PURPLE', 'RED')
    assert protocol.encode_symbols(1) == ('BLUE', 'RED', 'GREEN')
    assert protocol.decode(('BLUE', 'PURPLE', 'RED')) == 0
    assert protocol.decode(('RED', 'PURPLE', 'BLUE')) is None
    for command, symbols in protocol.commands.items():
        assert protocol.decode(symbols) == command
        assert protocol.decode(reversed(symbols)) is None
        assert len(symbols) == 3
        assert len(set(symbols)) == 3
    assert all(
        len(set(symbols)) > 1
        for symbols in protocol.commands.values()
    )


def test_unknown_values_are_rejected():
    protocol = FixedColorProtocol()
    assert protocol.encode_symbols(99) is None
    assert protocol.decode(('CYAN', 'PURPLE', 'RED')) is None


def test_reversed_commands_are_rejected_as_conflicts(tmp_path):
    config = tmp_path / 'protocol.yaml'
    config.write_text(
        'decode_reversed: true\n'
        'commands:\n'
        '  1: [RED, BLUE, GREEN]\n'
        '  2: [GREEN, BLUE, RED]\n',
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match='conflict under reversal'):
        FixedColorProtocol(config_path=str(config))


def test_reversed_commands_can_exist_when_reverse_decode_is_disabled(tmp_path):
    config = tmp_path / 'protocol.yaml'
    config.write_text(
        'decode_reversed: false\n'
        'commands:\n'
        '  1: [RED, BLUE, GREEN]\n'
        '  2: [GREEN, BLUE, RED]\n',
        encoding='utf-8',
    )

    protocol = FixedColorProtocol(config_path=str(config))
    assert protocol.decode(('RED', 'BLUE', 'GREEN')) == 1
    assert protocol.decode(('GREEN', 'BLUE', 'RED')) == 2
