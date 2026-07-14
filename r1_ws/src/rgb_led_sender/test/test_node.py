from types import SimpleNamespace

from rgb_led_sender.node import RgbLedSender


class StubProtocol:
    def encode_symbols(self, command):
        if command == 1:
            return ('BLUE', 'RED', 'GREEN')
        return None

    def encode_rgb(self, command):
        if command == 1:
            return ((0, 0, 255), (255, 0, 0), (0, 255, 0))
        return None


class StubTimer:
    def __init__(self):
        self.cancel_count = 0
        self.reset_count = 0

    def cancel(self):
        self.cancel_count += 1

    def reset(self):
        self.reset_count += 1


def make_sender():
    sender = object.__new__(RgbLedSender)
    sender.get_logger = lambda: SimpleNamespace(
        info=lambda *args: None,
        warning=lambda *args: None,
    )
    sender.protocol = StubProtocol()
    sender.pending_id = None
    sender.active_command = None
    sender.idle_effect_enabled = True
    sender.idle_effect_payload = 'idle-payload'
    sender.display_segments = ()
    sender.pixel_count = 6
    sender.brightness_mode = 'segment_bri'
    sender.wled_master_brightness = 255.0
    sender.idle_command_delay_sec = 0.1
    sender.idle_delay_pending = False
    sender._idle_timer = StubTimer()
    return sender


def test_zero_command_enters_idle_and_nonzero_command_leaves_idle(monkeypatch):
    sender = make_sender()
    sent = []
    monkeypatch.setattr(sender, '_write_serial_payload', lambda payload: sent.append(payload) or '')
    monkeypatch.setattr(
        'rgb_led_sender.node.build_wled_state_json',
        lambda *args: 'static-payload',
    )

    sender.initial_command_id = -1
    sender._queue_initial_command()
    assert sender.pending_id == 0
    assert sender.idle_delay_pending is True
    assert sender._idle_timer.reset_count == 1

    sender._dispatch_serial()
    assert sent == []

    sender._dispatch_delayed_idle()
    assert sent == ['idle-payload']
    assert sender.pending_id is None
    assert sender.active_command == 0

    sender._on_command(SimpleNamespace(data=1))
    assert sent == ['idle-payload', 'static-payload']
    assert sender.pending_id is None
    assert sender.active_command == 1

    sender._on_command(SimpleNamespace(data=0))
    assert sent == ['idle-payload', 'static-payload']
    sender._dispatch_delayed_idle()
    assert sent == ['idle-payload', 'static-payload', 'idle-payload']
    assert sender.active_command == 0

    sender._on_command(SimpleNamespace(data=1))
    assert sent == [
        'idle-payload',
        'static-payload',
        'idle-payload',
        'static-payload',
    ]
    assert sender.active_command == 1


def test_idle_command_is_retried_after_serial_failure(monkeypatch):
    sender = make_sender()
    sender.pending_id = 0
    sender.idle_delay_pending = False
    responses = iter((None, ''))
    monkeypatch.setattr(sender, '_write_serial_payload', lambda payload: next(responses))

    sender._dispatch_serial()
    assert sender.pending_id == 0
    assert sender.active_command is None

    sender._dispatch_serial()
    assert sender.pending_id is None
    assert sender.active_command == 0


def test_short_zero_command_is_cancelled_by_static_command(monkeypatch):
    sender = make_sender()
    dispatched = []
    monkeypatch.setattr(sender, '_dispatch', lambda: dispatched.append(sender.pending_id))

    sender._on_command(SimpleNamespace(data=0))
    assert sender.pending_id == 0
    assert sender.idle_delay_pending is True
    assert dispatched == []

    sender._on_command(SimpleNamespace(data=1))
    assert sender.pending_id == 1
    assert sender.idle_delay_pending is False
    assert dispatched == [1]
    assert sender._idle_timer.cancel_count == 1


def test_active_static_command_cancels_pending_idle_without_resend(monkeypatch):
    sender = make_sender()
    sender.active_command = 1
    dispatched = []
    monkeypatch.setattr(sender, '_dispatch', lambda: dispatched.append(sender.pending_id))

    sender._on_command(SimpleNamespace(data=0))
    assert sender.pending_id == 0

    sender._on_command(SimpleNamespace(data=1))
    assert sender.pending_id is None
    assert sender.active_command == 1
    assert sender.idle_delay_pending is False
    assert dispatched == []


def test_active_idle_command_is_deduplicated(monkeypatch):
    sender = make_sender()
    dispatched = []
    monkeypatch.setattr(sender, '_dispatch', lambda: dispatched.append(sender.pending_id))

    sender.active_command = 0
    sender._on_command(SimpleNamespace(data=0))
    assert sender.pending_id is None
    assert dispatched == []
