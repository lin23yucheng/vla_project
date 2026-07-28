from types import SimpleNamespace

from common.fixed_fps_validation import (
    find_episode_first_main_frame_ns,
    find_episode_last_main_frame_ns,
)


def test_episode_main_frame_bounds_use_log_time(monkeypatch):
    messages = [
        SimpleNamespace(log_time=110, publish_time=900),
        SimpleNamespace(log_time=190, publish_time=100),
    ]

    def fake_read(*args, **kwargs):
        return ([
            {
                "channel": SimpleNamespace(topic="/main"),
                "message": message,
            }
            for message in messages
        ], False)

    monkeypatch.setattr(
        "common.fixed_fps_validation.read_mcap_window_messages",
        fake_read,
    )

    assert find_episode_first_main_frame_ns(
        [object()], "/main", 100, 200, require_chunk_indexes=True,
    ) == 110
    assert find_episode_last_main_frame_ns(
        [object()], "/main", 100, 200, require_chunk_indexes=True,
    ) == 190
