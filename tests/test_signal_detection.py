import numpy as np

from muwave.audio.fsk import FSKModulator, FSKDemodulator
from muwave.core.config import Config


def test_start_end_signal_detection_with_config_frequencies():
    cfg = Config()
    fsk_cfg = cfg.create_fsk_config(symbol_duration_ms=cfg.get_speed_mode_settings().get("symbol_duration_ms", 60.0), num_channels=2)

    mod = FSKModulator(fsk_cfg)
    demod = FSKDemodulator(fsk_cfg)

    data = b"ok"
    samples = mod.encode_data(data)

    detected, start_pos = demod.detect_start_signal(samples)
    assert detected, "Start signal should be detected"
    assert start_pos > 0

    end_detected, end_pos = demod.detect_end_signal(samples[start_pos:])
    assert end_detected, "End signal should be detected"
    assert end_pos > 0
