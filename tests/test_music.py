import importlib
import sys
import types
from pathlib import Path
import numpy as np

# Ensure repository root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub external dependencies before importing music
scipy_stub = types.ModuleType('scipy')
scipy_io = types.ModuleType('io')
scipy_wavfile = types.ModuleType('wavfile')
scipy_wavfile.write = lambda *args, **kwargs: None
scipy_io.wavfile = scipy_wavfile
scipy_stub.io = scipy_io
sys.modules['scipy'] = scipy_stub
sys.modules['scipy.io'] = scipy_io
sys.modules['scipy.io.wavfile'] = scipy_wavfile

pedalboard_stub = types.ModuleType('pedalboard')
pedalboard_stub.Pedalboard = lambda *args, **kwargs: None
pedalboard_stub.LowpassFilter = lambda *args, **kwargs: None
pedalboard_stub.Reverb = lambda *args, **kwargs: None
pedalboard_stub.Compressor = lambda *args, **kwargs: None
sys.modules['pedalboard'] = pedalboard_stub

music = importlib.import_module('music')


def test_generate_sine_wave_simple():
    freq = 2
    duration = 1
    samplerate = 8
    volume = 1.0
    expected_t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    expected = volume * np.sin(2 * np.pi * freq * expected_t)
    result = music.generate_sine_wave(freq, duration, samplerate, volume)
    assert result.shape == (int(samplerate * duration),)
    assert np.allclose(result, expected)
