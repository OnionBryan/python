import timeit
import numpy as np
import music

# Old implementations for benchmarking

def old_generate_melody(frequencies, duration, samplerate, volume):
    melody = np.array([])
    t_note = 0
    while t_note < duration:
        note_duration = np.random.choice([0.5, 1, 1.5, 2], p=[0.4, 0.3, 0.2, 0.1])
        if t_note + note_duration > duration:
            note_duration = duration - t_note
        note_freq = np.random.choice(list(frequencies.values()))
        instrument = np.random.choice(['piano', 'guitar', 'sine'], p=[0.6, 0.3, 0.1])
        melody_piece = music.generate_instrument_wave(note_freq, note_duration, samplerate, volume, instrument)
        melody = np.concatenate((melody, melody_piece))
        t_note += note_duration
    return melody


def old_generate_bassline(frequencies, duration, samplerate, volume):
    bassline = np.array([])
    t_note = 0
    while t_note < duration:
        note_duration = np.random.choice([1, 2], p=[0.8, 0.2])
        if t_note + note_duration > duration:
            note_duration = duration - t_note
        note_freq = np.random.choice(list(frequencies.values()))
        bass_piece = music.generate_sine_wave(note_freq, note_duration, samplerate, volume)
        bassline = np.concatenate((bassline, bass_piece))
        t_note += note_duration
    return bassline


# Benchmark helper
def benchmark():
    setup = {
        'frequencies': music.note_frequencies,
        'bass_freqs': music.bass_frequencies,
        'duration': 30,
        'samplerate': music.SAMPLERATE,
        'volume': music.VOLUME,
    }

    benchmarks = []
    benchmarks.append(
        (
            'generate_melody',
            lambda: music.generate_melody(setup['frequencies'], setup['duration'], setup['samplerate'], setup['volume']),
            lambda: old_generate_melody(setup['frequencies'], setup['duration'], setup['samplerate'], setup['volume']),
        )
    )
    benchmarks.append(
        (
            'generate_bassline',
            lambda: music.generate_bassline(setup['bass_freqs'], setup['duration'], setup['samplerate'], setup['volume']),
            lambda: old_generate_bassline(setup['bass_freqs'], setup['duration'], setup['samplerate'], setup['volume']),
        )
    )

    for name, new_func, old_func in benchmarks:
        new_time = timeit.timeit(new_func, number=1)
        old_time = timeit.timeit(old_func, number=1)
        improvement = ((old_time - new_time) / old_time) * 100 if old_time else 0
        print(f"{name}: old={{old_time:.4f}}s, new={{new_time:.4f}}s, improvement={{improvement:.2f}}%".format(old_time=old_time, new_time=new_time, improvement=improvement))


if __name__ == "__main__":
    benchmark()
