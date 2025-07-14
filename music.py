

import numpy as np
from scipy.io.wavfile import write
import random
from pedalboard import Pedalboard, LowpassFilter, Reverb, Compressor

# --- Global Parameters ---
SAMPLERATE = 44100  # Samples per second, standard for audio
DURATION = 300      # Duration of the generated music in seconds
VOLUME = 0.4        # Overall volume level for the generated audio
BASE_BPM = 60       # Base Beats Per Minute for rhythmic elements

# --- Key, Scale, and Transposition Definitions ---
# Maps musical keys to their semitone offset from C (C=0, C#=1, D=2, etc.)
key_to_semitones = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
}
# Calculates the transposition factor for each key based on semitones
# (2^(semitones/12) is the formula for frequency transposition in music)
key_to_transposition_factor = {
    key: 2 ** (semitones / 12.0)
    for key, semitones in key_to_semitones.items()
}

# --- Note and Chord Definitions (Based on C Major before transposition) ---
# Frequencies for various notes, primarily in the middle to higher ranges
note_frequencies = {
    "A1": 55.00, "C2": 65.41, "D2": 73.42, "E2": 82.41, "F2": 87.31, "G2": 98.00,
    'A2': 110.00, 'B2': 123.47, 'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61, 'G3': 196.00,
    'A3': 220.00, 'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00,
    'A4': 440.00
}
# Frequencies for bass notes (half of the corresponding note_frequencies for lower octaves)
# Note: These are already very low. The kick drum will pick from these.
bass_frequencies = {note: freq / 2 for note, freq in note_frequencies.items() if '1' in note or '2' in note}
# Definitions of chords by the notes they contain (using the note names)
chords = {
    'Fmaj7': ['F2', 'A2', 'C3', 'E3'],
    'Am9': ['A1', 'C2', 'E2', 'G2', 'B2'],
    'Dm9': ['D2', 'F2', 'A2', 'C3', 'E3'],
    'Gsus4': ['G2', 'C3', 'D3'],
    'Cmaj13': ['C2', 'E2', 'G2', 'D3', 'A2'],
    'Cmaj7': ['C2', 'E2', 'G2', 'B2'], # Added for new progressions
    'G7': ['G2', 'B2', 'D3', 'F3'],    # Added for new progressions
    'Em7': ['E2', 'G2', 'B2', 'D3'],    # Added for new progressions
    'Am7': ['A2', 'C3', 'E3', 'G3'],    # Added for new progressions
    'D7': ['D2', 'F#2', 'A2', 'C3'],   # Added for new progressions
    'Gmaj7': ['G2', 'B2', 'D3', 'F#3'] # Added for new progressions
}

# Define multiple chord progressions to choose from
all_chord_progressions = [
    ["Fmaj7", "Am9", "Dm9", "Gsus4", "Cmaj13"], # Original progression
    ["Cmaj7", "Am7", "Dm7", "G7"],             # Common I-vi-ii-V
    ["Em7", "Am7", "Dm7", "G7"],               # Minor progression
    ["Cmaj7", "Fmaj7", "Dm7", "G7"],           # Another common progression
    ["Am7", "Gmaj7", "Fmaj7", "Em7"]           # Descending progression
]


# --- Functions ---

def transpose(note_freq_dict, factor):
    """
    Transposes a dictionary of note frequencies by a given factor.
    Args:
        note_freq_dict (dict): A dictionary where keys are note names and values are their frequencies.
        factor (float): The transposition factor (e.g., 2^(semitones/12)).
    Returns:
        dict: A new dictionary with transposed frequencies.
    """
    if factor == 1.0: return note_freq_dict # No transposition needed
    return {name: freq * factor for name, freq in note_freq_dict.items()}

def generate_sine_wave(frequency, duration, samplerate, volume):
    """
    Generates a simple sine wave.
    Args:
        frequency (float): The frequency of the sine wave in Hz.
        duration (float): The duration of the wave in seconds.
        samplerate (int): The number of samples per second.
        volume (float): The amplitude of the wave.
    Returns:
        np.array: A NumPy array representing the sine wave.
    """
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    return volume * np.sin(2 * np.pi * frequency * t)

def generate_instrument_wave(frequency, duration, samplerate, volume, instrument='piano'):
    """
    Generates a wave simulating different instruments with ADSR envelope.
    Args:
        frequency (float): The fundamental frequency of the note.
        duration (float): The duration of the note in seconds.
        samplerate (int): The number of samples per second.
        volume (float): The volume of the note.
        instrument (str): The type of instrument ('piano', 'guitar', 'sine').
    Returns:
        np.array: A NumPy array representing the instrument wave.
    """
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    wave = np.zeros_like(t)
    if frequency == 0 or duration <= 0: return wave # Return silent wave for invalid inputs

    if instrument == 'piano':
        # Piano sound with fundamental and overtones
        wave = (0.6 * np.sin(2 * np.pi * frequency * t) +
                0.3 * np.sin(4 * np.pi * frequency * t) +
                0.1 * np.sin(6 * np.pi * frequency * t))
        # ADSR envelope parameters for piano
        attack_time, decay_time, sustain_level, release_time = 0.05, 0.2, 0.7, 0.4

        # Calculate sample points for each ADSR stage
        attack_samples = int(samplerate * attack_time)
        decay_samples = int(samplerate * decay_time)
        release_samples = int(samplerate * release_time)
        sustain_samples = len(t) - attack_samples - decay_samples - release_samples

        # Adjust timings if total ADSR time exceeds note duration
        if sustain_samples < 0:
            sustain_samples = 0
            total_samples = attack_samples + decay_samples + release_samples
            if total_samples > 0: # Avoid division by zero
                attack_samples, decay_samples, release_samples = [
                    int(s * len(t) / total_samples) for s in [attack_samples, decay_samples, release_samples]
                ]
            sustain_samples = len(t) - attack_samples - decay_samples - release_samples
            if sustain_samples < 0: sustain_samples = 0 # Ensure sustain is not negative

        # Create the ADSR envelope
        envelope = np.concatenate([
            np.linspace(0, 1, attack_samples),                       # Attack phase
            np.linspace(1, sustain_level, decay_samples),           # Decay phase
            np.full(sustain_samples, sustain_level),                # Sustain phase
            np.linspace(sustain_level, 0, release_samples)          # Release phase
        ])
        # Pad envelope if it's shorter than the wave (due to rounding)
        if len(envelope) < len(wave):
            envelope = np.pad(envelope, (0, len(wave) - len(envelope)), 'constant')
        wave *= envelope[:len(wave)] # Apply envelope to the wave

    elif instrument == 'guitar':
        # Guitar sound with fundamental and overtones, and a quick decay
        wave = (0.7 * np.sin(2 * np.pi * frequency * t) +
                0.2 * np.sin(4 * np.pi * frequency * t) +
                0.1 * np.sin(8 * np.pi * frequency * t))
        decay = np.exp(-np.linspace(0, 5, len(t))) # Exponential decay for guitar
        wave *= decay
    else:
        # Default to a simple sine wave if instrument is not recognized
        wave = np.sin(0.5 * np.pi * frequency * t)
    return volume * wave

def generate_melody(frequencies, duration, samplerate, volume):
    """
    Generates a melodic line with random notes and instruments.
    Args:
        frequencies (dict): Dictionary of note frequencies to choose from.
        duration (float): Total duration of the melody in seconds.
        samplerate (int): Samples per second.
        volume (float): Volume of the melody.
    Returns:
        np.array: The generated melody audio.
    """
    pieces = []
    t_note = 0
    while t_note < duration:
        # Randomly choose note duration
        note_duration = np.random.choice([0.5, 1, 1.5, 2], p=[0.4, 0.3, 0.2, 0.1])
        # Ensure note doesn't exceed total duration
        if t_note + note_duration > duration:
            note_duration = duration - t_note
        # Randomly choose a note frequency
        note_freq = np.random.choice(list(frequencies.values()))
        # Randomly choose an instrument for the note
        instrument = np.random.choice(['piano', 'guitar', 'sine'], p=[0.6, 0.3, 0.1])
        # Generate the wave for the current note
        melody_piece = generate_instrument_wave(
            note_freq, note_duration, samplerate, volume, instrument
        )
        pieces.append(melody_piece)
        t_note += note_duration  # Advance time
    return np.concatenate(pieces) if pieces else np.array([], dtype=float)

def generate_bassline(frequencies, duration, samplerate, volume):
    """
    Generates a bassline with random notes.
    Args:
        frequencies (dict): Dictionary of bass note frequencies.
        duration (float): Total duration of the bassline.
        samplerate (int): Samples per second.
        volume (float): Volume of the bassline.
    Returns:
        np.array: The generated bassline audio.
    """
    pieces = []
    t_note = 0
    while t_note < duration:
        # Randomly choose note duration for bass (longer notes)
        note_duration = np.random.choice([1, 2], p=[0.8, 0.2])
        # Ensure note doesn't exceed total duration
        if t_note + note_duration > duration:
            note_duration = duration - t_note
        # Randomly choose a bass note frequency
        note_freq = np.random.choice(list(frequencies.values()))
        # Generate a sine wave for the bass note
        bass_piece = generate_sine_wave(note_freq, note_duration, samplerate, volume)
        pieces.append(bass_piece)
        t_note += note_duration  # Advance time
    return np.concatenate(pieces) if pieces else np.array([], dtype=float)

def generate_chord_progression(chord_map, chord_names, duration, samplerate, volume, current_note_frequencies):
    """
    Generates a chord progression based on a sequence of chord names.
    Args:
        chord_map (dict): Dictionary mapping chord names to lists of note names.
        chord_names (list): A list of chord names defining the progression.
        duration (float): Total duration of the progression.
        samplerate (int): Samples per second.
        volume (float): Volume of the chords.
        current_note_frequencies (dict): The currently transposed note frequencies.
    Returns:
        np.array: The generated chord progression audio.
    """
    pieces = []
    t_chord = 0
    while t_chord < duration:
        for name in chord_names:  # Iterate through the defined chord progression
            if t_chord >= duration:
                break  # Stop if duration is exceeded
            # Randomly choose chord duration (longer durations for chords)
            chord_duration = np.random.choice([4, 8], p=[0.5, 0.5])
            # Ensure chord doesn't exceed total duration
            if t_chord + chord_duration > duration:
                chord_duration = duration - t_chord

            chord_notes = chord_map.get(name, [])  # Get the notes for the current chord
            chord_wave = np.zeros(int(samplerate * chord_duration))  # Initialize chord wave
            for note_name in chord_notes:
                freq = current_note_frequencies.get(note_name, 0)  # Get the transposed frequency
                # Generate instrument wave for each note in the chord (using piano sound)
                note_wave = generate_instrument_wave(
                    freq, chord_duration, samplerate, volume, 'piano'
                )
                chord_wave += note_wave
            if chord_notes:
                chord_wave /= len(chord_notes)  # Normalize chord volume if notes exist
            pieces.append(chord_wave)
            t_chord += chord_duration  # Advance time

    return np.concatenate(pieces) if pieces else np.array([], dtype=float)

def generate_arpeggio_track(chord_map, chord_names, duration, samplerate, volume, current_note_frequencies, arpeggio_speed=0.25):
    """
    Generates an arpeggiated track based on the chord progression.
    Args:
        chord_map (dict): Dictionary mapping chord names to lists of note names.
        chord_names (list): A list of chord names defining the progression.
        duration (float): Total duration of the arpeggio track.
        samplerate (int): Samples per second.
        volume (float): Volume of the arpeggio.
        current_note_frequencies (dict): The currently transposed note frequencies.
        arpeggio_speed (float): Duration of each arpeggiated note in seconds.
    Returns:
        np.array: The generated arpeggio audio.
    """
    pieces = []
    t_arpeggio = 0
    while t_arpeggio < duration:
        for name in chord_names:
            if t_arpeggio >= duration:
                break
            chord_duration = np.random.choice([4, 8], p=[0.5, 0.5])  # Match chord duration
            if t_arpeggio + chord_duration > duration:
                chord_duration = duration - t_arpeggio

            chord_notes = chord_map.get(name, [])
            if not chord_notes:
                continue

            # Generate arpeggiated notes for the current chord
            current_chord_arpeggio = np.zeros(int(samplerate * chord_duration))
            notes_in_chord = [
                current_note_frequencies.get(note_name, 0)
                for note_name in chord_notes
                if current_note_frequencies.get(note_name, 0) != 0
            ]

            if not notes_in_chord:
                continue

            note_idx = 0
            note_start_time = 0
            while note_start_time < chord_duration:
                note_freq = notes_in_chord[note_idx % len(notes_in_chord)]
                note_wave = generate_instrument_wave(
                    note_freq,
                    arpeggio_speed,
                    samplerate,
                    volume * 0.8,
                    'piano',
                )

                start_sample = int(samplerate * note_start_time)
                end_sample = min(start_sample + len(note_wave), len(current_chord_arpeggio))
                current_chord_arpeggio[start_sample:end_sample] += note_wave[: end_sample - start_sample]

                note_start_time += arpeggio_speed
                note_idx += 1
            pieces.append(current_chord_arpeggio)
            t_arpeggio += chord_duration
    return np.concatenate(pieces) if pieces else np.array([], dtype=float)

def generate_pad_layer(chord_map, chord_names, duration, samplerate, volume, current_note_frequencies):
    """
    Generates a sustained pad layer following the chord progression.
    Args:
        chord_map (dict): Dictionary mapping chord names to lists of note names.
        chord_names (list): A list of chord names defining the progression.
        duration (float): Total duration of the pad layer.
        samplerate (int): Samples per second.
        volume (float): Volume of the pad.
        current_note_frequencies (dict): The currently transposed note frequencies.
    Returns:
        np.array: The generated pad layer audio.
    """
    pieces = []
    t_pad = 0
    while t_pad < duration:
        for name in chord_names:
            if t_pad >= duration:
                break
            # Use the same random chord duration as the main chord progression
            chord_duration = np.random.choice([4, 8], p=[0.5, 0.5])
            if t_pad + chord_duration > duration:
                chord_duration = duration - t_pad

            chord_notes = chord_map.get(name, [])
            if not chord_notes: continue

            # Generate a sustained blend of notes for the current chord
            current_chord_pad = np.zeros(int(samplerate * chord_duration))
            for note_name in chord_notes:
                freq = current_note_frequencies.get(note_name, 0)
                # Use a sine wave for a smooth synth sound, with a long attack/release
                # to simulate a pad. Adjust ADSR for a more sustained sound.
                # For a simple sine wave, we can just generate it and let the overall
                # pedalboard effects handle some of the 'pad' feel.
                # Let's create a custom ADSR for the pad here.
                pad_attack_time = chord_duration * 0.1 # 10% of chord duration
                pad_decay_time = chord_duration * 0.1 # 10%
                pad_sustain_level = 0.8
                pad_release_time = chord_duration * 0.2 # 20%

                t_note = np.linspace(0, chord_duration, int(samplerate * chord_duration), endpoint=False)
                note_wave = volume * 0.5 * np.sin(2 * np.pi * freq * t_note) # Base sine wave

                # Apply pad-specific ADSR envelope
                attack_samples = int(samplerate * pad_attack_time)
                decay_samples = int(samplerate * pad_decay_time)
                release_samples = int(samplerate * pad_release_time)
                sustain_samples = len(t_note) - attack_samples - decay_samples - release_samples

                if sustain_samples < 0:
                    sustain_samples = 0
                    total_samples = attack_samples + decay_samples + release_samples
                    if total_samples > 0:
                        attack_samples, decay_samples, release_samples = [
                            int(s * len(t_note) / total_samples) for s in [attack_samples, decay_samples, release_samples]
                        ]
                    sustain_samples = len(t_note) - attack_samples - decay_samples - release_samples
                    if sustain_samples < 0: sustain_samples = 0

                envelope = np.concatenate([
                    np.linspace(0, 1, attack_samples),
                    np.linspace(1, pad_sustain_level, decay_samples),
                    np.full(sustain_samples, pad_sustain_level),
                    np.linspace(pad_sustain_level, 0, release_samples)
                ])
                if len(envelope) < len(note_wave):
                    envelope = np.pad(envelope, (0, len(note_wave) - len(envelope)), 'constant')
                note_wave *= envelope[:len(note_wave)]

                current_chord_pad += note_wave

            if chord_notes:
                current_chord_pad /= len(chord_notes)  # Normalize
            pieces.append(current_chord_pad)
            t_pad += chord_duration
    return np.concatenate(pieces) if pieces else np.array([], dtype=float)


def generate_drum_beat(duration, samplerate, volume, base_bpm, bass_frequencies_dict):
    """Generates a simple drum beat with kick and snare sounds."""

    drum_track = np.zeros(int(samplerate * duration))

    bpm_variation = random.uniform(-5, 5)
    current_bpm = base_bpm + bpm_variation
    beat_duration = 60 / current_bpm
    samples_per_beat = int(samplerate * beat_duration)

    num_beats = int(duration / beat_duration)
    beat_indices = np.arange(num_beats)

    # Determine which beats are kicks and snares
    kick_mask = (beat_indices % 4 == 0) | (beat_indices % 4 == 2)
    snare_mask = ~kick_mask

    kick_starts = beat_indices[kick_mask] * samples_per_beat
    snare_starts = beat_indices[snare_mask] * samples_per_beat

    # --- Kick drum generation ---
    kick_duration = 0.2
    kick_samples = int(samplerate * kick_duration)
    kick_t = np.linspace(0, kick_duration, kick_samples, endpoint=False)
    kick_freqs = np.random.choice(list(bass_frequencies_dict.values()), len(kick_starts))
    kick_waves = volume * 0.8 * np.sin(2 * np.pi * kick_freqs[:, None] * kick_t)

    for start, wave in zip(kick_starts, kick_waves):
        end = min(start + kick_samples, len(drum_track))
        drum_track[start:end] += wave[: end - start]

    # --- Snare drum generation ---
    snare_duration = 0.15
    snare_samples = int(samplerate * snare_duration)
    snare_noise = np.random.uniform(-1, 1, (len(snare_starts), snare_samples))
    b = np.array([1, -0.9])
    snare_waves = volume * 0.5 * np.apply_along_axis(lambda x: np.convolve(x, b, mode="full")[: len(x)], 1, snare_noise)

    for start, wave in zip(snare_starts, snare_waves):
        end = min(start + snare_samples, len(drum_track))
        drum_track[start:end] += wave[: end - start]

    return drum_track

def generate_vinyl_noise(duration, samplerate, volume):
    """
    Generates vinyl crackle and pop noise.
    Args:
        duration (float): Total duration of the noise in seconds.
        samplerate (int): Samples per second.
        volume (float): Volume of the noise.
    Returns:
        np.array: The generated vinyl noise audio.
    """
    noise_track = np.zeros(int(samplerate * duration))
    total_samples = len(noise_track)

    # Generate continuous low-pass filtered white noise for crackle
    crackle_noise = np.random.uniform(-1, 1, total_samples)
    # Simple low-pass filter (moving average) for crackle
    window_size = 50 # Adjust for more or less muffled sound
    crackle_filtered = np.convolve(crackle_noise, np.ones(window_size)/window_size, mode='same')
    noise_track += crackle_filtered * volume * 0.1 # Lower volume for background crackle

    # Add occasional "pops"
    num_pops = int(duration / 2)  # Roughly one pop every two seconds

    pop_positions = np.random.randint(0, total_samples, size=num_pops)
    pop_durations = np.random.uniform(0.005, 0.02, size=num_pops)
    pop_samples = (samplerate * pop_durations).astype(int)
    pop_amplitudes = np.random.uniform(0.3, 0.7, size=num_pops)

    for pos, p_len, amp in zip(pop_positions, pop_samples, pop_amplitudes):
        wave = np.random.uniform(-amp, amp, p_len)
        wave *= np.linspace(1, 0, p_len)

        start = max(0, pos - p_len // 2)
        end = min(total_samples, start + p_len)
        add_len = min(end - start, len(wave))
        noise_track[start:start + add_len] += wave[:add_len]

    return noise_track


# --- Main Execution ---
def main():
    # 1. Select a random key and transpose note frequencies
    KEY = random.choice(list(key_to_transposition_factor.keys()))
    transposition_factor = key_to_transposition_factor[KEY]
    print(f"Generating music in key: {KEY}")
    # Transpose the base note and bass frequencies to the selected key
    transposed_note_frequencies = transpose(note_frequencies, transposition_factor)
    transposed_bass_frequencies = transpose(bass_frequencies, transposition_factor)

    # 2. Generate Audio Components
    print("Generating audio components...")
    # Generate melody, bassline, and chords with adjusted volumes
    melody = generate_melody(transposed_note_frequencies, DURATION, SAMPLERATE, VOLUME * 0.7)
    bassline = generate_bassline(transposed_bass_frequencies, DURATION, SAMPLERATE, VOLUME * 0.9)

    # Randomly select a chord progression
    chord_progression_names = random.choice(all_chord_progressions)
    print(f"Using chord progression: {chord_progression_names}")
    chords_audio = generate_chord_progression(
        chords, chord_progression_names, DURATION, SAMPLERATE, VOLUME * 0.5,
        current_note_frequencies=transposed_note_frequencies
    )
    # Generate arpeggio track
    arpeggio_track = generate_arpeggio_track(
        chords, chord_progression_names, DURATION, SAMPLERATE, VOLUME * 0.4,
        current_note_frequencies=transposed_note_frequencies
    )
    # Generate pad layer
    pad_layer = generate_pad_layer(
        chords, chord_progression_names, DURATION, SAMPLERATE, VOLUME * 0.3, # Adjust volume for pad
        current_note_frequencies=transposed_note_frequencies
    )
    # Pass transposed_bass_frequencies to generate_drum_beat for the kick drum frequency
    drum_beat = generate_drum_beat(DURATION, SAMPLERATE, VOLUME * 0.6, BASE_BPM, transposed_bass_frequencies)
    vinyl_noise = generate_vinyl_noise(DURATION, SAMPLERATE, VOLUME * 0.2) # Generate vinyl noise

    # 3. Mix Tracks
    print("Mixing tracks...")
    # Find the maximum length among all generated tracks
    max_len = max(len(melody), len(bassline), len(chords_audio), len(arpeggio_track), len(pad_layer), len(drum_beat), len(vinyl_noise))
    # Pad shorter tracks with zeros to match the longest track's length
    melody = np.pad(melody, (0, max_len - len(melody)))
    bassline = np.pad(bassline, (0, max_len - len(bassline)))
    chords_audio = np.pad(chords_audio, (0, max_len - len(chords_audio)))
    arpeggio_track = np.pad(arpeggio_track, (0, max_len - len(arpeggio_track)))
    pad_layer = np.pad(pad_layer, (0, max_len - len(pad_layer))) # Pad the new pad layer
    drum_beat = np.pad(drum_beat, (0, max_len - len(drum_beat)))
    vinyl_noise = np.pad(vinyl_noise, (0, max_len - len(vinyl_noise)))

    # Combine all tracks by summing their amplitudes
    track = melody + bassline + chords_audio + arpeggio_track + pad_layer + drum_beat + vinyl_noise

    # 4. Apply Lofi Effects with pedalboard
    print("Applying lofi effects...")
    # Initialize a Pedalboard with desired audio effects
    board = Pedalboard([
        Compressor(threshold_db=-10, ratio=4), # Compressor for dynamic range control
        LowpassFilter(cutoff_frequency_hz=2500), # Lowpass filter to simulate old recordings (muffled sound)
        Reverb(room_size=0.6, damping=0.8, wet_level=0.3, dry_level=0.7) # Reverb for spaciousness
    ])

    # Ensure the audio data is in float32 format before processing with pedalboard
    track_float32 = track.astype(np.float32)
    processed_track = board.process(track_float32, sample_rate=SAMPLERATE)

    # 5. Normalize and save the final track
    print("Normalizing and saving file...")
    # Find the maximum absolute amplitude for normalization
    max_val = np.max(np.abs(processed_track))
    if max_val == 0:
        final_track = processed_track # Avoid division by zero if track is silent
    else:
        # Normalize the track to prevent clipping and ensure consistent volume
        final_track = processed_track * (0.9 / max_val)

    # Define the output filename based on the selected key
    output_file = f'lofi_ambient_music_{KEY.replace("#", "s")}.wav'
    # Save the final processed track as a WAV file.
    # The data is already float32, but explicitly casting ensures compatibility.
    write(output_file, SAMPLERATE, final_track.astype(np.float32))

    print(f"\nSuccessfully generated lofi music file: {output_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n--- SCRIPT FAILED ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
