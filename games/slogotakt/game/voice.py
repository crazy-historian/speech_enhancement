from typing import Optional  # ← добавили
import time
import numpy as np
import parselmouth

from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

from . import settings as s

def analyze_voice(game_duration: float,
                  silence_threshold_db: float,
                  blocks_to_silent: int,
                  mic_device_index: Optional[int] = None):  # ← было: int | None
    """
    Генератор голосовой активности: 1 (голос) / 0 (тишина).
    """
    with InputStream(
        samplerate=16000,
        blocksize=s.BLOCKSIZE,
        channels=1,
        sampwidth=2,
        device=mic_device_index
    ) as stream:
        stream.set_methods(UnpackRawInFloat32())
        silent_counter = 0
        start_time = time.time()

        while time.time() - start_time < game_duration:
            raw_data = stream.read(s.BLOCKSIZE)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            intensity_values = sound.to_intensity().values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50

            pitch_obj = sound.to_pitch_ac(
                time_step=0.01,
                pitch_floor=s.PITCH_FLOOR,
                pitch_ceiling=s.PITCH_CEILING,
                voicing_threshold=s.VOICING_THRESHOLD,
                octave_cost=0.01,
                octave_jump_cost=0.35,
                voiced_unvoiced_cost=0.14,
            )
            pitch_values = pitch_obj.selected_array["frequency"]
            pitch_values[(pitch_values == 0) | (pitch_values > s.PITCH_CEILING)] = np.nan
            avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else np.nan

            above_silence_threshold = avg_intensity > silence_threshold_db
            valid_pitch = (avg_pitch > s.PITCH_FLOOR)

            if above_silence_threshold and valid_pitch:
                silent_counter = 0
                yield 1
            else:
                silent_counter += 1
                if silent_counter >= max(1, blocks_to_silent):
                    yield 0
