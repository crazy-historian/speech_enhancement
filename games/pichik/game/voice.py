import time
import numpy as np
import parselmouth

from collections import deque

from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
from . import settings as s
from ..guui.profiles import get_audio

def analyze_voice(window_size=3):
    config = get_audio(s.profile_name)
    device_index = config.get("mic_device_index")
    print(f"device_index: {device_index}")

    # Очередь для хранения последних N значений интенсивности
    smoothing_window = deque(maxlen=window_size)
    
    
    with InputStream(
        samplerate=16000,
        blocksize=s.BLOCKSIZE,
        channels=1,
        sampwidth=2,
        device=device_index
    ) as stream:
        stream.set_methods(UnpackRawInFloat32())
        silent_counter = 0
        start_time = time.time()
        last_valid_pitch = None  
        smoothing_window = deque(maxlen=window_size)  # окно сглаживания

        while True:
            raw_data = stream.read(s.BLOCKSIZE)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50

            pitch_obj = sound.to_pitch_ac(
                time_step=0.01,
                pitch_floor=s.PITCH_FLOOR,
                pitch_ceiling=s.PITCH_CEILING,
                voicing_threshold=0.6
            )
            pitch_values = pitch_obj.selected_array["frequency"]
            pitch_values[(pitch_values == 0) | (pitch_values > 600)] = np.nan
            avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else last_valid_pitch

            above_silence_threshold = avg_intensity > s.SILENCE_THRESHOLD_DB
            valid_pitch = avg_pitch is not None

            if above_silence_threshold and valid_pitch:
                silent_counter = 0
                last_valid_pitch = avg_pitch
                smoothing_window.append(avg_pitch)
                yield np.mean(smoothing_window)
            else:
                silent_counter += 1
                if silent_counter < s.BLOCKS_TO_SILENT and last_valid_pitch is not None:
                    smoothing_window.append(last_valid_pitch)
                    yield np.mean(smoothing_window)
                else:
                    yield None
