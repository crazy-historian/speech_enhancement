import time
import numpy as np
import parselmouth

from collections import deque

from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

from guigromik import load_config
from game.settings import BLOCKSIZE, GAME_DURATION


def analyze_voice(window_size=3):
    config = load_config()
    device_index = config.get("mic_device_index")
    print(f"device_index: {device_index}")

    # Очередь для хранения последних N значений интенсивности
    smoothing_window = deque(maxlen=window_size)

    with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2, device=device_index) as stream:
        stream.set_methods(UnpackRawInFloat32())
        start_time = time.time()
        while time.time() - start_time < GAME_DURATION:
            raw_data = stream.read(BLOCKSIZE)
            if not raw_data:
                continue
            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            intensity_obj = sound.to_intensity()
            intensity_values = intensity_obj.values.T.flatten()
            avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else 0

            smoothing_window.append(avg_intensity)
            smoothed = np.mean(smoothing_window)
            yield smoothed