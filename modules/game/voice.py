import threading
import numpy as np
import parselmouth
from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32
from game.settings import *


def analyze_voice(callback):
    def _listen():
        with InputStream(samplerate=16000, blocksize=BLOCKSIZE, channels=1, sampwidth=2) as stream:
            stream.set_methods(UnpackRawInFloat32())
            silent_counter = 0
            while True:
                raw_data = stream.read(BLOCKSIZE)
                if not raw_data:
                    continue
                signal = stream.chain_of_methods(raw_data)
                sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

                intensity_obj = sound.to_intensity()
                intensity_values = intensity_obj.values.T.flatten()
                avg_intensity = np.mean(intensity_values) if len(intensity_values) > 0 else -50

                pitch_obj = sound.to_pitch_ac(
                    time_step=0.01,
                    pitch_floor=PITCH_FLOOR,
                    pitch_ceiling=PITCH_CEILING,
                    voicing_threshold=VOICING_THRESHOLD
                )
                pitch_values = pitch_obj.selected_array['frequency']
                pitch_values[(pitch_values == 0) | (pitch_values > PITCH_CEILING)] = np.nan
                avg_pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else np.nan

                above_silence_threshold = avg_intensity > SILENCE_THRESHOLD_DB
                valid_pitch = (avg_pitch > PITCH_FLOOR)
                if above_silence_threshold and valid_pitch:
                    silent_counter = 0
                    callback(1)
                else:
                    silent_counter += 1
                    if silent_counter >= BLOCKS_TO_SILENT:
                        callback(0)
    threading.Thread(target=_listen, daemon=True).start()