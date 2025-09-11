import numpy as np
import parselmouth

from audiochains.streams import InputStream
from audiochains.block_methods import UnpackRawInFloat32

from . import settings as s
from ..guui.profiles import get_audio


def analyze_voice():
    """
    Генератор (voice_status, pitch).
    voice_status: 1, если есть голос, 0 — нет голоса.
    pitch — частота, если есть (или последняя зафиксированная при недавнем голосе).
    """

    config = get_audio(s.profile_name)
    device_index = config.get("mic_device_index")
    print(f"device_index: {device_index}")
    with InputStream(samplerate=16000, blocksize=s.BLOCKSIZE, channels=1, sampwidth=2, device=device_index) as stream:
        stream.set_methods(UnpackRawInFloat32())
        last_valid_pitch = None
        silent_counter = 0

        while True:
            raw_data = stream.read(s.BLOCKSIZE)
            if not raw_data:
                continue

            signal = stream.chain_of_methods(raw_data)
            sound = parselmouth.Sound(values=signal, sampling_frequency=stream.samplerate)

            intensity = np.mean(sound.to_intensity().values) if sound.to_intensity().values.size > 0 else -50
            pitch_values = sound.to_pitch_ac(
                pitch_floor=s.PITCH_FLOOR,
                pitch_ceiling=s.PITCH_CEILING,
                voicing_threshold=s.VOICING_THRESHOLD
            ).selected_array['frequency']
            pitch_values[(pitch_values == 0) | (pitch_values > s.PITCH_CEILING)] = np.nan
            pitch = np.nanmean(pitch_values) if np.any(~np.isnan(pitch_values)) else last_valid_pitch

            if intensity > s.SILENCE_THRESHOLD_DB and pitch is not None:
                # Есть голос
                last_valid_pitch = pitch
                silent_counter = 0
                yield 1, pitch
            else:
                # Голоса нет (либо тихо, либо pitch ещё не появился)
                silent_counter += 1
                if last_valid_pitch is not None:
                    # Выдаём "1, last_valid_pitch" короткое время, если голос только что пропал
                    yield 1, last_valid_pitch
                #else:
                    # Если никогда не было pitch — совсем 0, None
                    #yield 0, None

                # Если некоторое число блоков подряд нет голоса — стабильно 0
                if silent_counter >= s.BLOCKS_TO_SILENT:
                    yield 0, last_valid_pitch
