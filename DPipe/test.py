import os
import time
import torch
from df.enhance import enhance, init_df, save_audio
from df.utils import download_file  # Only needed if you want to download files; not used here.

if __name__ == "__main__":
    # Load the default DeepFilterNet model.
    model, df_state, _ = init_df()

    # Specify the directory where you want to save the audio files.
    output_dir = "E:\mai\Diploma\speech_enhancement\DPipe\data\output_dir"
    os.makedirs(output_dir, exist_ok=True)

    # Создаём случайный входной аудиосигнал в виде тензора с 1024 сэмплами и добавляем размерность канала
    input_audio = torch.rand(1024)  # shape: [1024]
    if input_audio.dim() == 1:
        input_audio = input_audio.unsqueeze(0)  # shape становится [1, 1024]

    # Замеряем время работы модели
    start_time = time.time()
    enhanced_audio = enhance(model, df_state, input_audio)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Время работы модели: {elapsed_time:.4f} секунд")

    # Приводим тензоры к формату numpy для сохранения (удаляем размерность канала, если необходимо)
    if isinstance(enhanced_audio, torch.Tensor):
        enhanced_audio = enhanced_audio.squeeze().cpu().numpy()
    input_audio_np = input_audio.squeeze().cpu().numpy()

    # Получаем частоту дискретизации из состояния модели
    sample_rate = df_state.sr()

    # Сохраняем входной и улучшенный аудиофайлы
    save_audio(os.path.join(output_dir, "input.wav"), input_audio_np, sample_rate)
    save_audio(os.path.join(output_dir, "enhanced.wav"), enhanced_audio, sample_rate)