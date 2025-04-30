
from datasets import load_dataset
import soundfile as sf
import librosa, os
from tqdm import tqdm

TARGET_SR      = 16_000

#OUT_DIR        = r"E:\mai\Diploma\speech_enhancement\DFN_train\noise_files_wav"
#LIST_PATH      = r"E:\mai\Diploma\speech_enhancement\DFN_train\noise_files.txt"


ds = load_dataset("BAAI/ChildMandarin", cache_dir='E:\mai\Diploma\cahce_dir')  # cached after first call

train_ds, test_ds = ds["train"].select(range(4000)), ds["test"].select(range(1000))

paths = [
    (train_ds,
    r'E:\mai\Diploma\speech_enhancement\DFN_train\data\speech_train',
    r'E:\mai\Diploma\speech_enhancement\DFN_train\data\speech_train_files.txt'),

    (test_ds,
    r'E:\mai\Diploma\speech_enhancement\DFN_train\data\speech_test',
    r'E:\mai\Diploma\speech_enhancement\DFN_train\data\speech_test_files.txt')
]

for ds, out_dir, list_path in paths:
    os.makedirs(out_dir, exist_ok=True)

    with open(list_path, "w") as list_file:
        for idx, sample in tqdm(enumerate(ds), total=len(ds)):
            
            audio = sample["wav"]
            y, sr = audio["array"], audio["sampling_rate"] 
            if sr != TARGET_SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
                sr = TARGET_SR
            wav_path = os.path.join(
                out_dir, f"{idx:05d}_.wav"
            )
            sf.write(wav_path, y, sr, subtype="PCM_16")
            list_file.write(wav_path + "\n")
