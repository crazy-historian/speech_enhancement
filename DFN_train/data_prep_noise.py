'''from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("Sunbird/urban-noise", "small", cache_dir='E:\mai\Diploma\cahce_dir')
print(ds)
print(set(ds['train']['class']))
ds_t = ds['train']
for i in ds_t:
    print(i)
    exit()'''
#print(set(ds['train'][0]['audio']))

# export_urban_noise_to_wav.py
from datasets import load_dataset
import soundfile as sf
import librosa, os
from tqdm import tqdm

DATASET_NAME   = "Sunbird/urban-noise"
CONFIG         = "small"          # or "large"
SPLIT          = "train"          # Sunbird only has one split
TARGET_SR      = 16_000

#OUT_DIR        = r"E:\mai\Diploma\speech_enhancement\DFN_train\noise_files_wav"
#LIST_PATH      = r"E:\mai\Diploma\speech_enhancement\DFN_train\noise_files.txt"


ds = load_dataset(DATASET_NAME, CONFIG, split=SPLIT)  # cached after first call
ds = ds.class_encode_column("class_id") 
split = ds.train_test_split(
        test_size=0.2,
        shuffle=True,
        seed=42,
        stratify_by_column="class_id"
    )

train_ds, test_ds = split["train"], split["test"]

paths = [
    (train_ds,
     r'E:\mai\Diploma\speech_enhancement\DFN_train\data\noise_train',
     r'E:\mai\Diploma\speech_enhancement\DFN_train\data\noise_train_files.txt'),
     (test_ds,
     r'E:\mai\Diploma\speech_enhancement\DFN_train\data\noise_test',
     r'E:\mai\Diploma\speech_enhancement\DFN_train\data\noise_test_files.txt')
]

for ds, out_dir, list_path in paths:
    os.makedirs(out_dir, exist_ok=True)

    with open(list_path, "w") as list_file:
        for idx, sample in tqdm(enumerate(ds), total=len(ds)):
            audio = sample["audio"]
            y, sr = audio["array"], audio["sampling_rate"]  # Sunbird ships 16 kHz Ogg
            if sr != TARGET_SR:
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
                sr = TARGET_SR
            wav_path = os.path.join(
                out_dir, f"{idx:05d}_{sample['class'].replace(' ', '_')}.wav"
            )
            sf.write(wav_path, y, sr, subtype="PCM_16")
            list_file.write(wav_path + "\n")
