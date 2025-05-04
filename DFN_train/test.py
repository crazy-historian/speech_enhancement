'''from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("BAAI/ChildMandarin", cache_dir='E:\mai\Diploma\cahce_dir')
print(ds)

print(ds['train'])

for i in ds['train']:
    print(i)
    exit()'''

#source_files = [
#r'E:\mai\Diploma\speech_enhancement\VAD_testing\Vino2.wav',
#r'E:\mai\Diploma\speech_enhancement\VAD_testing\vinograd2.wav' ,
#r'E:\mai\Speech\speech_enhancement\audios\ребенок_4_5.wav',
#r'E:\mai\Speech\speech_enhancement\audios\учитель_6.wav',
#r"E:\mai\Speech\speech_enhancement\audios\ребенок_4.wav",
#r"E:\mai\Diploma\speech_enhancement\DPipe\data\background_noise\knocking\knocking.wav",
#r"E:\mai\Diploma\speech_enhancement\DPipe\data\background_noise\breath\breath.wav",
#]

import os

source_files = [r'E:\mai\Diploma\speech_enhancement\VAD_testing\noisy_files' +'\\' + i for i in os.listdir(r'E:\mai\Diploma\speech_enhancement\VAD_testing\noisy_files')]

denoisers_common = r'E:\mai\Diploma\fd_ckpt' 

denoisers_unique = ['100epo', '75epo', '50epo', '25epo']

for suffix in denoisers_unique:
    for source_file in source_files:
        cmnd = f'python DeepFilterNet/df/enhance.py {source_file}' + r' --output-dir E:\mai\Diploma\speech_enhancement\VAD_testing\noisy_files_donoised '+ f'-m {denoisers_common}\{suffix}'
        print(cmnd)
#python DeepFilterNet/df/enhance.py E:\mai\Diploma\speech_enhancement\DPipe\data\background_noise\knocking\knocking.wav --output-dir E:\mai\Diploma\speech_enhancement\VAD_testing\df_my\75 -m E:\mai\Diploma\fd_ckpt

folder_path = r'E:\mai\Diploma\speech_enhancement\VAD_testing\noisy_files_donoised'
files = os.listdir(folder_path)

# Оставим только файлы (без папок)
file_names = set([f for f in files if os.path.isfile(os.path.join(folder_path, f))])
to_pr = set([name.split('.wav')[0]+'_100epo.wav' for name in source_files])

print(len(to_pr - file_names))

exit()
#source_files = [r'E:\mai\Diploma\speech_enhancement\VAD_testing\noisy_files' +'\\' + i for i in os.listdir(r'E:\mai\Diploma\speech_enhancement\VAD_testing\noisy_files')]

noise_types = ['breath', 'knocking', 'gaussian', 'uniform']
dn_types = ['100epo', '75epo', '50epo', '25epo']
for init_name in os.listdir(r'E:\mai\Diploma\speech_enhancement\VAD_testing\noisy_files'):
    for noise_type in noise_types:
        for dn_type in dn_types:
