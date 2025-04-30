from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("BAAI/ChildMandarin", cache_dir='E:\mai\Diploma\cahce_dir')
print(ds)

print(ds['train'])

for i in ds['train']:
    print(i)
    exit()

