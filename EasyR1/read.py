from datasets import load_dataset

# Path to the specific .arrow file
local_file = "/project/airesearch/haolin/EasyR1/geometry3k/geometry3k-train.arrow"

print(f"Loading local file into a dataset: {local_file}")

# Use load_dataset, specify the format ('arrow'), and provide the local file path
# We map 'train' to our file. You could also map 'test', etc.
ds = load_dataset("arrow", data_files={"train": local_file})

print("\n[Dataset Structure]")
print(ds)

print("\n[First 5 Samples from 'train' split]")
print(ds['train'][:5])