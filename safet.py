from safetensors import safe_open

# Open the safetensors file and read its contents
with safe_open("src/model.safetensors", framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"Tensor Name: {key}, Shape: {tensor.shape}, Dtype: {tensor.dtype}")

