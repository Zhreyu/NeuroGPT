import torch

# Load the tensor file
tensor_file = "tuh_tensors/aaaaaqtl_s001_t000_1_raw_eeg.pt"  # Replace "your_tensor_file.pth" with the path to your tensor file
loaded_tensor = torch.load(tensor_file)

# Check the size of the loaded tensor
print("Size of the loaded tensor:", loaded_tensor.size())
