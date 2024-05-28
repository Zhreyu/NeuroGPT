import torch

# Replace 'path/to/your/training_args.bin' with the actual path to your .bin file
file_path = 'results\\models\\upstream\\GPT_lrs-6_hds-16_ChunkLen-512_NumChunks-34_ovlp-51_2024-03-22_22\\model_final\\training_args.bin'

# Load the file
training_args = torch.load(file_path)

# Print the loaded object
print(training_args)
