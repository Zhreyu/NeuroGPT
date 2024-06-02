import torch

# Dummy class to match the expected one during unpickling
class AcceleratorConfig:
    pass

# Manually adding the dummy class to the expected module
import transformers.trainer_pt_utils
transformers.trainer_pt_utils.AcceleratorConfig = AcceleratorConfig

# Load the training arguments from the file
training_args_path = 'src\\results\\models\\upstream\\dst_our-0\\model_final\\training_args.bin'
training_args = torch.load(training_args_path)

# Print the contents of the loaded training arguments
print(training_args)
