# Import the EEGDataset class from your module
from src.batcher.downstream_dataset import EEGDatasetCls as EEGDataset

# Specify the directory containing EEG data files
folder_path = 'test'

# Create an instance of the EEGDataset class
eeg_dataset = EEGDataset(folder_path)

# Print the output for each item in the dataset
for i in range(5):
    data_item = eeg_dataset[i]
    print(f"Data item {i}:")
    print('Inputs: ',data_item['inputs'].shape)
    print('Labels: ',data_item['labels'])
    print("-------------------------------")
