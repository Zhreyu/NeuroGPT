# Import the EEGDataset class from your module
from downstream_dataset import EEGDatasetCls as EEGDataset

# Specify the directory containing EEG data files
folder_path = 'src/BCICIV_2a_gdf'

# Create an instance of the EEGDataset class
eeg_dataset = EEGDataset(folder_path)

# Print the output for each item in the dataset
for i in range(len(eeg_dataset)):
    data_item = eeg_dataset[i]
    print(f"Data item {i}:")
    print(data_item)
    print("-------------------------------")
