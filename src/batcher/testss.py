from multiprocessing import freeze_support
from base import EEGDataset
import os
import time
import pandas as pd
from torch.utils.data import DataLoader




# Main function to load data and measure iteration time
def read_threshold_sub(csv_file, lower_bound=2599, upper_bound=1000000):
    df_read = pd.read_csv(csv_file)
    # Access the list of filenames and time_len
    filenames = df_read['filename'].tolist()
    time_lens = df_read['time_len'].tolist()
    filtered_files = []
    for fn, tlen in zip(filenames, time_lens):
        if (tlen > lower_bound) and (tlen < upper_bound):
            filtered_files.append(fn)
    return filtered_files
root_path = 'C:\\Users\\shreyas\\Documents\\GitHub\\Archives\\NeuroGPT\\tuh_tensors\\'
files = read_threshold_sub('C:\\Users\\shreyas\\Documents\\GitHub\\Archives\\NeuroGPT\\inputs\\sub_list2.csv', lower_bound=1000, upper_bound=1000000)
# Create the updated dataset
dataset = EEGDataset(filenames=files,sample_keys=[
            'inputs',
            'attention_mask'
        ],root_path=root_path)

# Create the DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True,pin_memory=True)
num_batches = 0
# Measure time for specified number of batches
start_time = time.time()
for i, batch in enumerate(dataloader):
    num_batches += 1

end_time = time.time()
time_taken = end_time - start_time

print(f'Time taken to iterate through {num_batches} batches: {time_taken:.2f} seconds')

