# Example diagnostic script
from batcher.downstream_dataset import EEGDatasetCls


dataset = EEGDatasetCls('../train')
for i in range(len(dataset)):
    data, label = dataset[i]
    # Check and print the types and shapes
    print(f"Data Type: {data.type}, Label Type: {label.dtype}")
    print(f"Data Shape: {data.shape}, Label Shape: {label.shape}")
