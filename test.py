import pandas as pd
import tensorflow as tf
from transformers import TFGPT2Model
import numpy as np



file_path = 'Data1.csv'
eeg_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(eeg_data.head())


# Assuming `eeg_data` is a pandas DataFrame of your EEG data...

# Select data for one sample (e.g., trial 0)
sample_data = eeg_data[eeg_data['trial number'] == 0]

# Normalize sensor values
sensor_values = sample_data['sensor value'].values
normalized_sensor_values = (sensor_values - np.mean(sensor_values)) / np.std(sensor_values)

# Reshape data to match the input shape of the model
# This is an example; you'll need to adjust the shape according to your specific model's input requirements
num_channels = len(sample_data['sensor position'].unique())
sample_length = len(sample_data['sample num'].unique())
# Ensure the total number of sensor values matches num_channels * sample_length
assert len(normalized_sensor_values) == num_channels * sample_length

# Reshape the normalized sensor values
eeg_sample_tensor = tf.reshape(normalized_sensor_values, (1, num_channels, sample_length))

# encoder_output = eeg_encoder(eeg_sample_tensor)



class EEGEncoder(tf.keras.Model):
    def __init__(self, num_channels, sample_length, transformer_model_name='gpt2', num_attention_heads=12, embedding_dim=768):
        super(EEGEncoder, self).__init__()
        # Convolutional Layers
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 25), activation='relu', input_shape=(1, num_channels, sample_length, 1))
        self.conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(num_channels, 1), activation='relu')
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))
        self.flatten = tf.keras.layers.Flatten()

        # Reshape for transformer compatibility - ensure the sequence length is set correctly
        self.reshape = tf.keras.layers.Reshape((sample_length // (2 * 25), 128 * num_channels)) # Adjust the 128 to match the number of filters in the last conv layer

        # Transformer Model (GPT-2)
        self.transformer = TFGPT2Model.from_pretrained(transformer_model_name, num_attention_heads=num_attention_heads, hidden_size=embedding_dim)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.reshape(x)  # Reshape for transformer compatibility
        
        # Process through GPT-2 (or similar) model
        transformer_output = self.transformer(inputs=x, return_dict=True).last_hidden_state
        return transformer_output

# Example usage
num_channels = 64  # Number of EEG channels (adjust as per your data)
sample_length = 256  # Number of time points in each sample/epoch (adjust as per your data)

# Instantiate the EEGEncoder with the appropriate number of channels and sample length
eeg_encoder = EEGEncoder(num_channels=num_channels, sample_length=sample_length)

# Load your preprocessed EEG data tensor, ensuring it is in the correct shape
# eeg_sample_tensor = ...

# Use the encoder to predict the embeddings for your EEG data sample
embeddings = eeg_encoder(eeg_sample_tensor)

# embeddings now contains the embeddings produced by the encoder from your EEG data
print(embeddings)
