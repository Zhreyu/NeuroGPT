import tensorflow as tf
from transformers import TFGPT2Model

class EEGEncoder(tf.keras.Model):
    def __init__(self, num_channels, transformer_model_name='gpt2', num_attention_heads=10, embedding_dim=1080):
        super(EEGEncoder, self).__init__()
        # Convolutional Layers
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 25), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(num_channels, 1), activation='relu')
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))
        self.flatten = tf.keras.layers.Flatten()

        # Transformer Model (GPT-2)
        self.transformer = TFGPT2Model.from_pretrained(transformer_model_name, num_hidden_layers=6, num_attention_heads=num_attention_heads, hidden_size=embedding_dim)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.flatten(x)
        transformer_output = self.transformer(x)[0]
        return transformer_output

# Example usage
num_channels = 22  # Number of EEG channels (adjust as per your data)
eeg_encoder = EEGEncoder(num_channels=num_channels)
