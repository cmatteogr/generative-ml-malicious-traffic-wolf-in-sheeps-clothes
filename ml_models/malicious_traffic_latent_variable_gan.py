"""

"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), # Input layer (noise)
            nn.ReLU(),
            nn.Linear(128, 256), # Hidden layer
            nn.ReLU(),
            nn.Linear(256, output_dim), # Output layer (generated data)
            nn.Tanh() # Activation function for image generation
        )

    def forward(self, x):
        return self.model(x)

# Example usage:
noise = torch.randn((1, 100)) # 100-dimensional random noise
gen = Generator(100, 784) # 784 = 28x28 pixels (for image generation)
generated_data = gen(noise)
print(generated_data.shape)
