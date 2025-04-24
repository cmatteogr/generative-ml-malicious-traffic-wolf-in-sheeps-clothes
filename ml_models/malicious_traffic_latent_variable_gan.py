"""
VAE for Continuous Data using Gaussian Likelihood (Implicit Fixed Variance via MSE)
"""
import math # Use math.pi
import torch
import torch.nn as nn

from pytorch_model_summary import summary


# Use math.pi for scalar, torch.pi for tensors if needed later
PI = torch.tensor(math.pi)
EPS = 1.e-5 # Small epsilon for numerical stability

# --- Log Probability Functions ---
# log_categorical is removed as it's no longer needed

def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    """Calculates log probability log N(x|mu, diag(exp(log_var))) of diagonal Gaussian."""
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    # Sum over feature dimension D by default (usually dim=1)
    if dim is None:
        dim = 1 # Assume feature dimension is 1
    if reduction == 'sum':
        return torch.sum(log_p, dim)
    elif reduction == 'avg':
         # Average over batch *and* dimensions if dim is sequence
        if isinstance(dim, (list, tuple)):
             return torch.mean(log_p)
        # Average over specified dimension (usually feature dim)
        return torch.mean(log_p, dim)
    else: # No reduction (return per-element log prob)
        return log_p


def log_standard_normal(x, reduction=None, dim=None):
    """Calculates log probability log N(x|0, I) of standard Gaussian."""
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
    # Sum over feature dimension L by default (usually dim=1)
    if dim is None:
        dim = 1 # Assume feature dimension is 1
    if reduction == 'sum':
        return torch.sum(log_p, dim)
    elif reduction == 'avg':
        if isinstance(dim, (list, tuple)):
             return torch.mean(log_p)
        return torch.mean(log_p, dim)
    else: # No reduction
        return log_p

# --- Encoder ---

class Encoder(nn.Module):
    """Encodes input x into latent space parameters (mu, log_var)."""
    def __init__(self, encoder_net):
        super(Encoder, self).__init__()
        self.encoder = encoder_net

    @staticmethod
    def reparameterization(mu, log_var):
        """Applies the reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std) # Sample epsilon from standard normal
        return mu + std * eps

    def encode(self, x):
        """Passes input through the encoder network to get mu_e and log_var_e."""
        h_e = self.encoder(x)
        # Split the output into mean and log variance
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)
        return mu_e, log_var_e

    def sample(self, x=None, mu_e=None, log_var_e=None):
        """Samples z from the latent distribution q(z|x) = N(z|mu_e, exp(log_var_e))."""
        if (mu_e is None) and (log_var_e is None):
            if x is None:
                 raise ValueError('Either x or mu_e/log_var_e must be provided.')
            mu_e, log_var_e = self.encode(x)
        elif (mu_e is None) or (log_var_e is None):
             raise ValueError('Both mu_e and log_var_e must be provided if x is None.')

        z = self.reparameterization(mu_e, log_var_e)
        return z

    def log_prob(self, z, mu_e, log_var_e):
        """Calculates log probability log q(z|x) = log N(z|mu_e, exp(log_var_e))."""
        if (mu_e is None) or (log_var_e is None) or (z is None):
            raise ValueError('mu_e, log_var_e, and z cannot be None!')
        # Calculate log prob using Gaussian log likelihood, sum over latent dims (L)
        return log_normal_diag(z, mu_e, log_var_e, reduction='sum', dim=1)

# --- Decoder ---

class Decoder(nn.Module):
    """Decodes latent variable z into reconstructed data parameters (Gaussian mean)."""
    def __init__(self, decoder_net):
        super(Decoder, self).__init__()
        self.decoder = decoder_net
        # We implicitly assume p(x|z) is Gaussian N(x | mu=decoder(z), sigma^2*I)
        # where sigma is fixed. Often sigma=1 is assumed when using MSE loss.

    def decode(self, z):
        """Passes latent variable through the decoder network to get reconstruction mean."""
        # Output is the mean of the Gaussian p(x|z)
        mu_d = self.decoder(z)
        # Apply sigmoid if data is normalized to [0, 1] (e.g., images)
        # Or tanh if data is normalized to [-1, 1]
        # Or no activation if data is standardized (zero mean, unit variance)
        # Let's assume data is standardized or raw for now (no activation)
        # If your data is e.g. images in [0,1], uncomment the next line:
        # mu_d = torch.sigmoid(mu_d)
        return mu_d # Return the predicted mean

    def sample(self, z):
        """Generates the most likely reconstruction x' = E[p(x|z)] = mu_d."""
        # For a Gaussian N(mu, sigma^2*I), the mean mu is the most likely value.
        # If you wanted to sample *from* p(x|z), you'd need to assume/predict
        # a variance and add noise: mu_d + sigma * torch.randn_like(mu_d)
        mu_d = self.decode(z)
        return mu_d

    def log_prob(self, x, z):
        """Calculates reconstruction loss term.
        Using negative Mean Squared Error which is proportional to log N(x|mu_d, fixed_variance).
        Returns a value per batch item (reduction='none' essentially before sum).
        """
        mu_d = self.decode(z)
        # Calculate MSE loss per batch element, summing over feature dimension D
        # MSE = mean((x - mu_d)^2) over D. We want sum over D.
        # So, sum((x - mu_d)^2) over D.
        # The ELBO wants log p(x|z). For N(x|mu, sigma^2*I), log p = -0.5*log(2*pi*sigma^2)*D - 0.5/sigma^2 * sum((x-mu)^2)
        # If sigma=1, log p = C - 0.5 * sum((x-mu)^2).
        # Maximizing log p is equivalent to minimizing sum((x-mu)^2).
        # We return a value ~ log p(x|z) per batch item.
        # Let's return -sum((x - mu_d)**2) which is proportional to log p(x|z) up to constants.
        neg_sum_sq_error = -torch.sum((x - mu_d)**2, dim=1) # Sum over feature dimension D
        return neg_sum_sq_error


# --- Prior ---

class Prior(nn.Module):
    """Standard Gaussian Prior p(z) = N(z|0, I)."""
    def __init__(self, L):
        super(Prior, self).__init__()
        self.L = L # Latent dimension

    def sample(self, batch_size):
        """Samples z ~ N(0, I) of shape (batch_size, L)."""
        # Ensure device consistency if using GPU
        # A simple way to get device if the module has no parameters yet
        # is to register a dummy buffer.
        self.register_buffer('dummy_buffer', torch.tensor(0))
        device = self.dummy_buffer.device
        z = torch.randn((batch_size, self.L), device=device)
        return z

    def log_prob(self, z):
        """Calculates log probability log p(z) = log N(z|0, I)."""
        # Sum log prob over latent dimensions L
        return log_standard_normal(z, reduction='sum', dim=1)

# --- VAE ---

class VAE(nn.Module):
    """Variational Autoencoder (VAE) model for Continuous Data."""
    def __init__(self, encoder_net, decoder_net, L=16):
        super(VAE, self).__init__()

        print('VAE (Continuous Likelihood - Gaussian Mean Decoder) Initialized.')

        self.encoder = Encoder(encoder_net=encoder_net)
        self.decoder = Decoder(decoder_net=decoder_net)
        self.prior = Prior(L=L)

        self.L = L # Store latent dim if needed elsewhere

    def forward(self, x, reduction='avg'):
        """Calculates the negative Evidence Lower Bound (ELBO) loss.
           loss = -ELBO = Reconstruction Loss + KL Divergence
                 = -E_q[log p(x|z)] + KL(q(z|x) || p(z))
        """
        # Encode: Get q(z|x) parameters: mu_e, log_var_e
        mu_e, log_var_e = self.encoder.encode(x)
        # Sample z ~ q(z|x) using reparameterization trick
        z = self.encoder.reparameterization(mu_e, log_var_e)

        # Calculate ELBO components (per batch item)
        # Reconstruction Term: E_q[log p(x|z)]
        # Approximated using single sample z: log p(x|z)
        # We use -SumSqError which is proportional to log p(x|z) for fixed variance Gaussian
        log_px_given_z = self.decoder.log_prob(x, z) # Shape: (batch_size,)

        # KL Divergence Term: KL(q(z|x) || p(z))
        # KL = E_q[log q(z|x) - log p(z)]
        log_qzx = self.encoder.log_prob(z, mu_e, log_var_e) # Shape: (batch_size,)
        log_pz = self.prior.log_prob(z)                   # Shape: (batch_size,)
        KL = log_qzx - log_pz                             # Shape: (batch_size,)

        # ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
        # elbo = log_px_given_z - KL # Shape: (batch_size,)
        # We want to maximize ELBO, which means minimizing -ELBO
        # -ELBO = KL - log_px_given_z
        # If log_px_given_z = -SumSqError, then -ELBO = KL + SumSqError
        # This matches the common VAE loss: Reconstruction Loss + KL Divergence
        # Let's define Reconstruction Loss = -log_px_given_z
        reconstruction_loss = -log_px_given_z # Shape: (batch_size,)
        neg_elbo = reconstruction_loss + KL   # Shape: (batch_size,)

        # Return average or sum of negative ELBO (loss to be minimized)
        if reduction == 'sum':
            return neg_elbo.sum()
        else: # Default 'avg'
            return neg_elbo.mean()

    def sample(self, batch_size=64):
        """Generates new samples x' by sampling z ~ p(z) and decoding."""
        # Sample z from prior p(z) = N(0, I)
        z = self.prior.sample(batch_size=batch_size)
        # Decode z to get the mean reconstruction x' = E[p(x|z)]
        return self.decoder.sample(z)


# --- Example Usage ---
D = 784  # input dimension (e.g., flattened MNIST 28x28)
L = 20   # number of latents
M = 400  # hidden layer dimension

lr = 1e-3 # learning rate
num_epochs = 100 # max. number of epochs
# max_patience = 20 # early stopping patience (implement in training loop)

# Define Encoder Network
encoder_net = nn.Sequential(
    nn.Linear(D, M),
    nn.ReLU(), # Changed to ReLU, common choice
    # nn.Linear(M, M), # Optional extra layer
    # nn.ReLU(),
    nn.Linear(M, 2 * L) # Output mu and log_var
)

# Define Decoder Network
# Output size must be D for continuous data (predicting the mean)
decoder_net = nn.Sequential(
    nn.Linear(L, M),
    nn.ReLU(),
    # nn.Linear(M, M), # Optional extra layer
    # nn.ReLU(),
    nn.Linear(M, D)
    # Add Sigmoid here if input data is normalized to [0, 1]
    # nn.Sigmoid()
)

# Instantiate the VAE (Continuous)
model = VAE(encoder_net=encoder_net, decoder_net=decoder_net, L=L)

# Print the summary
print("\n--- Model Summary ---")
# Use a dummy input tensor with the correct shape
# Assuming input is (batch_size, D)
dummy_input = torch.randn(1, D) # Use randn for continuous data
print("ENCODER:\n", summary(encoder_net, dummy_input, show_input=False, show_hierarchical=False))

dummy_latent = torch.randn(1, L)
print("\nDECODER:\n", summary(decoder_net, dummy_latent, show_input=False, show_hierarchical=False))

# Example forward pass (requires actual data)
# Create dummy continuous data (e.g., normalized between 0 and 1)
dummy_data = torch.rand(4, D) # Example batch of continuous data [0, 1]
# If using sigmoid in decoder, data should be in [0,1].
# If no activation, data could be standardized (e.g., N(0,1)).
loss = model(dummy_data)
print(f"\nExample Loss: {loss.item()}")

# Example sampling
samples = model.sample(batch_size=5)
print(f"\nGenerated Samples Shape: {samples.shape}") # Should be (5, D)
# print(f"Generated Samples Min/Max: {samples.min().item():.2f}/{samples.max().item():.2f}")
# Check if range matches expected output (e.g., [0,1] if using sigmoid)