"""
VAE for Continuous Data using Gaussian Likelihood (Implicit Fixed Variance via MSE)
"""
import math # Use math.pi
import torch
import torch.nn as nn
import numpy as np

# --- Example Usage ---
D = 42  # input dimension by default
L = 22   # number of latents by default
M = 36  # hidden layer dimension bu default

# Use math.pi for scalar, torch.pi for tensors if needed later
PI = torch.tensor(math.pi)
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
        # Reparameterization trick: https://www.youtube.com/watch?v=xL1BJBarzWI
        # Reparameterization allows separate the randomness of the latent space distribution in a variable (eps), that way the back propagation works only with the mean and standard deviation (mu, std)
        """Applies the reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std) # Sample epsilon from standard normal
        return mu + std * eps

    def encode(self, x):
        """Passes input through the encoder network to get mu_e and log_var_e."""
        h_e = self.encoder(x)
        # Split the output into mean and log variance
        # The encoder output is the mean and standard deviation (mu, std)
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)
        return mu_e, log_var_e

    def sample(self, x=None, mu_e=None, log_var_e=None):
        """Samples z from the latent distribution q(z|x) = N(z|mu_e, exp(log_var_e))."""
        # Same is using to generate new data from the input x or the latent space (mu, std)
        # Here is where we can apply interpolation, to generate combined data (the wolf in sheep's clothes)
        # Even we can use this function + Reinforcement Learning (RL) to generate new unknown data points (new unknown wolf clothes/shapes/colors/smells,etc)
        # This image comes from a non-related video but it's a good representation of the different possible interpolation paths using these methods
        # https://youtu.be/qJZ1Ez28C-A?t=1537
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
        # ln qφ(z) is the probability density of z given by a specific distribution (parameterized by ϕ) chosen from a family of distributions (e.g., Gaussian).
        # We are approximating to the 'posterior' qϕ(z∣x) is designed to approximate the true posterior distribution p(z∣x)
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
        # z comes from the latent space after reparameterization trick using -> z = mu + std * eps
        # The decoder reconstructs the input x from the latent variable z
        # The goal is train the decoder parameters to do it. Keeping in mind z is a Gaussian Like distribution, and we are generating a "mapping" between two distributions: x, z
        mu_d = self.decoder(z)
        return mu_d # Return the predicted mean

    def sample(self, z):
        """Generates the most likely reconstruction x' = E[p(x|z)] = mu_d."""
        # For a Gaussian N(mu, sigma^2*I), the mean mu is the most likely value. This is correct
        # If you wanted to sample *from* p(x|z), you'd need to assume/predict
        # a variance and add noise: mu_d + sigma * torch.randn_like(mu_d).
        # Sampling comes from the latent space
        mu_d = self.decode(z)
        return mu_d

    def log_prob(self, x, z):
        """Calculates reconstruction loss term.
        Using negative Mean Squared Error which is proportional to log N(x|mu_d, fixed_variance).
        Returns a value per batch item (reduction='none' essentially before sum).
        """
        # From the latent space z, reconstruct the input x, as result mu_d is calculated, the mean is used because it's the most probably value from a distribution
        mu_d = self.decode(z)
        # Calculate MSE loss per batch element, summing over feature dimension D
        # MSE = mean((x - mu_d)^2) over D. We want sum over D.
        # MSE video: https://www.youtube.com/watch?v=VaOlkbKQFcY
        # So, sum((x - mu_d)^2) over D, all dimensions, input features + any other input arg (condition in the CGANs).
        # The ELBO wants log p(x|z). For N(x|mu, sigma^2*I), log p = -0.5*log(2*pi*sigma^2)*D - 0.5/sigma^2 * sum((x-mu)^2)
        # If sigma=1, log p = C - 0.5 * sum((x-mu)^2).
        # Maximizing log p is equivalent to minimizing sum((x-mu)^2).
        # We return a value ~ log p(x|z) per batch item.
        # Let's return -sum((x - mu_d)**2) which is proportional to log p(x|z) up to constants.
        # This one here is the MSE formula
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
        # here the register_buffer is used to save the device used.
        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
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
    def __init__(self, input_dim=D, latent_dim=L, hidden_dim=M):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        print('VAE (Continuous Likelihood - Gaussian Mean Decoder) Initialized.')

        """
        The architecture has constant dimension in the hidden layers, this is intentional:

        Advantages:
        * Hidden Layers are for Feature Transformation: hidden layers focus on feature transformations and not on reduction.
        * Simplicity: only focus on M,D and L values. not on the dimension for each layer D-> (M1,M2,M3, ... Mn)->L

        Disadvantages:
        * Computational Cost: More parameters to train more computation/time is needed. Depends on M value.
        * Inefficiency in Parameters: From M to L, just before the latent space the parameters needed could be larger or smaller. Generating inefficiency
        * Less Explicit Hierarchical Feature Learning: The learning works on a hierarchical way through the layers-neurons. The same number of dimension for all layer it.  
        """

        # Define Encoder Network
        encoder_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),  # Changed to ReLU, common choice
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            # NOTE: This last layer is the Encoder returns twice the latent space dimension L because it returns Mean - Standard Derivation, to generate Gaussian representations
            # VAE architecture: https://youtu.be/qJeaCHQ1k2w?t=799
            # Generative Deep Learning Book: Variational Autoencoders - The Encoder: page 135
            nn.Linear(self.hidden_dim, 2 * self.latent_dim)  # Output mu and log_var
        )

        # Define Decoder Network
        # Output size must be D for continuous data (predicting the mean)
        decoder_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            # The Latent space of the decoder is connected to the z representation after use the reparameterization trick to transform (Mean - Standard Derivation) to z
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            # The input data is normalized to [0, 1], to use Sigmoid to force the same output normalization
            nn.Sigmoid()
        )

        # init encoder and decoder
        self.encoder = Encoder(encoder_net=encoder_net)
        self.decoder = Decoder(decoder_net=decoder_net)
        # init prior, prior is the previous hypothesis based on the Bayes Theorem
        # https://youtu.be/HZGCoVF3YvM?t=317
        # in our case the hypothesis is the Standard Gaussian distribution in the latent space
        self.prior = Prior(L=latent_dim)

        # latent space dimension
        self.L = latent_dim

    def forward(self, x, reduction='avg'):
        """Calculates the negative Evidence Lower Bound (ELBO) loss.
           loss = -ELBO = Reconstruction Loss + KL Divergence
                 = -E_q[log p(x|z)] + KL(q(z|x) || p(z))
            ELBO:
            https://youtu.be/iwEzwTTalbg?t=727
        """
        # Encode: Get q(z|x) parameters: mu_e, log_var_e
        # from the input x, generate the latent space parameters mean and standard deviation (mu, std)
        mu_e, log_var_e = self.encoder.encode(x)
        # Sample z ~ q(z|x) using reparameterization trick.
        # It's needed because we are using distributions, randomness is part of the deal, we need reparameterization to allows back propagation
        z = self.encoder.reparameterization(mu_e, log_var_e)

        # Calculate ELBO components (per batch item)
        # Reconstruction Term: E_q[log p(x|z)]
        # Approximated using single sample z: log p(x|z)
        # Here is where we are using Jensen Inequality: https://www.youtube.com/watch?v=u0_X2hX6DWE
        # The book Understanding Deep Learning explains better how it's used in the VAE.
        # We need to apply Jensen Inequality to get the ELBO function, which give us an approximation of log p(x) likelihood,
        # the input distribution likelihood in function of VAE parameters, we want to maximize this value (log p(x) likelihood)
        # but we can not track it, that's why we use ELBO, ELBO is an approximation of log p(x) likelihood based in the VAE parameters
        # Then having a good ELBO, we can calculate the max ELBO likelihood and find a good approximation of the VAE arguments
        # The books 'Understanding Deep Learning' and 'Deep Generative Modeling' explains better how it's used in the VAE.

        # We use -SumSqError which is proportional to log p(x|z) for fixed variance Gaussian
        log_px_given_z = self.decoder.log_prob(x, z) # Shape: (batch_size,)

        # KL Divergence Term: KL(q(z|x) || p(z))
        # KL = E_q[log q(z|x) - log p(z)]
        # This one is the log q(z|x), in the 'Deep Generative Modeling' is denoted as ln qφ(z).
        # ln qφ(z) is the probability density of z given by a specific distribution (parameterized by ϕ) chosen from a family of distributions (e.g., Gaussian).
        # We are approximating to the 'posterior' qϕ(z∣x) is designed to approximate the true posterior distribution p(z∣x)
        # NOTE: remember qϕ(z∣x) is an approximation of p(z∣x), the approximation is needed due: https://youtu.be/iwEzwTTalbg?t=602
        log_qzx = self.encoder.log_prob(z, mu_e, log_var_e) # Shape: (batch_size,)
        # This one is log p(z), in the 'Deep Generative Modeling' is denoted as ln p(z).
        # Denotes the latent space shape given z, assuming a Gaussian Like distribution
        log_pz = self.prior.log_prob(z)                   # Shape: (batch_size,)
        # KL Divergence is calculated subtracting the distributions probabilities, the approximation qϕ(z∣x) and the prior hypothesis p(z), assuming a Gaussian distribution
        KL = log_qzx - log_pz                             # Shape: (batch_size,)

        # ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
        # elbo = log_px_given_z - KL # Shape: (batch_size,)
        # We want to maximize ELBO, which means minimizing -ELBO
        # -ELBO = KL - log_px_given_z
        # If log_px_given_z = -SumSqError, then -ELBO = KL + SumSqError
        # This matches the common VAE loss: Reconstruction Loss + KL Divergence
        # Let's define Reconstruction Loss = -log_px_given_z
        reconstruction_loss = -log_px_given_z # Shape: (batch_size,)
        # Now, ELBO is the sum of both terms, MSE and KL Divergence, the goal is minimize this metric, that way we ensure
        # the VAE reconstructs the x instances as well as possible and Maps as good as possible the input distribution and the latent distribution (Gaussian like)
        # Remember using an approximation and the function depends on the VAE parameters, it means weights.
        #neg_elbo = reconstruction_loss + KL   # Shape: (batch_size,)

        # encoder
        mu_e, log_var_e = self.encoder.encode(x)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)

        # Return average or sum of negative ELBO (loss to be minimized)
        # We are using batches so it make sense
        if reduction == 'sum':
            reconstruction_loss_value = reconstruction_loss.sum()
            kl_value = KL.sum()
            # neg_elbo_value = neg_elbo.sum()
        else:
            reconstruction_loss_value = reconstruction_loss.mean()
            kl_value = KL.mean()
            # neg_elbo_value = neg_elbo.mean()

        return reconstruction_loss_value, kl_value


    def sample(self, batch_size=64):
        """Generates new samples x' by sampling z ~ p(z) and decoding."""
        # Sample z from prior p(z) = N(0, I)
        z = self.prior.sample(batch_size=batch_size)
        # Decode z to get the mean reconstruction x' = E[p(x|z)]
        return self.decoder.sample(z)

    def interpolation(self, x1=None, x2=None, z1=None, z2=None, num_steps=10):
        """Generates new samples x' by sampling z ~ p(z) and decoding."""
        # check if x1 and x2 were provided, z1 and z2 exclusive
        if (x1 is None) and (x2 is None) and (z1 is None) and (z2 is None):
            raise Exception(f"Provide at least x1, x2 or z1, z2.")
        # check exclusive condition
        if ((x1 is not None) and (x2 is not None)) and ((z1 is not None) and (z2 is not None)):
            raise Exception(f"Provide only x1, x2 or z1, z2, not both.")

        # if x1 and x2 were provide calculate the z representations
        if (x1 is None) and (x2 is None):
            z1 = self.encoder.sample(x=x1)
            z2 = self.encoder.sample(x=x1)

        # Apply interpolation
        for alpha_val in np.linspace(0, 1, num_steps):
            z_interp = (1 - alpha_val) * z1 + alpha_val * z2
            with torch.no_grad():
                interpolated_sample = self.decoder.sample(z_interp)
                return interpolated_sample

