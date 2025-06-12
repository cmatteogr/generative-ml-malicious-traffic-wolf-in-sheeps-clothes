"""
Beta - Total Correlation Variational Autoencoder. b-tcvae
"""
import math
import torch
import torch.nn as nn

from pipeline.postprocessing.postprocessing_base import post_process_data

D = 42  # input dimension by default
L = 22   # number of latents by default
M = 36  # hidden layer dimension bu default

# Use math.pi for scalar, torch.pi for tensors if needed later
PI = torch.tensor(math.pi)

def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    """Calculates log probability log N(x|mu, diag(exp(log_var))) of diagonal Gaussian."""
    # Calculates the log probability density of x given a Gaussian distribution, How probably you can find x in that distribution
    # This value is used as part of the KL Divergence metric, we are looking for to increase this probability value, that way
    # we are "mapping" the x inputs with their z-Gaussian like representations.
    # NOTE: We are using logs to simplify the calculation fue the logs properties, instead of multiplications they are sums
    # NOTE: The Diagonal name is because all the off-diagonal elements are zero. This implies that the different dimensions of x are uncorrelated.
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


def log_q_zd_given_xk_all_dim(z, mu_e, log_var_e):
    """
    Estimate log q(z_j). We need to calculate the log q(z_j) where j represent the dimensions for all the instances that comes from the input z

    :param z: Samples from q(z|x), shape (batch_size, latent_dim).
    :param mu_e: Mean of q(z|x) from encoder, shape (batch_size, latent_dim).
    :param log_var_e: Log variance of q(z|x) from encoder, shape (batch_size, latent
    :return: torch.Tensor: Log q(z_j) for all x in all dimensions

    """
    # the mathematical definition is:
    # log_q_zd_given_xk_all_d[i, k, d] = log N(z[i,d] | mu_e[k,d], exp(log_var_e[k,d])). This is equal to
    # log_q_zd_given_xk_all_d[i, k, d] = - 1/2 log(2π) − 1/2 log_var − 1/2 exp(−log_var)(z−μ)**2
    # This formula allow us to calculate the log probability density in the instance z, we are calculating how likely the given z value is

    #  unsqueeze method:  https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html
    log_q_zd_given_xk_all_d = -0.5 * torch.log(2. * PI.to(z.device)) \
                              - 0.5 * log_var_e.unsqueeze(0) \
                              - 0.5 * torch.exp(-log_var_e.unsqueeze(0)) * \
                              (z.unsqueeze(1) - mu_e.unsqueeze(0)) ** 2
    return log_q_zd_given_xk_all_d

def calculate_mutual_information(log_qzx, log_qz):
    """
    Calculates the Mutual Information term: I(x;z) approx KL(q(z|x) || q(z)).
    This is E_q(z|x)[log q(z|x) - log q(z)].

    Using this metric we can identify how much information the channel, latent space keeps from the input x
    https://medium.com/swlh/a-deep-conceptual-guide-to-mutual-information-a5021031fad0

    :param log_qzx: log q probability of z given x
    :param log_qz: log q probability of z integrated in each dimension
    :return: torch.Tensor: Scalar value of the estimated mutual information between z and x.

    """
    # MI = E_q(z|x) [log q(z|x) - log q(z)]
    # Approximated by mean over the batch.
    mi = (log_qzx - log_qz).mean()
    return mi


def calculate_total_correlation(log_qz, log_q_zd):
    """
    Calculates the Total Correlation term: TC = KL(q(z) || prod_j q(z_j)).
    This is E_q(z) [log q(z) - sum_j log q(z_j)]. https://youtu.be/RNAZA7iytNQ?t=890
    This formula is based on the Total Correlation concept: https://dit.readthedocs.io/en/latest/measures/multivariate/total_correlation.html#:~:text=The%20total%20correlation%20consists%20of,variables%20it%20is%20shared%20among.

    With this metrics we are looking for to reduce the dimension dependency (redundancy) in the latent space, promoting the disentanglement
    Each dimension is independent from each others, we can achieve that if the log q(z) is similar/equal to the Sum of all the log q(z_j), where j represent the latent space dimensions
    https://youtu.be/RNAZA7iytNQ?t=543

    :param log_qz: log q probability of z integrated in each dimension
    :param log_q_zd: log q probability of z sum up in each dimension
    :return: torch.Tensor: Scalar value of the estimated total correlation.
    """
    # sum_j log q(z_j) for each sample i in batch (sum over latent dimensions d)
    sum_log_q_zd = log_q_zd.sum(dim=1)

    # TC = E_q(z) [log q(z) - sum_j log q(z_j)]
    # Expectation approximated by mean over the batch.
    tc = (log_qz - sum_log_q_zd).mean()
    return tc


def calculate_dimension_wise_kl(log_q_zd, log_pz_j):
    """
    Calculates the Dimension-wise KL term: sum_j KL(q(z_j) || p(z_j)).

    NOTE: The Dimension-wise KL term is named so because it specifically measures how much each individual dimension of the aggregated latent distribution qϕ(z_j)
    deviates from its designated prior p(z_j))

    This is sum_j E_q(z_j) [log q(z_j) - log p(z_j)]. E_q is the expectation of a random Variable
    check the following videos to understand what a Expectation of a rando variable is and How is it applied using the Jensen Inequality:
    * Expectation of a random variable: https://www.youtube.com/watch?v=sheoa3TrcCI
    * Jensen's Inequality: https://www.youtube.com/watch?v=u0_X2hX6DWE. The book Deep Generative Modeling shows in detail how it works
    :param log_q_zd: log q probability of z sum up in each dimension
    :param log_pz_j: Prior Normal/Gaussian distribution in each dimension
    :return: torch.Tensor: Scalar value of the estimated dimension-wise KL.
    """
    # Calculate the KL divergence between q(z_j) and p(z_j)
    # KL_j = E_q(z_j) [log q(z_j) - log p(z_j)]
    # Expectation E_q(z_j) is approximated by mean over batch samples of z_j.
    # So for each dimension j (latent_dim), KL_j = mean_i (log_q_zd[i,j] - log_pz_j[i,j])
    kl_j_per_sample = log_q_zd - log_pz_j
    kl_j = kl_j_per_sample.mean(dim=0)

    # sum_j KL_j
    dwkl = kl_j.sum()
    return dwkl


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

        # apply postprocessing
        # mu_d = post_process_data(mu_d)

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
        neg_sum_sq_error = -torch.sum((x - mu_d)**2, dim=1)
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

class B_TCVAE(nn.Module):
    """Variational Autoencoder (VAE) model for Continuous Data."""
    def __init__(self, input_dim=D, latent_dim=L, hidden_dim=M):
        super(B_TCVAE, self).__init__()
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
        # sum_sq_error = self.decoder.log_prob(x, z)
        log_px_given_z = self.decoder.log_prob(x, z)

        # Instead of the KL divergence like in the b-VAE
        # we use 3 metrics to guarantee:
        # 1. The information transmission (Mutual Information).
        # 2. Latent space disentanglement (Total Correlation)
        # 3. The Prio and Posterior correlation in each dimension (Dimension-wise KL)

        # define the batch size and latent dimension, it's used to apply unsqueeze
        batch_size, latent_dim = z.shape

        # --- Calculate log probability distributions ----
        # These values are used to calculate the MI, TC and DWKL metrics
        # NOTE: In previous versions these log probabilities were calculated in each MI, TC and DWKL function,
        # causing redundant code, and slow execution due they were calculated more than one time. To fix it, they were grouped

        # Calculate log q probability of z given x
        # in normal diagonal format to simplify operation: log q(z|x)
        # log_qzx[i] = log q(z_i | x_i)
        log_qzx = log_normal_diag(z, mu_e, log_var_e, reduction='sum', dim=1)

        # Calculate log q probability of z given x for each one of the dimensions j
        # log_q_z_given_xj_matrix[i, j] = log q(z_i | x_j)
        # z_i is z[i,:], x_j corresponds to mu_e[j,:], log_var_e[j,:]
        #  unsqueeze method:  https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html
        # .unsqueeze(1) gives (batch_size, 1, latent_dim)
        # .unsqueeze(0) gives (1, batch_size, latent_dim)
        # After expansion, inputs to log_normal_diag are (batch_size, batch_size, latent_dim)
        # log_normal_diag sums over dim=2 (latent_dim)
        log_q_z_given_xj_matrix = log_normal_diag(
            z.unsqueeze(1).expand(batch_size, batch_size, -1),
            mu_e.unsqueeze(0).expand(batch_size, batch_size, -1),
            log_var_e.unsqueeze(0).expand(batch_size, batch_size, -1),
            reduction='sum',
            dim=2
        )
        # log q(z_i) = log (1/N * sum_j q(z_i | x_j)) = logsumexp_j (log q(z_i | x_j)) - log N
        log_qz = torch.logsumexp(log_q_z_given_xj_matrix, dim=1) - math.log(batch_size)

        # Calculate log q probability of z for each one of the dimensions j: log q(z_j).
        # We need to calculate the log q(z_j) where j represent the dimensions for all the instances that comes from the input z
        log_q_zd_given_xk_all_d = log_q_zd_given_xk_all_dim(z, mu_e, log_var_e)
        # sum all the q(zj) contributions (integrate) from all the samples given x_j
        # The command torch.logsumexp, returns the log of summed exponentials of each row of the input:
        # - https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html
        log_q_zd = torch.logsumexp(log_q_zd_given_xk_all_d, dim=1) - math.log(batch_size)

        # Calculate log q probability of z from a Normal/Gaussian distribution: log p(z_j) = log N(z_j | 0, 1)
        # log_standard_normal(z, reduction=None) will give element-wise log N(z[i,j]|0,1)
        log_pz_j = log_standard_normal(z)

        # ---- Calculate KL Divergences components: MI, TC, DWKL ----

        # Calculate Mutual information
        # MI = E_q(z|x) [log q(z|x) - log q(z)]
        mi = calculate_mutual_information(log_qzx, log_qz)

        # Calculate the Total Correlation term: TC = KL(q(z) || prod_j q(z_j)).
        # TC = E_q(z) [log q(z) - sum_j log q(z_j)]
        # Expectation approximated by mean over the batch.
        tc = calculate_total_correlation(log_qz, log_q_zd)

        # Calculate the KL divergence between q(z_j) and p(z_j)
        # KL_j = E_q(z_j) [log q(z_j) - log p(z_j)]
        dw_kl = calculate_dimension_wise_kl(log_q_zd, log_pz_j)

        # ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
        # elbo = log_px_given_z - KL # Shape: (batch_size,)
        # We want to maximize ELBO, which means minimizing -ELBO
        # -ELBO = KL - log_px_given_z
        # If log_px_given_z = -SumSqError, then -ELBO = KL + SumSqError
        # This matches the common VAE loss: Reconstruction Loss + KL Divergence

        reconstruction_loss = -log_px_given_z
        # Now, ELBO is the sum of both terms, MSE and KL Divergence, the goal is minimize this metric, that way we ensure
        # the VAE reconstructs the x instances as well as possible and Maps as good as possible the input distribution and the latent distribution (Gaussian like)
        # Remember using an approximation and the function depends on the VAE parameters, it means weights.
        #neg_elbo = reconstruction_loss + KL

        # Return average or sum of negative ELBO (loss to be minimized)
        # We are using batches so it make sense
        # TODO: Check if the MI, TC and DW_KL could have this sum and average effect, for now in both return the same value
        if reduction == 'sum':
            reconstruction_loss_value = reconstruction_loss.sum()
        else:
            reconstruction_loss_value = reconstruction_loss.mean()

        # return all the metrics
        return reconstruction_loss_value, mi, tc, dw_kl


    def sample(self, batch_size=64):
        """Generates new samples x' by sampling z ~ p(z) and decoding."""
        # Sample z from prior p(z) = N(0, I)
        z = self.prior.sample(batch_size=batch_size)
        # Decode z to get the mean reconstruction x' = E[p(x|z)]
        return self.decoder.sample(z)

    def reconstruct_x(self, x):
        """Use x and reconstruct it through the Encoder and Decoder"""
        # sample via encoder
        z = self.encoder.sample(x)
        # decode via decoder
        return self.decoder.sample(z)

    def interpolation(self, x1=None, x2=None, z1=None, z2=None, alpha=0.5):
        """Generates new samples x' by sampling z ~ p(z) and decoding."""
        # check if x1 and x2 were provided, z1 and z2 exclusive
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}.")

            # Ensure either (x1, x2) or (z1, z2) is provided, but not both mixed.
        if (x1 is None or x2 is None) and (z1 is None or z2 is None):
            raise ValueError("Provide either (x1 and x2) or (z1 and z2) for interpolation.")
        if (x1 is not None and z1 is not None) or \
                (x2 is not None and z2 is not None):  # Avoid mixing x and z inputs for the same point
            raise ValueError("Provide either x inputs or z inputs, not a mix for the same point.")

            # Encode x1 and x2 to their latent representations if z1 and z2 are not given
            # Ensure model is in eval mode and use no_grad for inference if encoding
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculations for this part
            if z1 is None and x1 is not None:
                mu1, log_var1 = self.encoder.encode(x1)
                z1 = self.encoder.reparameterization(mu1, log_var1)  # Or just self.encoder.sample(x=x1)
            if z2 is None and x2 is not None:
                mu2, log_var2 = self.encoder.encode(x2)
                z2 = self.encoder.reparameterization(mu2, log_var2)  # Or just self.encoder.sample(x=x2)

        if z1 is None or z2 is None:  # Should not happen if logic above is correct
            raise ValueError("Latent vectors z1 and z2 could not be determined.")

        # Perform linear interpolation in the latent space
        # z_interp = z1 + alpha * (z2 - z1)
        z_interp = (1.0 - alpha) * z1 + alpha * z2

        # Decode the interpolated latent vector to get the sample
        with torch.no_grad():  # Disable gradient calculations for decoding
            interpolated_sample = self.decoder.sample(z_interp)  # Or self.decoder.decode(z_interp)

        return interpolated_sample, z_interp
