"""
Evaluation step
"""
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ml_models.malicious_traffic_b_vae import VAE
import plotly.express as px
from sklearn.decomposition import PCA
from ydata_profiling import ProfileReport

from utils.plots import plot_latent_space_vae
from utils.reports import compare_y_data_profiling


def evaluation(model_filepath: str, data_test_filepath: str, results_folder_path: str):
    """
    evaluate classifier on test data
    :param model_filepath: model filepath
    :param data_test_filepath: test data filepath
    :param results_folder_path: results folder path
    """
    print("read test traffic data")
    traffic_df = pd.read_csv(data_test_filepath)
    labels = traffic_df.pop('Label')
    tensor_data = torch.tensor(traffic_df.values, dtype=torch.float32)
    evaluation_loader = DataLoader(tensor_data, batch_size=512, shuffle=False)
    # define n columns
    n_features = len(traffic_df.columns)

    # define device to use
    device = torch.device("cpu")

    latent_dim = 20
    hidden_dim = 38
    kl_beta = 0.03

    # init model with hyperparameters
    model: VAE = VAE(input_dim=n_features, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_filepath))

    # set model to evaluation mode
    reconstruction_loss_value, kl_value = model(tensor_data, reduction='avg')
    reconstruction_loss_value = reconstruction_loss_value.item()
    # Calculate the ELBO, KL Divergence value and MSE averages value
    avg_eval_reconstruction_loss = reconstruction_loss_value / len(evaluation_loader)

    # NOTE: There are different ways to measure the VAE performance. We will apply 3 related to measure
    # the reconstruction performance and the latent space consistency
    # - MSE should be similar to the training/validation value, about >= 0.1 (it means 10% general error) this ensures the reconstruction is good enough
    # - Latent space shape, it should be a Gaussian like distribution after apply PCA or select random features, to plot in 3D

    # print MSE to evaluate the value
    print(f"mse_val_value: {avg_eval_reconstruction_loss}")

    # Use the VAE-Encoder to generate z values and plot the shape
    z = model.encoder.sample(x=tensor_data)

    # plot the latent space in 3D
    latent_space_pca_filepath = os.path.join(results_folder_path, "vae_latent_space_pca_bvea.html")
    plot_latent_space_vae(z.detach().numpy(), labels, labels, latent_space_pca_filepath)

    # generate comparison data report
    # sample data from the VAE
    z = model.encoder.sample(tensor_data)
    generated_data = model.decoder.sample(z)
    # transform data to dataframe
    generated_data_df = pd.DataFrame(generated_data.detach().numpy(), columns=traffic_df.columns)
    # generate report
    original_vs_generated_filepath = os.path.join(results_folder_path, "original_data_vs_generated_data_bvea.html")
    compare_y_data_profiling(traffic_df, "Original data", generated_data_df, "Generated data",
                             original_vs_generated_filepath)
