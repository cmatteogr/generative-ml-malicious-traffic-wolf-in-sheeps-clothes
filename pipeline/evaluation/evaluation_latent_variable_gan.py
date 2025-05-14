"""
Evaluation step
"""
import os
import pandas as pd
import mlflow
import torch
import json
from torch.utils.data import DataLoader
from ml_models.malicious_traffic_latent_variable_gan import VAE
import plotly.express as px
from sklearn.decomposition import PCA
from ydata_profiling import ProfileReport



def evaluation(model_filepath: str, model_hyperparams_filepath: str, data_test_filepath: str, results_folder_path: str) -> dict:
    """
    evaluate classifier on test data
    :param model_filepath: model filepath
    :param model_hyperparams_filepath: VAE Hyperparameter
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

    # get model args
    #with open(model_hyperparams_filepath, 'r') as file:
    #    model_hyperparams_data = json.load(file)
    #    latent_dim = model_hyperparams_data['latent_dim']
    #    hidden_dim = model_hyperparams_data['hidden_dim']
    #    kl_beta = model_hyperparams_data['kl_beta']

    latent_dim = 20
    hidden_dim = 38
    kl_beta = 0.03

    # init model with hyperparameters
    model: VAE = VAE(input_dim=n_features, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_filepath))

    # set model to evaluation mode
    reconstruction_loss_value, kl_value = model(tensor_data, reduction='avg')
    reconstruction_loss_value = reconstruction_loss_value.item()
    kl_value = kl_value.item()
    eval_loss_elbo = reconstruction_loss_value + kl_value
    # Calculate the ELBO, KL Divergence value and MSE averages value
    avg_eval_loss = eval_loss_elbo / len(evaluation_loader)
    avg_eval_kl = (kl_value / len(evaluation_loader)) / kl_beta
    avg_eval_reconstruction_loss = reconstruction_loss_value / len(evaluation_loader)

    # log metrics
    #mlflow.log_metric("eval/neg_elbo_value", avg_eval_loss)
    #mlflow.log_metric("eval/kl_divergence_value", avg_eval_kl)
    #mlflow.log_metric("eval/mse_val_value", avg_eval_reconstruction_loss)

    # NOTE: There are different ways to measure the VAE performance. We will apply 3 related to measure
    # the reconstruction performance and the latent space consistency
    # - MSE should be similar to the training/validation value, about >= 0.1 (it means 10% general error) this ensures the reconstruction is good enough
    # - Latent space shape, it should be a Gaussian like distribution after apply PCA or select random features, to plot in 3D

    # print MSE to evaluate the value
    print(f"mse_val_value: {avg_eval_reconstruction_loss}")

    # Use the VAE-Encoder to generate z values and plot the shape
    z = model.encoder.sample(x=tensor_data)

    # Apply PCA or select 3 dimensions randomly to show our latent space is a Gaussian like distribution
    # reduce the latent space dimensionality to plot the latent space representation
    pca = PCA(n_components=3)
    z_transformed_pca = pca.fit_transform(z.detach().numpy())
    plot_cols_pca = ['PCA1', 'PCA2', 'PCA3']
    plot_title_pca = 'VAE Latent Space (PCA - 3 Components)'\
    # plot the latent space in 3D
    df_pca_plot = pd.DataFrame(z_transformed_pca, columns=plot_cols_pca)
    df_pca_plot['Label'] = labels
    fig_pca = px.scatter_3d(df_pca_plot,
                            x=plot_cols_pca[0],
                            y=plot_cols_pca[1],
                            z=plot_cols_pca[2],
                            color='Label',
                            title=plot_title_pca,
                            labels={'color': 'True Label'})
    fig_pca.update_traces(marker=dict(size=2, opacity=0.7))
    # save plot
    latent_space_pca_filepath = os.path.join(results_folder_path, "vae_latent_space_pca.html")
    fig_pca.write_html(latent_space_pca_filepath)
    #mlflow.log_artifact(latent_space_pca_filepath, "vae_latent_space_plots")

    # Use ydata-profiling to compare input and generated data
    original_data_report = ProfileReport(traffic_df, title="Original data")
    # sample data from the VAE
    z = model.encoder.sample(tensor_data)
    generated_data = model.decoder.sample(z)
    # transform data to dataframe
    generated_data_df = pd.DataFrame(generated_data.detach().numpy(), columns=traffic_df.columns)
    # generate report with generated data
    generated_data_report = ProfileReport(generated_data_df, title="Generated data")
    # compare reports, original data and generated data
    comparison_report = original_data_report.compare(generated_data_report)
    original_data_vs_generated_data_filepath = os.path.join(results_folder_path, "original_data_vs_generated_data.html")
    comparison_report.to_file(original_data_vs_generated_data_filepath)

