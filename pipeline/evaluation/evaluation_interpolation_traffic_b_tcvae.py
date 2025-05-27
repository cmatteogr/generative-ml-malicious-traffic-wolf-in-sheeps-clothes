"""
Evaluation Interpolation
"""
import os
import pandas as pd
import torch
import xgboost as xgb
from ml_models.malicious_traffic_b_tcvae import VAE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import joblib
from utils.plots import plot_latent_space_vae


def evaluation_interpolation(model_generator_filepath: str, model_discriminator_filepath: str,
                             scale_model_filepath: str, label_encoder_model_filepath: str,
                             traffic_data_filepath: str, results_folder_path: str):
    """
    evaluate Interpolation in latent space
    :param model_generator_filepath: model generator filepath
    :param model_discriminator_filepath: model discriminator filepath
    :param scale_model_filepath: scale model filepath
    :param label_encoder_model_filepath: label encoder model filepath
    :param traffic_data_filepath: test data filepath
    :param results_folder_path: results folder path
    """
    # define device to use
    device = torch.device("cpu")

    # Load discriminator the model
    discriminator_model = xgb.XGBClassifier()
    discriminator_model.load_model(model_discriminator_filepath)
    # load scaler and label encoding model
    scaler_model: MinMaxScaler = joblib.load(scale_model_filepath)
    label_encoder_model: LabelEncoder = joblib.load(label_encoder_model_filepath)

    # read data
    traffic_df = pd.read_csv(traffic_data_filepath)
    # filter by source and destination labels
    # TODO: This logic could be an input argument to make the test modular/reusable
    source_label = 'PortScan'
    destination_label = 'BENIGN'
    # filter by labels to interpolate
    traffic_df = traffic_df.loc[traffic_df['Label'].isin([source_label, destination_label])]
    # sample
    traffic_df = traffic_df.sample(35000)
    # select two instances, one BENIGN other PortScan
    source_instance = traffic_df[traffic_df['Label'] == source_label].head(1)
    destination_instance = traffic_df[traffic_df['Label'] == destination_label].head(1)
    # remove label
    destination_instance.pop('Label')
    source_instance.pop('Label')

    # define n columns
    n_features = len(destination_instance.columns)
    # TODO: latent_dim and hidden_dim could be input arguments, or any other option to follow the best programming practices
    latent_dim = 16
    hidden_dim = 36
    # init generative model
    generator_model: VAE = VAE(input_dim=n_features, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    generator_model.load_state_dict(torch.load(model_generator_filepath))

    # transform source and destination instances to tensors
    source_instance_tensor = torch.tensor(source_instance.values, dtype=torch.float32).to(device)
    destination_instance_data = torch.tensor(destination_instance.values, dtype=torch.float32).to(device)

    # encode source and destination instances to latent space
    destination_z = generator_model.encoder.sample(source_instance_tensor)
    source_z = generator_model.encoder.sample(destination_instance_data)

    # generate the chain of interpolation
    # star from source
    z_chain_labels = [source_label]
    z_chain = [source_z]
    # for each interpolation instance add to chain
    percentage_steps = 0.02
    for step_percentage in np.arange(0, 1, percentage_steps):
        # apply interpolation from source to destination in step_percentage
        interpolated_sample, z_interp = generator_model.interpolation(z1=destination_z, z2=source_z, alpha=step_percentage)
        # To convert it to a NumPy array, first move it to the CPU
        interpolated_sample_numpy_array = interpolated_sample.cpu().numpy()
        # Apply inverse scaler to use the discriminator and get predicted target
        reconstructed_sample = scaler_model.inverse_transform(interpolated_sample_numpy_array)
        label_index_pred = discriminator_model.predict(reconstructed_sample)
        # Apply Label encoder inverse to get label name
        pred_label = label_encoder_model.inverse_transform(label_index_pred)
        # append interpolation and label to chain
        z_chain.append(z_interp)
        z_chain_labels.append(pred_label[0])

    # append destination instance
    z_chain.append(destination_z)
    z_chain_labels.append(destination_label)

    # concatenate all the tensors instances
    z_tensor_chain = torch.cat(z_chain, dim=0)

    # plot the instances in the latent space
    # NOTE: The latent space is a projection so you may not see a straight line in the plot
    results_filepath = os.path.join(results_folder_path, "vae_latent_space_pca_btcvea_interpolation.html")
    plot_title = 'B-TCVAE-GAN Latent Space'
    plot_latent_space_vae(z_tensor_chain.detach().numpy(), z_chain_labels, results_filepath,
                          market_size=4, plot_title=plot_title, equal_range_axis=True)