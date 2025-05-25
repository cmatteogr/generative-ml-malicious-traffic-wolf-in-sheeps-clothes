"""
Evaluation step
"""
import os
import pandas as pd
import torch
import xgboost as xgb
from ml_models.malicious_traffic_b_tcvae import VAE
import plotly.express as px
from sklearn.decomposition import PCA
from ydata_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import joblib



def evaluation_interpolation(model_generator_filepath: str, model_discriminator_filepath: str,
                             scale_model_filepath: str,
                             label_encoder_model_filepath: str,
                             data_test_filepath: str,
                             results_folder_path: str) -> dict:
    """
    evaluate classifier on test data
    :param model_generator_filepath: model generator filepath
    :param model_discriminator_filepath: model discriminator filepath
    :param data_test_filepath: test data filepath
    :param results_folder_path: results folder path
    """
    print("read test traffic data")

    # define device to use
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

    # Load the model
    discriminator_model = xgb.XGBClassifier()
    discriminator_model.load_model(model_discriminator_filepath)
    # load inverse scaler
    scaler_model: MinMaxScaler = joblib.load(scale_model_filepath)
    label_encoder_model: LabelEncoder = joblib.load(label_encoder_model_filepath)


    traffic_df = pd.read_csv(data_test_filepath)

    # filter by labels to interpolate
    traffic_df = traffic_df.loc[traffic_df['Label'].isin(['BENIGN', 'PortScan'])]
    # sample
    traffic_df = traffic_df.sample(35000)
    # select two instances, one BENIGN other PortScan
    benign_serie = traffic_df[traffic_df['Label'] == 'BENIGN'].head(1)
    portscan_serie = traffic_df[traffic_df['Label'] == 'PortScan'].head(1)
    # remove label
    benign_serie.pop('Label')
    portscan_serie.pop('Label')
    labels = traffic_df.pop('Label')

    # define n columns
    n_features = len(traffic_df.columns)
    latent_dim = 16
    hidden_dim = 36

    # init model with hyperparameters
    generator_model: VAE = VAE(input_dim=n_features, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    generator_model.load_state_dict(torch.load(model_generator_filepath))

    benign_serie_data = torch.tensor(benign_serie.values, dtype=torch.float32).to(device)
    portscan_serie_data = torch.tensor(portscan_serie.values, dtype=torch.float32).to(device)

    benign_z = generator_model.encoder.sample(benign_serie_data)
    portscan_z = generator_model.encoder.sample(portscan_serie_data)

    z_chain_labels = ['PortScan']
    z_chain = [portscan_z]
    for _ in range(10):
        interpolated_sample, z_interp = generator_model.interpolation(z1=portscan_z, z2=benign_z)

        # To convert it to a NumPy array, first move it to the CPU
        cpu_tensor = interpolated_sample.cpu()
        # Now you can convert the CPU tensor to a NumPy array
        numpy_array = cpu_tensor.numpy()

        reconstructed_sample = scaler_model.inverse_transform(numpy_array)
        y_pred = discriminator_model.predict(reconstructed_sample)

        pred_label = label_encoder_model.inverse_transform(y_pred)

        z_chain.extend(z_interp)
        z_chain_labels.extend(pred_label)

    # Apply PCA or select 3 dimensions randomly to show our latent space is a Gaussian like distribution
    # reduce the latent space dimensionality to plot the latent space representation
    pca = PCA(n_components=3)
    z_transformed_pca = pca.fit_transform(z_chain.detach().numpy())
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
    latent_space_pca_filepath = os.path.join(results_folder_path, "vae_latent_space_pca_btcvea.html")
    fig_pca.write_html(latent_space_pca_filepath)
    #mlflow.log_artifact(latent_space_pca_filepath)

    # Use ydata-profiling to compare input and generated data
    original_data_report = ProfileReport(traffic_df, title="Original data")
    # sample data from the VAE
    z = generator_model.encoder.sample(benign_serie_loader)
    generated_data = generator_model.decoder.sample(z)
    # transform data to dataframe
    generated_data_df = pd.DataFrame(generated_data.detach().numpy(), columns=traffic_df.columns)
    # generate report with generated data
    generated_data_report = ProfileReport(generated_data_df, title="Generated data")
    # compare reports, original data and generated data
    comparison_report = original_data_report.compare(generated_data_report)
    original_data_vs_generated_data_filepath = os.path.join(results_folder_path, "original_data_vs_generated_data_btcvea.html")
    comparison_report.to_file(original_data_vs_generated_data_filepath)
    # save report in artifacts
    #mlflow.log_artifact(original_data_vs_generated_data_filepath)


