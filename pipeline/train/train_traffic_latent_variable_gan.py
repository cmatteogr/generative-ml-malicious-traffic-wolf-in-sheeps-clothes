"""
Author: Cesar M. Gonzalez

"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import optuna
import pandas as pd
import mlflow
import os
import time
import json
from torchinfo import summary
from ml_models.callbacks import EarlyStopping
from ml_models.malicious_traffic_latent_variable_gan import VAE
from utils.constants import TRAFFIC_GENERATOR_MODEL_FILENAME


def train(traffic_data_filepath: str, results_folder_path: str, train_size_percentage=0.8, batch_size=1024):
    """
    Beta-VAE training

    :param traffic_data_filepath: Traffic dataset
    :param results_folder_path: Folder path where save the results
    :param train_size_percentage: Train size percentage, remaining is validation size
    :param batch_size: Batch size
    :return:
    """
    print('# Start VAE training')

    # Check input arguments
    print('check training input arguments')
    assert 0.7 <= train_size_percentage < 1, 'Train size percentage should be between 0.7 and 1.'

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Convert to PyTorch tensor
    traffic_df = pd.read_csv(traffic_data_filepath)
    # remove label column
    # NOTE: This feature may be needed in the future to build the CGAN
    # traffic_df.pop('Label')
    n_features = len(traffic_df.columns)
    print(f"Training dataset, {n_features} features")
    tensor_data = torch.tensor(traffic_df.values, dtype=torch.float32)

    # Split into training and validation sets
    print('Split dataset into train and validation set')
    train_size = int(train_size_percentage * len(tensor_data))
    val_size = len(tensor_data) - train_size
    train_dataset, val_dataset = random_split(tensor_data, [train_size, val_size])

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Init the autoencoder Hyperparameters
    num_epochs = 350
    early_stopping_patience = 15
    kl_beta = 0.03

    # log in mlflow training params
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("early_stopping_patience", early_stopping_patience)
    mlflow.log_param("kl_beta", kl_beta)

    # Build the model tunner using optuna
    print('build VAE for generation model')

    def train_model(trial):
        # Init the Hyperparameters to change
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
        hidden_dim = trial.suggest_int('hidden_dim', 24, 38)
        latent_dim = trial.suggest_int('latent_dim', 12, 22)
        kl_beta = trial.suggest_float('kl_beta', 1e-2, 1)

        # Init the Autoencoder, loss function metric and optimizer\
        # Instantiate the VAE (Continuous)
        # TODO: Update VAE to allow update input dimension and latent dimension as hyperparameter: n_features, L
        model: VAE = VAE(input_dim=n_features, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Early stopping is added to avoid overfitting
        early_stopping = EarlyStopping(patience=early_stopping_patience)

        # Training loop
        print('Training VAE model')
        best_trial_val_loss = float('inf')
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            model.train()
            train_loss_accum = 0.0

            reconstruction_loss_accum = 0.0
            kl_loss_accum = 0.0

            for data in train_loader:
                # Forward pass
                data = data.to(device)
                optimizer.zero_grad()

                reconstruction_loss_value, kl_value = model(data, reduction='avg')
                kl_value = kl_value * kl_beta
                loss = reconstruction_loss_value + kl_value

                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()

                reconstruction_loss_accum += reconstruction_loss_value.item()
                kl_loss_accum += kl_value.item()

            # Calculate train batch loss
            avg_train_loss = train_loss_accum / len(train_loader)

            avg_reconstruction_loss = reconstruction_loss_accum / len(train_loader)
            avg_kl_loss = kl_loss_accum / len(train_loader)

            print(f'train MSE: {avg_reconstruction_loss}, train KL-divergence: {avg_kl_loss/kl_beta}, train KL-divergence Beta:{avg_kl_loss}')

            # Validation
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    reconstruction_loss_value, kl_value = model(data, reduction='avg')
                    kl_value = kl_value * kl_beta
                    loss = reconstruction_loss_value + kl_value

                    if torch.isnan(loss):  # Check for NaN loss
                        print(f"Warning: NaN loss detected in validation epoch {epoch + 1}. Pruning trial.")
                        raise optuna.exceptions.TrialPruned()
                    val_loss_accum += loss.item()
            avg_val_loss = val_loss_accum / len(val_loader)

            epoch_duration = time.time() - epoch_start_time
            print(
                f'  Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Duration: {epoch_duration:.2f}s')

            # Optuna Pruning / Reporting
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                print("  Trial pruned by Optuna.")
                raise optuna.exceptions.TrialPruned()

            # Early Stopping Check
            if early_stopping.step(avg_val_loss):
                print("  Early stopping triggered during trial.")
                break

            # Track best validation loss for this trial
            best_trial_val_loss = min(best_trial_val_loss, avg_val_loss)

        # Return the best validation loss achieved in this trial
        print(f"--- Trial {trial.number} Finished. Best Val Loss: {best_trial_val_loss:.6f} ---")
        return best_trial_val_loss  # Optuna minimizes this value

    # Execute optuna optimizer study
    print('train VAE')
    study_name = "malicious_traffic_latent_variable_gan_v5"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name= study_name,
                                storage=storage_name,
                                load_if_exists=True,
                                direction='minimize')
    study.optimize(train_model, n_trials=50)
    # Get Best parameters
    best_params = study.best_params
    best_value = study.best_value
    print(f'best params: {best_params}')
    print(f'best elbo: {best_value}')
    mlflow.log_param('vae_params', str(best_params))

    # Retrain the best model
    # TODO:: The retraining is done from scratch using the best hyperparameters to ensure the values benefit the model

    # Init the Autoencoder, loss function metric and optimizer
    latent_dim = best_params['latent_dim']
    hidden_dim = best_params['hidden_dim']
    kl_beta = best_params['kl_beta']
    learning_rate = best_params['learning_rate']
    model: VAE = VAE(input_dim=n_features, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Early stopping is added to avoid overfitting
    early_stopping = EarlyStopping(patience=early_stopping_patience*2)

    # Training loop
    print(f'Training Beta-VAE. Best params: {best_params}')
    best_val_loss = float('inf')
    for epoch in range(num_epochs * 2):
        epoch_start_time = time.time()
        model.train()
        train_loss_accum = 0.0

        reconstruction_loss_accum = 0.0
        kl_loss_accum = 0.0

        for data in train_loader:
            # Forward pass
            data = data.to(device)
            optimizer.zero_grad()

            reconstruction_loss_value, kl_value = model(data, reduction='avg')
            kl_value = kl_value * kl_beta
            loss = reconstruction_loss_value + kl_value

            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item()

            reconstruction_loss_accum += reconstruction_loss_value.item()
            kl_loss_accum += kl_value.item()

        # Calculate train batch loss
        avg_train_loss = train_loss_accum / len(train_loader)
        # Calculate train KL Divergence and MSE
        avg_reconstruction_loss = reconstruction_loss_accum / len(train_loader)
        avg_kl_loss = (kl_loss_accum / len(train_loader)) / kl_beta

        print(
            f'train MSE: {avg_reconstruction_loss}, train KL-divergence: {avg_kl_loss}, train KL-divergence Beta:{avg_kl_loss *  kl_beta}')

        # log metrics
        mlflow.log_metric('train/kl_divergence_value', avg_kl_loss, step=epoch)
        mlflow.log_metric('train/mse_value', avg_reconstruction_loss, step=epoch)
        mlflow.log_metric('train/neg_elbo_value', avg_train_loss, step=epoch)

        # Validation
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                reconstruction_loss_value, kl_value = model(data, reduction='avg')
                kl_value = kl_value * kl_beta
                loss = reconstruction_loss_value + kl_value

                if torch.isnan(loss):  # Check for NaN loss
                    print(f"Warning: NaN loss detected in validation epoch {epoch + 1}. Pruning trial.")
                    raise optuna.exceptions.TrialPruned()
                val_loss_accum += loss.item()
        # Calculate validation batch loss
        avg_val_loss = val_loss_accum / len(val_loader)
        # Calculate train KL Divergence and MSE
        avg_val_reconstruction_loss = reconstruction_loss_accum / len(train_loader)
        avg_val_kl_loss = (kl_loss_accum / len(train_loader)) / kl_beta

        epoch_duration = time.time() - epoch_start_time
        print(
            f'  Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Duration: {epoch_duration:.2f}s')

        # log metrics
        mlflow.log_metric('validation/kl_divergence_val_value', avg_val_kl_loss, step=epoch)
        mlflow.log_metric('validation/mse_val_value', avg_val_reconstruction_loss, step=epoch)
        mlflow.log_metric('validation/neg_elbo_val_value', avg_val_loss, step=epoch)

        # Early Stopping Check
        if early_stopping.step(avg_val_loss):
            print("  Early stopping triggered during trial.")
            break

        # Track best validation loss for this trial
        best_val_loss = min(best_val_loss, avg_val_loss)

    print(f'ELBO, best score: {best_val_loss:.8f}')
    # Build train report
    print('create training report dict')
    report_dict = {
        'elbo': float(best_val_loss)
    }
    # Log metrics
    mlflow.log_metric("elbo_train_value", float(best_val_loss))

    # Save model
    model_filepath = os.path.join(results_folder_path, TRAFFIC_GENERATOR_MODEL_FILENAME)
    torch.save(model.state_dict(), model_filepath)

    # Log training parameters.
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("latent_dim", latent_dim)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("loss_function", 'ELBO')
    mlflow.log_param("optimizer", optimizer.__class__.__name__)

    # Log model summary.
    model_summary_filepath = os.path.join(results_folder_path, "beta_vae_summary.txt")
    with open(model_summary_filepath, "w") as f:
        f.write(str(summary(model)))
    mlflow.log_artifact(model_summary_filepath)

    print('end training sparse autoencoder anomaly detection model')
    # Return model filepath
    return model_filepath, report_dict

