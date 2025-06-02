"""
Author: Cesar M. Gonzalez

"""
import torch
import torch.optim as optim
from overrides import override
from torch.utils.data import DataLoader, random_split
import optuna
import pandas as pd
import mlflow
import os
import time
from torchinfo import summary
from ml_models.callbacks import EarlyStopping
from ml_models.malicious_traffic_b_tcvae import B_TCVAE
from utils.constants import TRAFFIC_GENERATOR_MODEL_FILENAME


def train(traffic_data_filepath: str, results_folder_path: str, train_size_percentage=0.75, batch_size=1024):
    """
    Beta - Total Correlation VAE training

    :param traffic_data_filepath: Traffic dataset
    :param results_folder_path: Folder path where save the results
    :param train_size_percentage: Train size percentage, remaining is validation size
    :param batch_size: Batch size
    :return:
    """
    print('# Start Beta - Total Correlation VAE training')

    # Check input arguments
    print('check training input arguments')
    assert 0.7 <= train_size_percentage < 1, 'Train size percentage should be between 0.7 and 1.'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

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
    num_epochs = 550
    early_stopping_patience = 15

    # log in mlflow training params
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("early_stopping_patience", early_stopping_patience)

    # Build the model tunner using optuna
    print('build VAE for generation model')

    def train_model(trial):
        # Init the Hyperparameters to change
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
        hidden_dim = trial.suggest_int('hidden_dim', 24, 38)
        latent_dim = trial.suggest_int('latent_dim', 12, 22)

        # lambda for MSE, lower as possible
        lambda_recon = trial.suggest_float('lambda_recon', 10.0, 12.0)
        # alpha for Mutual Information, around 1.0
        #alpha_mi = trial.suggest_float('alpha_mi', 0.8, 2.0)
        alpha_mi = trial.suggest_float('alpha_mi', 0.6, 0.8)
        # beta for Total Correlation, lower as possible
        beta_tc = trial.suggest_float('beta_tc', 1.0, 30.0)
        # gamma for Dimension-wise KL, around 1.0
        #gamma_dw_kl = trial.suggest_float('gamma_dw_kl', 0.8, 5.0)
        gamma_dw_kl = trial.suggest_float('gamma_dw_kl', 0.6, 0.8)

        # Init the Autoencoder, loss function metric and optimizer\
        # Instantiate the VAE (Continuous)
        # TODO: Update VAE to allow update input dimension and latent dimension as hyperparameter: n_features, L
        model: B_TCVAE = B_TCVAE(input_dim=n_features, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)

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
            mi_loss_accum = 0.0
            tc_loss_accum = 0.0
            dw_kl_loss_accum = 0.0

            for data in train_loader:
                # Forward pass
                data = data.to(device)
                optimizer.zero_grad()
                # use the model
                reconstruction_loss_value, mi, tc, dw_kl = model(data, reduction='avg')
                # multiply by the factors
                reconstruction_loss_value = reconstruction_loss_value * lambda_recon
                mi = mi * alpha_mi
                tc = tc * beta_tc
                dw_kl = dw_kl * gamma_dw_kl
                # sum total loss
                loss = reconstruction_loss_value + mi + tc + dw_kl

                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()

                reconstruction_loss_accum += reconstruction_loss_value.item()
                mi_loss_accum += mi.item()
                tc_loss_accum += tc.item()
                dw_kl_loss_accum += dw_kl.item()

            # Calculate train batch loss
            avg_train_loss = train_loss_accum / len(train_loader)

            avg_reconstruction_loss = reconstruction_loss_accum / len(train_loader)
            avg_mi_loss = mi_loss_accum / len(train_loader)
            avg_tc_loss = tc_loss_accum / len(train_loader)
            avg_dw_kl_loss = dw_kl_loss_accum / len(train_loader)

            print(f'train -> MSE: {avg_reconstruction_loss/lambda_recon}, MSE lambda: {avg_reconstruction_loss}, MI: {avg_mi_loss/alpha_mi}, MI alpha:{avg_mi_loss}, TC: {avg_tc_loss/beta_tc}, TC beta:{avg_tc_loss}, DW_KL: {avg_dw_kl_loss/gamma_dw_kl}, DW_KL gamma:{avg_dw_kl_loss}')

            # Validation
            model.eval()
            val_loss_accum = 0.0

            val_reconstruction_loss_accum = 0.0
            val_mi_loss_accum = 0.0
            val_tc_loss_accum = 0.0
            val_dw_kl_loss_accum = 0.0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)

                    reconstruction_loss_value, mi, tc, dw_kl = model(data, reduction='avg')
                    reconstruction_loss_value = reconstruction_loss_value * lambda_recon
                    mi = mi * alpha_mi
                    tc = tc * beta_tc
                    dw_kl = dw_kl * gamma_dw_kl
                    loss = reconstruction_loss_value + mi + tc + dw_kl

                    val_loss_accum += loss.item()
                    val_reconstruction_loss_accum += reconstruction_loss_value.item()
                    val_mi_loss_accum += mi.item()
                    val_tc_loss_accum += tc.item()
                    val_dw_kl_loss_accum += dw_kl.item()

                    if torch.isnan(loss):  # Check for NaN loss
                        print(f"Warning: NaN loss detected in validation epoch {epoch + 1}. Pruning trial.")
                        raise optuna.exceptions.TrialPruned()
                    val_loss_accum += loss.item()

            avg_val_loss = val_loss_accum / len(val_loader)

            avg_val_reconstruction_loss = val_reconstruction_loss_accum / len(val_loader)
            avg_val_mi_loss = (val_mi_loss_accum / len(val_loader)) / alpha_mi
            avg_val_tc_loss = (val_tc_loss_accum / len(val_loader)) / beta_tc
            avg_val_dw_kl_loss = (val_dw_kl_loss_accum / len(val_loader)) / gamma_dw_kl

            print(
                f'valid -> MSE: {avg_val_reconstruction_loss / lambda_recon}, MSE lambda: {avg_val_reconstruction_loss}, MI: {avg_val_mi_loss / alpha_mi}, MI alpha:{avg_val_mi_loss}, TC: {avg_val_tc_loss / beta_tc}, TC beta:{avg_val_tc_loss}, DW_KL: {avg_val_dw_kl_loss / gamma_dw_kl}, DW_KL gamma:{avg_val_dw_kl_loss}')
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
    study_name = "malicious_traffic_latent_variable_gan_b_tcvae_v20"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name= study_name,
                                storage=storage_name,
                                load_if_exists=True,
                                direction='minimize')
    study.optimize(train_model, n_trials=1)
    # Get Best parameters
    best_params = study.best_params
    best_value = study.best_value
    print(f'best params: {best_params}')
    print(f'best elbo: {best_value}')
    mlflow.log_param('vae_params', str(best_params))

    # save optuna best trails to review parameters selection
    best_trails_df = study.trials_dataframe()
    best_trails_df = best_trails_df.sample(10)
    best_trails_filepath = os.path.join(results_folder_path, 'best_trails.csv')
    best_trails_df.to_csv(best_trails_filepath, index=False)
    mlflow.log_artifact(best_trails_filepath)

    # Retrain the best model
    # TODO:: The retraining is done from scratch using the best hyperparameters to ensure the values benefit the model

    # Init the Autoencoder, loss function metric and optimizer
    latent_dim = best_params['latent_dim']
    hidden_dim = best_params['hidden_dim']
    alpha_mi = best_params['alpha_mi']
    beta_tc = best_params['beta_tc']
    gamma_dw_kl = best_params['gamma_dw_kl']
    learning_rate = best_params['learning_rate']
    lambda_recon = best_params['lambda_recon']
    model: B_TCVAE = B_TCVAE(input_dim=n_features, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Early stopping is added to avoid overfitting
    early_stopping = EarlyStopping(patience=early_stopping_patience*2)
    num_epochs_val = num_epochs

    # Training loop
    print(f'Training Beta-TCVAE. Best params: {best_params}')
    best_val_loss = float('inf')
    for epoch in range(num_epochs_val):
        epoch_start_time = time.time()
        model.train()
        train_loss_accum = 0.0

        reconstruction_loss_accum = 0.0
        mi_loss_accum = 0.0
        tc_loss_accum = 0.0
        dw_kl_loss_accum = 0.0

        for data in train_loader:
            # Forward pass
            data = data.to(device)
            optimizer.zero_grad()

            # use the model
            reconstruction_loss_value, mi, tc, dw_kl = model(data, reduction='avg')
            # multiply by the factors
            reconstruction_loss_value = reconstruction_loss_value * lambda_recon
            mi = mi * alpha_mi
            tc = tc * beta_tc
            dw_kl = dw_kl * gamma_dw_kl
            # sum total loss
            loss = reconstruction_loss_value + mi + tc + dw_kl

            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item()

            reconstruction_loss_accum += reconstruction_loss_value.item()
            mi_loss_accum += mi.item()
            tc_loss_accum += tc.item()
            dw_kl_loss_accum += dw_kl.item()

        # Calculate train batch loss
        avg_train_loss = train_loss_accum / len(train_loader)
        # Calculate train KL Divergence and MSE
        avg_reconstruction_loss = reconstruction_loss_accum / len(train_loader)
        avg_mi_loss = mi_loss_accum / len(train_loader)
        avg_tc_loss = tc_loss_accum / len(train_loader)
        avg_dw_kl_loss = dw_kl_loss_accum / len(train_loader)

        print(f'train -> MSE: {avg_reconstruction_loss/lambda_recon}, MSE lambda: {avg_reconstruction_loss}, MI: {avg_mi_loss/alpha_mi}, MI alpha:{avg_mi_loss}, TC: {avg_tc_loss/beta_tc}, TC beta:{avg_tc_loss}, DW_KL: {avg_dw_kl_loss/gamma_dw_kl}, DW_KL gamma:{avg_dw_kl_loss}')

        # log metrics
        mlflow.log_metric('train/avg_mi_loss', avg_mi_loss/alpha_mi, step=epoch)
        mlflow.log_metric('train/tc_loss_accum', avg_tc_loss/beta_tc, step=epoch)
        mlflow.log_metric('train/dw_kl_loss_accum', avg_dw_kl_loss/gamma_dw_kl, step=epoch)
        mlflow.log_metric('train/mse_value', avg_reconstruction_loss/lambda_recon, step=epoch)
        mlflow.log_metric('train/neg_elbo_value', avg_train_loss, step=epoch)

        # Validation
        model.eval()

        val_loss_accum = 0.0
        val_reconstruction_loss_accum  = 0.0
        val_mi_loss_accum = 0.0
        val_tc_loss_accum = 0.0
        val_dw_kl_loss_accum = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)

                reconstruction_loss_value, mi, tc, dw_kl = model(data, reduction='avg')
                # multiply by the factors
                reconstruction_loss_value = reconstruction_loss_value * lambda_recon
                mi = mi * alpha_mi
                tc = tc * beta_tc
                dw_kl = dw_kl * gamma_dw_kl
                # sum total loss
                loss = reconstruction_loss_value + mi + tc + dw_kl

                val_loss_accum += loss.item()
                val_reconstruction_loss_accum += reconstruction_loss_value.item()
                val_mi_loss_accum += mi.item()
                val_tc_loss_accum += tc.item()
                val_dw_kl_loss_accum += dw_kl.item()

        # Calculate validation batch loss
        avg_val_loss = val_loss_accum / len(val_loader)
        # Calculate train KL Divergence and MSE
        avg_val_reconstruction_loss = val_reconstruction_loss_accum / len(val_loader)
        avg_val_mi_loss = (val_mi_loss_accum / len(val_loader)) / alpha_mi
        avg_val_tc_loss = (val_tc_loss_accum / len(val_loader)) / beta_tc
        avg_val_dw_kl_loss = (val_dw_kl_loss_accum / len(val_loader)) / gamma_dw_kl

        print(
            f'valid -> MSE: {avg_val_reconstruction_loss / lambda_recon}, MSE lambda: {avg_val_reconstruction_loss}, MI: {avg_val_mi_loss / alpha_mi}, MI alpha:{avg_val_mi_loss}, TC: {avg_val_tc_loss / beta_tc}, TC beta:{avg_val_tc_loss}, DW_KL: {avg_val_dw_kl_loss / gamma_dw_kl}, DW_KL gamma:{avg_val_dw_kl_loss}')
        epoch_duration = time.time() - epoch_start_time
        print(
            f'  Epoch [{epoch + 1}/{num_epochs_val}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Duration: {epoch_duration:.2f}s')

        # log metrics
        mlflow.log_metric('validation/avg_mi_loss', avg_val_mi_loss / alpha_mi, step=epoch)
        mlflow.log_metric('validation/tc_loss_accum', avg_val_tc_loss / beta_tc, step=epoch)
        mlflow.log_metric('validation/dw_kl_loss_accum', avg_val_dw_kl_loss / gamma_dw_kl, step=epoch)
        mlflow.log_metric('validation/mse_value', avg_val_reconstruction_loss / lambda_recon, step=epoch)
        mlflow.log_metric('validation/neg_elbo_value', avg_val_loss, step=epoch)

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

