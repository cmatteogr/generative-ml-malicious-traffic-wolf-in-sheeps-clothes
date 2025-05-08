"""
Author: Cesar M. Gonzalez

"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import optuna
import pandas as pd
import mlflow
import time
from ml_models.callbacks import EarlyStopping
from ml_models.malicious_traffic_latent_variable_gan import VAE


def train(traffic_data_filepath: str, train_size_percentage=0.8, batch_size=1024):
    """
    VAE training

    :param traffic_data_filepath: Traffic dataset
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
    num_epochs = 300
    early_stopping_patience = 15
    kl_beta = 0.2

    # Build the model tunner using optuna
    print('build VAE for generation model')

    def train_model(trial):
        # Init the Hyperparameters to change
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
        hidden_dim = trial.suggest_int('hidden_dim', 24, 38)
        latent_dim = trial.suggest_int('latent_dim', 12, 22)

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

                reconstruction_loss_value, kl_value = model(data, reduction='avg')  # Use average loss over batch
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

            # log metrics
            mlflow.log_metric('neg_elbo_value', avg_train_loss)

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
    study_name = "malicious_traffic_latent_variable_gan_v3"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name= study_name,
                                storage=storage_name,
                                load_if_exists=True,
                                direction='minimize', )
    study.optimize(train_model, n_trials=150)
    # Get Best parameters
    best_params = study.best_params
    print('best params: {}'.format(best_params))
    mlflow.log_param('vae_params', str(best_params))
    mlflow.log_metric("best_elbo", study.best_value)

