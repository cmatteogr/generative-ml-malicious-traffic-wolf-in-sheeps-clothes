"""
Author: Cesar M. Gonzalez

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import optuna
import json
import pandas as pd

from ml_models.callbacks import EarlyStopping
from ml_models.malicious_traffic_latent_variable_gan import VAE


def train(traffic_data_filepath: str, train_size_percentage=0.8, batch_size=512):
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
    assert 1 <= batch_size <= 512, 'Batch size should be between 1 and 512.'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to PyTorch tensor
    traffic_df = pd.read_csv(traffic_data_filepath)
    # remove label column
    # NOTE: This feature may be needed in the future to build the CGAN
    # traffic_df.pop('Label')
    n_features = len(traffic_df.columns)
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

    # Build the model tunner using optuna
    print('build VAE for generation model')

    def train_model(trial):
        # Init the Hyperparameters to change
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        # Init the Autoencoder, loss function metric and optimizer\
        # Instantiate the VAE (Continuous)
        # TODO: Update VAE to allow update input dimension and latent dimension as hyperparameter: n_features, L
        model: VAE = VAE().to(device)


        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Early stopping is added to avoid overfitting
        early_stopping = EarlyStopping(patience=early_stopping_patience)

        # Training loop
        print('Training autoencoder anomaly detection model')
        best_trial_val_loss = float('inf')
        for epoch in range(num_epochs):
            model.train()
            train_loss_accum = 0.0
            for data in train_loader:
                # Forward pass
                data = data.to(device)
                optimizer.zero_grad()

                loss = model(data, reduction='avg')  # Use average loss over batch

                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
            # Calculate train batch loss
            avg_train_loss = train_loss_accum / len(train_loader)

            # Validation
            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    loss = model(data, reduction='avg')
                    if torch.isnan(loss):  # Check for NaN loss
                        print(f"Warning: NaN loss detected in validation epoch {epoch + 1}. Pruning trial.")
                        raise optuna.exceptions.TrialPruned()
                    val_loss_accum += loss.item()
            avg_val_loss = val_loss_accum / len(val_loader)

            print(
                f'  Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

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
    study = optuna.create_study(direction='minimize')
    study.optimize(train_model, n_trials=50)
    # Get Best parameters
    best_params = study.best_params
    print('best params: {}'.format(best_params))

