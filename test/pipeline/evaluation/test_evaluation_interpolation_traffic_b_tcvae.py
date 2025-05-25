"""
Test case evaluation
"""
from pipeline.evaluation.evaluation_interpolation_traffic_b_tcvae import evaluation_interpolation


def test_evaluation():
    test_traffic_filepath = 'traffic_preprocessed_test_btcvae.csv'
    model_hyperparams_filepath = ''
    model_filepath = 'beta_btcvae_traffic_generator_model.pth'
    results_folder_path = 'results'
    model_discriminator_filepath = 'xgb_server_traffic_classifier.json'
    scale_model_filepath = 'scaler.pkl'
    label_encoder_model_filepath ='label_encoder.pkl'

    evaluation_interpolation(model_generator_filepath=model_filepath,
                             model_discriminator_filepath=model_discriminator_filepath,
                             label_encoder_model_filepath=label_encoder_model_filepath,
                             scale_model_filepath=scale_model_filepath,
                             data_test_filepath=test_traffic_filepath,
                             results_folder_path=results_folder_path)