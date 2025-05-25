"""
Test case evaluation
"""
from pipeline.evaluation.evaluation_interpolation_traffic_b_tcvae import evaluation_interpolation


def test_evaluation():
    generator_model_filepath = 'beta_btcvae_traffic_generator_model.pth'
    model_discriminator_filepath = 'xgb_server_traffic_classifier.json'
    scale_model_filepath = 'scaler.pkl'
    label_encoder_model_filepath = 'label_encoder.pkl'
    traffic_data_filepath = 'traffic_preprocessed_test_btcvae.csv'
    results_folder_path = 'results'

    evaluation_interpolation(model_generator_filepath=generator_model_filepath,
                             model_discriminator_filepath=model_discriminator_filepath,
                             scale_model_filepath=scale_model_filepath,
                             label_encoder_model_filepath=label_encoder_model_filepath,
                             traffic_data_filepath=traffic_data_filepath,
                             results_folder_path=results_folder_path)