"""
Test case evaluation
"""
from pipeline.evaluation.evaluation_latent_variable_gan import evaluation


def test_evaluation():
    test_traffic_filepath = 'traffic_preprocessed_test_btcvae.csv'
    model_filepath = 'beta_btcvae_traffic_generator_model.pth'
    results_folder_path = 'results'

    evaluation(model_filepath=model_filepath,
               data_test_filepath=test_traffic_filepath,
               results_folder_path=results_folder_path)