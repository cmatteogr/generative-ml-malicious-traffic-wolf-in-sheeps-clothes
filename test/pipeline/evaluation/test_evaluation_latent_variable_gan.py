"""
Test case evaluation
"""
from pipeline.evaluation.evaluation_latent_variable_gan import evaluation


def test_evaluation():
    test_traffic_filepath = '/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/results/traffic_preprocessed_test.csv'
    model_hyperparams_filepath = ''
    model_filepath = '/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/results/beta_vae_traffic_generator_model.pth'
    results_folder_path = '/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/results'
    evaluation(model_filepath=model_filepath,
               model_hyperparams_filepath=model_hyperparams_filepath,
               data_test_filepath=test_traffic_filepath,
               results_folder_path=results_folder_path)