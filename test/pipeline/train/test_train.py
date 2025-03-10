from pipeline.train.train_traffic_classifier import train


def test_train():
    train_traffic_filepath = '/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/test/pipeline/preprocess/traffic_preprocessed_train.csv'
    model_filepath = "xgb_server_traffic_classifier.json"
    train(train_traffic_filepath, model_filepath)