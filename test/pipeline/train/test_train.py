from pipeline.train.training import train


def test_preprocess():
    train_traffic_filepath = '/home/cesarealice/PycharmProjects/generative-ml-malicious-traffic-wolf-in-sheeps-clothes/test/pipeline/preprocess/traffic_preprocessed_train.csv'

    train(train_traffic_filepath)