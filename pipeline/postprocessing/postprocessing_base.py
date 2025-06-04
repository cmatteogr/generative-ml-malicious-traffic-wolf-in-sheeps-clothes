"""
postprocessing base operations
"""

def post_process_data(features):
    # clone tensor
    processed_features = features.clone()
    # format binary columns
    # define binary columns, this is the index
    # NOTE: if the order of the features change or any other is added this index could change
    # before transform the df to tensor the index could be collected to use them here, calculate them automatically
    binary_columns = [22, 23, 36, 37, 38, 39, 40, 41]
    # define threshold index
    binary_threshold = 0.5
    processed_features[:, binary_columns] = (processed_features[:, binary_columns] >= binary_threshold).float()
    return processed_features