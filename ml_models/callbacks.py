class EarlyStopping:
    """
        class EarlyStopping:

        EarlyStopping is a utility class to stop training when a monitored loss has stopped improving.

        :param patience: Number of epochs to wait after the last improvement to stop the training process. Default is 10.
        :type patience: int
        :param min_delta: Minimum change in the monitored loss to qualify as an improvement. Default is 0.
        :type min_delta: float
    """

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, loss):
        """
            Performs a step in early stopping by checking if the loss has improved.
            Updates the best loss if there is an improvement, resets the counter if the loss is improved.
            Increments the counter if no improvement. Returns a boolean indicating whether to stop training.

            :param loss: The current loss value to compare with the best loss.
            :type loss: float

            :returns: True if the training should be stopped, False otherwise.
            :rtype: bool
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        return False
