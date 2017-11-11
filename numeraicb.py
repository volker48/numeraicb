from keras.callbacks import Callback
from math import log
from sklearn.metrics import log_loss


class Consistency(Callback):
    """
    Callback class that calculates Numerai consistency metric at each epoch
    of training. It also adds the consistency to the training history.
    """

    def __init__(self, tournament_df):
        """
        :param tournament_df: Pandas DataFrame containing the Numerai tournament data
        """
        super(Consistency, self).__init__()
        self.era_indices = self._get_era_indices(tournament_df)

    def _get_era_indices(self, tournament_df):
        indices = []
        validation = tournament_df[tournament_df.data_type == 'validation']
        for era in validation.era.unique():
            indices.append(validation.era == era)
        return indices

    def _consistency(self, era_indices, validation_y, validation_yhat):
        num_better_random = 0.0
        for indices in era_indices:
            labels = validation_y[indices]
            era_preds = validation_yhat[indices]
            ll = log_loss(labels, era_preds)
            if ll < -log(.5):
                num_better_random += 1.0
        return num_better_random / len(era_indices)

    def on_epoch_end(self, epoch, logs=None):
        y_hat = self.model.predict(self.validation_data[0])
        y = self.validation_data[1]
        c = self._consistency(self.era_indices, y, y_hat)
        if logs is not None:
            logs['consistency'] = c
        print('  consistency: {:.2%}'.format(c))
