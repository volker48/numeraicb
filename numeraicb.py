from keras.callbacks import Callback
from math import log
from sklearn.metrics import log_loss


class Consistency(Callback):
    def __init__(self, tournament_df):
        super(Consistency, self).__init__()
        self.era_indices = self.get_era_indices(tournament_df)

    def get_era_indices(self, tournament_df):
        indices = []
        validation = tournament_df[tournament_df.data_type == 'validation']
        for era in validation.era.unique():
            indices.append(validation.era == era)
        return indices

    def consistency(self, era_indices, validation_y, validation_yhat):
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
        c = self.consistency(self.era_indices, y, y_hat)
        if logs is not None:
            logs['consistency'] = c
        print('  consistency: {:.2%}'.format(c))
