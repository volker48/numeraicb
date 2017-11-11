==========================
Keras Consistency Callback
==========================

A Keras callback that calculates your model's consistency during training at
each epoch. The callback prints the consistency and also adds the consistency at
the end of each epoch to the training history under the ``consistency`` key.

Usage
-----

Here is a usage example::

    import pandas as pd
    from numeraicb import Consistency
    from keras.models import Sequential
    from keras.layers.core import Dense

    train = pd.read_csv('numerai_training_data.csv')
    tourn = pd.read_csv('numerai_tournament_data.csv')

    validation = tourn[tourn.data_type == 'validation']

    features = ['feature{}'.format(i) for i in range(1, 51)]

    X = train[features].values
    Y = train.target.values

    X_validation = validation[features].values
    Y_validation = validation.target.values

    model = Sequential()
    model.add(Dense(30, kernel_initializer='uniform', input_dim=X.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adamax', loss='binary_crossentropy')

    cb = Consistency(tourn)

    # Now your models consistency will be printed at each epoch
    history = model.fit(X, Y, callbacks=[cb], validation_data=(X_validation, Y_validation))

    # Consistency is stored in the history as well
    for epoch, consistency in enumerate(history.history['consistency']):
        print('consistency at epoch {}: {:.2%}'.format(epoch, consistency))











