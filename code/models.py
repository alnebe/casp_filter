import numpy as np

class ANNs:
    @staticmethod
    def create_baseline(input_shape):

        import tensorflow
        from tensorflow.keras.layers import Dense, BatchNormalization
        from tensorflow.keras.models import Sequential

        # create model
        model = Sequential()
        model.add(Dense(1000,
                        activation='relu',
                        input_shape=input_shape))
        model.add(BatchNormalization(trainable=True))
        model.add(Dense(1,
                        activation='sigmoid'))
        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[BaseMetrics.precision,
                               BaseMetrics.recall,
                               AccMetrics.f1,
                               AccMetrics.fbeta,
                               AccMetrics.bac,
                               AccMetrics.acc,
                               tensorflow.keras.metrics.AUC(name="auc")],
                      run_eagerly=True)
        # print model representation
        model.summary()
        return model

    @staticmethod
    def mlp_classifier(input_shape):

        import tensorflow
        from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
        from tensorflow.keras.models import Sequential

        # create model
        model = Sequential()
        model.add(Dense(1000,
                        activation='relu',
                        input_shape=input_shape))
        model.add(BatchNormalization(trainable=True))
        model.add(Dropout(0.3))
        model.add(Dense(500,
                        activation='relu'))
        model.add(BatchNormalization(trainable=True))
        model.add(Dense(1,
                        activation='sigmoid'))
        # compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[BaseMetrics.precision,
                               BaseMetrics.recall,
                               AccMetrics.f1,
                               AccMetrics.fbeta,
                               AccMetrics.bac,
                               AccMetrics.acc,
                               tensorflow.keras.metrics.AUC(name="auc")],
                      run_eagerly=True)
        # print model representation
        model.summary()
        return model


def generator(validation_set, batch_size, desc_dict):
    samples_per_epoch = np.array(validation_set).shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter = 0
    while True:
        x_data = validation_set[batch_size*counter:batch_size*(counter+1)]
        x_batch = np.stack([desc_dict[x][0][0] for x in x_data])
        y_batch = np.stack([desc_dict[x][1] for x in x_data])
        counter += 1
        yield x_batch, y_batch
        if counter >= number_of_batches:
            counter = 0
