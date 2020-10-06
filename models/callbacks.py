import numpy as np
import keras
from keras.utils import get_custom_objects
from keras import backend as K

class TauDecay(keras.callbacks.Callback):
    def __init__(self, tau, **kargs):
        self.tau = tau
        self.anneal_rate = 0.00003
        self.min_temperature = 0.1

    def on_train_end(self, logs={}):
        tau_value = K.get_value(self.tau)
        print('Tau Value: %s' % tau_value)

    def on_epoch_end(self, epoch, logs={}):
        tau_value = K.get_value(self.tau)
        print('Epoch %s, Current Tau Value: %s' % (epoch + 1, tau_value))

    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            decay_rate = np.exp(-self.anneal_rate * batch)
            tau_value = K.get_value(self.tau) * decay_rate
            tau_value = np.max([tau_value, self.min_temperature])
            K.set_value(self.tau, tau_value)


get_custom_objects().update({
    'TauDecay': TauDecay,
})


