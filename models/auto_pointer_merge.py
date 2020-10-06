import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from keras.utils import get_custom_objects


class AutoPointerMerger(Layer):
    """
    """
    def __init__(self,
                 args,
                 **kwargs):
        self.args = args
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['args'] = self.args
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        #length = len(input_shape)
        length = int(len(input_shape) / 2)
        ret = []
        for i in range(length):
            ret.append(input_shape[i])
        return ret

    def call(self, inputs, **kwargs):
        length = int(len(inputs) / 2)
        query_cosines = []
        for i in range(length):
            query_cosines.append(inputs[i])
        auto_cosines = []
        for i in range(length, 2 * length):
            auto_cosines.append(inputs[i])

        rate = self.args.auto_pointer_rate
        final_cosines = [rate * query_cosine + (1-rate)*auto_cosine for query_cosine, auto_cosine in zip(query_cosines, auto_cosines)]

        return final_cosines


get_custom_objects().update({
    'AutoPointerMerger': AutoPointerMerger,
})


