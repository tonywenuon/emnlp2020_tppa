import numpy as np
import keras
from keras import backend as K
from keras.layers import Layer
from keras.utils import get_custom_objects


class QueryRetrievalEncoderMask(Layer):
    """
    """
    def __init__(self, 
                 query_retrieval_number,
                 **kwargs):
        """
        :param kwargs: any extra arguments typical for a Keras layer,
          such as name, etc.
        """
        self.qr_number = query_retrieval_number
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['query_retrieval_number'] = self.qr_number
        return config


    def compute_output_shape(self, input_shape):
        shape1 = input_shape[-1]
        ret = []
        for i in range(self.qr_number):
            ret.append(shape1)
        return ret

    def call(self, inputs, **kwargs):
        qr_masks = inputs[:self.qr_number]
        enc_outputs = inputs[self.qr_number:]
        assert(len(qr_masks) == len(enc_outputs))
        qr_masks = [K.expand_dims(qr_mask, axis=1) for qr_mask in qr_masks]
        ret = []
        for qr_mask, enc_output in zip(qr_masks, enc_outputs):
            new_enc_output = enc_output * qr_mask
            ret.append(new_enc_output)

        return ret

class ElementWiseProduct(Layer):
    """
    """
    def __init__(self, 
                 **kwargs):
        """
        :param kwargs: any extra arguments typical for a Keras layer,
          such as name, etc.
        """
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        #config[''] = self.
        return config

    def compute_output_shape(self, input_shape):
        shape1 = input_shape[0]
        return shape1

    def call(self, inputs, **kwargs):
        input_layer, self_mask = inputs
        ret = input_layer * self_mask
        return ret


get_custom_objects().update({
    'QueryRetrievalEncoderMask': QueryRetrievalEncoderMask,
    'ElementWiseProduct': ElementWiseProduct,
})

