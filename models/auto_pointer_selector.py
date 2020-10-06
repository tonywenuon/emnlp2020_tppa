import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from keras.utils import get_custom_objects


class UsefulSentenceAutoPointer(Layer):
    """
    """
    def __init__(self, 
                 tau,
                 batch_size,
                 query_retrieval_number,
                 multiplier=4,
                 use_transition=True,
                 **kwargs):
        self.tau = tau
        self.batch_size = batch_size
        self.qr_number = query_retrieval_number
        self.multiplier = multiplier
        self.use_transition = use_transition
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['tau'] = self.tau
        config['batch_size'] = self.batch_size
        config['query_retrieval_number'] = self.qr_number
        config['multiplier'] = self.multiplier
        config['use_transition'] = self.use_transition
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        # comment for qr
        d_model = input_shape[0][-1]
        #d_model = input_shape[-1]
        if self.use_transition:
            self.W1 = self.add_weight(
                shape=(d_model, self.multiplier * d_model),
                initializer='uniform',
                name='W1',
                trainable=True)
            self.b1 = self.add_weight(
                shape=(self.multiplier * d_model, ),
                initializer='uniform',
                name='b1',
                trainable=True)
            self.W2 = self.add_weight(
                shape=(self.multiplier * d_model, d_model),
                initializer='uniform',
                name='W2',
                trainable=True)
            self.b2 = self.add_weight(
                shape=(d_model, ),
                initializer='uniform',
                name='b2',
                trainable=True)
        self.selector = self.add_weight(
            shape=(d_model, 1),
            initializer='uniform',
            name='selector',
            trainable=True)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 0:
            raise ValueError('Please input right shape to UsefulWordAutoPointer, received length: %s'%len(input_shape))
        # comment for qr
        enc_output = input_shape[0]
        #enc_output = input_shape
        # 2 means: 1 selector and 1 sentence
        return (enc_output[0], enc_output[-1])

    def gumbel_sampling(self, logits):
        ep = 1e-20
        U = K.random_uniform(K.shape(logits), 0, 1)                                                                                   
        # add Gumbel noise
        y = logits - K.log(-K.log(U + ep) + ep)
        y = K.softmax(y / self.tau)

        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keepdims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
        #y = y_hard
        return y

    def apply_dropout_if_needed(self, _input, training=None):
        def dropped_softmax():
            return K.dropout(_input, 0.5)

        return K.in_train_phase(dropped_softmax, _input,
                                    training=training)


    def call(self, inputs, **kwargs):
        #qrs_sem, qr_tags = inputs[:self.qr_number], inputs[-1]
        qrs_sem = inputs
        # comment for qr
        qrs_sem = [K.expand_dims(qr_sem, axis=1) for qr_sem in qrs_sem]
        qrs_sem = K.concatenate(qrs_sem, axis=1)


        d_model = qrs_sem.shape[-1]
        # there are query_retrieval_number samples
        final_qrs_sem = qrs_sem
        if self.use_transition:
            qr_trans = K.reshape(qrs_sem, (-1, d_model))
            qr_trans = K.dot(qr_trans, self.W1)
            qr_trans = K.bias_add(qr_trans, self.b1)

            qr_trans = K.dot(qr_trans, self.W2)
            qr_trans = K.bias_add(qr_trans, self.b2)

            qr_trans = K.reshape(qr_trans, (-1, self.qr_number, d_model))
            final_qrs_sem = qr_trans

        final_qrs_sem = self.apply_dropout_if_needed(final_qrs_sem)
        final_qrs_sem = K.reshape(final_qrs_sem, (-1, d_model))

        auto_sent = K.dot(final_qrs_sem, self.selector)
        auto_sent = K.reshape(auto_sent, (-1, self.qr_number))

        sample = self.gumbel_sampling(auto_sent)
        sample = K.reshape(sample, (-1, self.qr_number, 1))

        # previous gumbel softmax
        final_qrs_sem = K.reshape(final_qrs_sem, (-1, self.qr_number, d_model))
        sample = K.reshape(sample, (-1, self.qr_number, 1))
        final_qrs_sem = final_qrs_sem * sample
        final_qrs_sem = K.max(final_qrs_sem, axis=1)

        return final_qrs_sem



get_custom_objects().update({
    'UsefulSentenceAutoPointer': UsefulSentenceAutoPointer,
})


