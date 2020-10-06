import numpy as np
import keras
import copy
from keras import regularizers
from keras.models import Model
# noinspection PyPep8Naming
from keras import backend as K
from keras.utils import get_custom_objects
from keras.layers import Input, Softmax, Embedding, Add, Dense, Dot, Lambda, RepeatVector
from keras.layers import Reshape, Concatenate, Activation, GlobalMaxPooling1D
from keras.layers import SeparableConv1D 
from keras.layers import Conv1D

from models.keras_transformer.extras import ReusableEmbedding
from models.keras_transformer.masks import PaddingMaskLayer, SingleSequenceMaskLayer
from models.functions import ElementWiseProduct
from models.keras_transformer.position import TransformerCoordinateEmbedding
from models.keras_transformer.transformer_blocks import TransformerEncoderBlock
from models.functions import QueryRetrievalEncoderMask
from models.auto_pointer_selector import UsefulSentenceAutoPointer
from models.auto_pointer_merge import AutoPointerMerger


class AutoPointerDssmModel:
    def __init__(self, args, tau,
                 transformer_dropout: float = 0.05,
                 embedding_dropout: float = 0.05,
                 l2_reg_penalty: float = 1e-4,
                 use_same_embedding = True,
                 use_vanilla_transformer = True,
                 ):
        self.args = args
        self.tau = tau
        self.pos_number = args.positive_number
        self.neg_number = args.negative_number
        self.query_retrieval_number = args.query_retrieval_number
        self.semantic_dim = args.semantic_dim
        self.transformer_dropout = transformer_dropout 
        self.embedding_dropout = embedding_dropout

        self.query_dense = Dense(self.semantic_dim, activation='tanh', name='query_sem')
        self.query_retrieval_dense = Dense(self.semantic_dim, activation='tanh', name='query_retrieval_sem')
        self.fact_dense = Dense(self.semantic_dim, activation='tanh', name='fact_sem')
        self.semantic_dim_dense = Dense(self.args.semantic_dim, activation='tanh', name='semantic_dim_sem')

        self.query_conv = SeparableConv1D(self.args.embedding_dim, self.args.max_pooling_filter_length, padding="same", activation="tanh")
        self.query_max = GlobalMaxPooling1D(data_format='channels_last', name='query_max_pooling')
        self.fact_conv = SeparableConv1D(self.args.embedding_dim, self.args.max_pooling_filter_length, padding="same", activation="tanh")
        self.fact_max = GlobalMaxPooling1D(data_format='channels_last', name='fact_max_pooling')

        self.cosine_merger_layer = AutoPointerMerger(name='cosine_merger', args=self.args)

        # prepare layers
        l2_regularizer = (regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)
        if use_same_embedding:
            self.query_embedding_layer = self.fact_embedding_layer = ReusableEmbedding(
                self.args.vocab_size, self.args.embedding_dim,
                name='embeddings',
                # Regularization is based on paper "A Comparative Study on
                # Regularization Strategies for Embedding-based Neural Networks"
                # https://arxiv.org/pdf/1508.03721.pdf
                embeddings_regularizer=l2_regularizer)
        else:
            self.query_embedding_layer = ReusableEmbedding(
                self.args.vocab_size, self.args.embedding_dim,
                name='query_embeddings',
                embeddings_regularizer=l2_regularizer)
            self.fact_embedding_layer = ReusableEmbedding(
                self.args.vocab_size, self.args.embedding_dim,
                name='fact_embeddings',
                embeddings_regularizer=l2_regularizer)

        self.query_coord_embedding_layer = TransformerCoordinateEmbedding(
            self.args.src_seq_length,
            1 if use_vanilla_transformer else self.args.transformer_depth,
            name='query_coordinate_embedding')

        self.output_softmax_layer = Softmax(name='pos_neg_predictions')

        self.query_encoder_blocks = [TransformerEncoderBlock(
            name='query_encoder%s'%i, 
            num_heads=self.args.num_heads,
            residual_dropout=self.transformer_dropout,
            attention_dropout=self.transformer_dropout,
            activation='relu',
            vanilla_wiring=True) for i in range(self.args.transformer_depth)]


    def __get_query_encoder(self, input_layer, pad_id, _name):
        #print('This is Query Encoder...')
        self_attn_mask = PaddingMaskLayer(src_len=self.args.src_seq_length, pad_id=pad_id)(input_layer)

        next_step_input, _ = self.query_embedding_layer(input_layer)
        next_step_input = self.query_coord_embedding_layer(next_step_input, step=0)
        for i in range(self.args.transformer_depth):
            next_step_input = self.query_encoder_blocks[i]([next_step_input, self_attn_mask])

        return next_step_input

    def __get_query_encoder2(self, input_layer, pad_id, _name):
        #print('This is Query Encoder...')
        self_seq_mask = SingleSequenceMaskLayer(pad_id=pad_id)(input_layer)
        next_step_input, _ = self.query_embedding_layer(input_layer)
        next_step_input = ElementWiseProduct()([next_step_input, self_seq_mask])
        return next_step_input


    def __get_fact_encoder(self, input_layer, pad_id, _name):
        #print('This is Fact Encoder...')
        self_seq_mask = SingleSequenceMaskLayer(pad_id=pad_id)(input_layer)
        next_step_input, _ = self.fact_embedding_layer(input_layer)
        next_step_input = ElementWiseProduct()([next_step_input, self_seq_mask])
        return next_step_input

    def get_model(self, pad_id):
        q_tokens = Input(
            shape=(None, ),
            name='q_tokens'
        )

        qr_masks = [Input(
            shape=(None, ),
            name='qr_masks_%s'%j) for j in range(self.query_retrieval_number)
        ]

        qr_tokens = [Input(
            shape=(None, ),
            name='q_retrieval_%s'%j) for j in range(self.query_retrieval_number)
        ]

        pos_tokens = [Input(
            shape=(None, ),
            name='pos_tokens_%s'%j) for j in range(self.pos_number)
        ]
        
        neg_tokens = [Input(
            shape=(None, ),
            name='neg_fact_input_%s'%j) for j in range(self.neg_number)
        ]

        enc_query_output = self.__get_query_encoder(q_tokens, pad_id, 'query')
        enc_qrs_output = [self.__get_query_encoder(q_retrieval, pad_id, 'qr%s'%index) for index, q_retrieval in \
            enumerate(qr_tokens)]
        enc_pos_facts_output = [self.__get_fact_encoder(pos_fact, pad_id, 'pos%s'%index) for index, pos_fact in \
            enumerate(pos_tokens)]
        enc_neg_facts_output = [self.__get_fact_encoder(neg_fact, pad_id, 'neg%s'%index) for index, neg_fact in \
            enumerate(neg_tokens)]

        # query convolution and maxpooling
        query_emb = self.query_max(enc_query_output)
        query_sem = query_emb

        qrs_emb = QueryRetrievalEncoderMask(self.query_retrieval_number)(qr_masks + enc_qrs_output)

        # comment for 1 qr
        qrs_emb = [self.query_max(qr_emb) for qr_emb in qrs_emb]
        qrs_sem = qrs_emb

        # just choose one sentence, so the shape: bs, embedding_dim
        useful_sent = UsefulSentenceAutoPointer(tau=self.tau,
            batch_size=self.args.batch_size,
            query_retrieval_number=self.query_retrieval_number,
            use_transition=True)(qrs_sem )

        pos_facts_emb = [self.fact_max(pos_fact) for pos_fact in enc_pos_facts_output]
        neg_facts_emb = [self.fact_max(neg_fact) for neg_fact in enc_neg_facts_output]

        pos_facts_sem = pos_facts_emb
        neg_facts_sem = neg_facts_emb

        # cosine similarity
        #query_pos_facts_cosine = [Lambda(lambda x:K.sigmoid(x))(Dot(axes=1, normalize=True)([query_sem, pos_fact_sem])) for pos_fact_sem in pos_facts_sem]
        #query_neg_facts_cosine = [Lambda(lambda x:K.sigmoid(x))(Dot(axes=1, normalize=True)([query_sem, neg_fact_sem])) for neg_fact_sem in neg_facts_sem]
        query_pos_facts_cosine = [Dot(axes=1, normalize=True)([query_sem, pos_fact_sem]) for pos_fact_sem in pos_facts_sem]
        query_neg_facts_cosine = [Dot(axes=1, normalize=True)([query_sem, neg_fact_sem]) for neg_fact_sem in neg_facts_sem]
        auto_pos_facts_cosine = [Dot(axes=1, normalize=True)([useful_sent, pos_fact_sem]) for pos_fact_sem in pos_facts_sem]
        auto_neg_facts_cosine = [Dot(axes=1, normalize=True)([useful_sent, neg_fact_sem]) for neg_fact_sem in neg_facts_sem]

        query_pos_facts_cosine = [self.cosine_merger_layer(query_pos_facts_cosine + auto_pos_facts_cosine)]
        query_neg_facts_cosine = self.cosine_merger_layer(query_neg_facts_cosine + auto_neg_facts_cosine)

        concat_cosine = Concatenate()(query_pos_facts_cosine + query_neg_facts_cosine)

        concat_cosine = Reshape((self.pos_number + self.neg_number, 1))(concat_cosine)
        # gamma 
        weight = np.asarray([1]).reshape(1, 1, 1)
        with_gamma = Conv1D(1, 1, padding='same', input_shape=(self.pos_number + self.neg_number, 1),
            activation='linear', use_bias=False, weights=[weight])(concat_cosine)
        with_gamma = Reshape((self.pos_number + self.neg_number, ))(with_gamma)

        # softmax
        prob = self.output_softmax_layer(with_gamma)
        
        model = Model(
            inputs=[q_tokens ] + qr_masks + qr_tokens + pos_tokens + neg_tokens,
            outputs=prob
        )

        model_pos_consine = K.function([q_tokens] + qr_masks + qr_tokens + pos_tokens, query_pos_facts_cosine )
        query_embeddings = [enc_query_output]
        pos_embeddings = enc_pos_facts_output
        query_word_embedding_fn = K.function([q_tokens], query_embeddings )
        pos_embedding_fn = K.function(pos_tokens, pos_embeddings )

        return model, model_pos_consine, query_word_embedding_fn, pos_embedding_fn  


