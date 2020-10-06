import os, sys, time, math

project_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
if project_path not in sys.path:
    sys.path.append(project_path)

import tensorflow as tf
import argparse
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import get_custom_objects

from keras.backend.tensorflow_backend import set_session
from models.auto_pointer_sent_dssm import AutoPointerDssmModel
from models.callbacks import TauDecay
from commonly_used_code import helper_fn, config
from run_script.args_parser import auto_pointer_dssm_arguments
from data_reader_auto_pointer_dssm import DataSet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class AutoPointerDssm:
    def __init__(self, args):
        self.tau = K.variable(0.5, name='temperature')
        self.dssm_model = AutoPointerDssmModel(args=args, tau=self.tau, use_same_embedding=False)
        self.args = args
        exp_names = []
        exp_names.append(args.data_set)
        exp_names.append(args.exp_name)
        exp_names.append('pos')
        exp_names.append(str(args.positive_number))
        exp_names.append('neg')
        exp_names.append(str(args.negative_number))
        exp_names.append('qr')
        exp_names.append(str(args.query_retrieval_number))
        exp_names.append('apr')
        exp_names.append(str(args.auto_pointer_rate))
        #exp_names.append('layers')
        #exp_names.append(str(args.transformer_depth))
        exp_name = '_'.join(exp_names)

        # create experiment dir
        self.exp_dir= os.path.join(args.checkpoints_dir, exp_name)
        helper_fn.makedirs(self.exp_dir)
        hist_name = exp_name + '.hist'
        model_name = exp_name + '_final_model.h5'

        self.history_path = os.path.join(self.exp_dir, hist_name)
        self.model_path = os.path.join(self.exp_dir, model_name)

        outputs_dir = args.outputs_dir
        helper_fn.makedirs(outputs_dir)

        self.out_name = exp_name + '.dssm'
        self.out_path = os.path.join(outputs_dir, self.out_name)

    def compile_new_model(self, pad_id):
        model, model_pos_cosine, query_word_embedding_fn, pos_embedding_fn = self.dssm_model.get_model(pad_id)
        model.compile(
                       optimizer=Adam(lr=self.args.lr),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'],
                      )
        return model, model_pos_cosine, query_word_embedding_fn, pos_embedding_fn 

    def train(self):
        ds = DataSet(self.args)

        train_generator = ds.data_generator('train')
        valid_generator = ds.data_generator('valid')

        if os.path.exists(self.model_path):
            print('Loading model from: %s' % self.model_path)
            custom_dict = get_custom_objects()
            model = load_model(self.model_path, custom_objects=custom_dict)
        else:
            print('Compile new model...')
            model, _, _, _ = self.compile_new_model(ds.pad_id)

        model.summary()

        verbose = 1
        tau_decayer = TauDecay(self.tau)
        earlystopper = EarlyStopping(monitor='val_loss', patience=self.args.early_stop_patience, verbose=verbose)
        ckpt_name = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
        ckpt_path = os.path.join(self.exp_dir, ckpt_name)
        checkpoint = ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=verbose, save_weights_only=True, save_best_only=True, mode='min')
        lrate = ReduceLROnPlateau(
                                  monitor='val_loss', 
                                  factor=0.5, 
                                  patience=self.args.lr_decay_patience, 
                                  verbose=verbose, 
                                  mode='auto', 
                                  min_delta=0.0001, 
                                  cooldown=0, 
                                  min_lr=self.args.lr_min,
                                 )

        callback_list = [earlystopper, lrate, tau_decayer]
    
        hist = model.fit_generator(
                        generator=train_generator, 
                        steps_per_epoch=(ds.train_sample_num//self.args.batch_size + 1),
                        validation_data=valid_generator,
                        validation_steps=(ds.valid_sample_num//self.args.batch_size + 1),
                        epochs=self.args.epochs,
                        callbacks=callback_list, 
                       )
        with open(self.history_path,'w') as f:
            f.write(str(hist.history))

        #model.save(self.model_path)
        model.save_weights(self.model_path)

    def test(self):
        ds = DataSet(args)
        test_generator = ds.data_generator('test')
        test_facts = ds.get_test_question_retrieval_file()

        # load_model
        print('Loading model from: %s' % self.model_path)
        #custom_dict = get_custom_objects()
        #model = load_model(self.model_path, custom_objects=custom_dict)

        model, model_pos_cosine, query_word_embedding_fn, pos_embedding_fn= self.compile_new_model(ds.pad_id)
        model.load_weights(self.model_path)
        print('model loading done...')

        dssm_outobj = open(self.out_path, 'w')
        for batch, (ids, ret_list, y) in enumerate(test_generator):
            if batch > (ds.test_sample_num // self.args.batch_size):
                break
            print('Current batch: {}/{}. '.format(batch, ds.test_sample_num // self.args.batch_size))
            ids = np.asarray(ids)
            cur_batch_size = ids.shape[0]

            q_tokens = ret_list[0]
            qr_masks = []
            for i in range(self.args.query_retrieval_number):
                qr_masks.append(ret_list[1][:, i, :])
            qr_tokens = []
            for i in range(self.args.query_retrieval_number):
                qr_tokens.append(ret_list[2][:, i, :])

            # [batch_size, fact_id, score]
            preds = []
            for i in range(cur_batch_size):
                preds.append(dict())

            # iterate every facts
            for i in range(self.args.test_top_k):
                neg_tokens = ret_list[-1][:, i, :]

                simis = model_pos_cosine([q_tokens] + qr_masks + qr_tokens + [neg_tokens])

                # iterate each batch score
                for pos_index in range(self.args.positive_number):
                    for index, score in enumerate(simis[pos_index]):
                        if index >= len(preds):
                            continue
                        score = score[0]
                        preds[index][i] = score
            for index, dic_factid_score in enumerate(preds):
                qa_id = ids[index]
                # (fact_id, fact)
                dic_facts = test_facts[qa_id]

                seq = []
                seq.append(qa_id)
                if len(dic_factid_score) == 0:
                    seq.append('no_fact')
                else:
                    sorted_id_score = sorted(dic_factid_score.items(), key=lambda x:x[1], reverse=True)
                    for fact_id, score in sorted_id_score:
                        fact = dic_facts.get(fact_id, -1)
                        if fact == -1:
                            continue
                        merge = fact
                        seq.append(merge)

                write_line = '\t'.join(seq) + '\n'
                dssm_outobj.write(write_line)
                dssm_outobj.flush()

        dssm_outobj.close()
        print(self.out_path)

    def case_study(self):
        ds = DataSet(args)
        test_generator = ds.data_generator('test')
        test_facts = ds.get_test_question_retrieval_file()
        test_qas = ds.get_test_qa()
        dic_stop_words = ds.get_stop_word()

        # load_model
        print('Loading model from: %s' % self.model_path)
        model, model_pos_cosine, query_word_embedding_fn, pos_embedding_fn= self.compile_new_model(ds.pad_id)
        model.load_weights(self.model_path)
        print('model loading done...')

        vis_samples = []
        for batch, (ids, ret_list, y) in enumerate(test_generator):
            if batch > (ds.test_sample_num // self.args.batch_size):
                break
            print('Current batch: {}/{}. '.format(batch, ds.test_sample_num // self.args.batch_size))
            ids = np.asarray(ids)
            print('ids: ', ids.shape)
            cur_batch_size = ids.shape[0]
            q_tokens = ret_list[0]

            # iterate every facts
            test_top_k = self.args.test_top_k 
            test_top_k = 1
            fact_index = 0
            for i in range(test_top_k):
                neg_tokens = ret_list[-1][:, fact_index, :]
                query_embeddings = query_word_embedding_fn([q_tokens])
                fact_embeddings = pos_embedding_fn([neg_tokens])
                query_embeddings = np.asarray(query_embeddings )
                fact_embeddings = np.asarray(fact_embeddings)
                query_shape = query_embeddings.shape
                fact_shape = fact_embeddings.shape
                #query_shape :  (1, 32, 35, 300)
                #fact_shape:  (1, 32, 35, 300)
                print('query_shape : ', query_shape )
                print('fact_shape: ', fact_shape)

                #sample_index = 5
                for sample_index in range(cur_batch_size):

                    print('*' * 100)
                    qa_id = ids[sample_index]
                    if len(test_facts[qa_id]) < fact_index + 1:
                        continue
                    print('qa_id: ', qa_id)
                    qa = test_qas[qa_id]
                    ques = qa.strip().split('\t')[1]
                    res = qa.strip().split('\t')[2]
                    ques_tokens = ques.strip().split(' ')
                    fact = test_facts[qa_id][fact_index]
                    fact_tokens = fact.strip().split(' ')
                    print('question:', ques)
                    print('answer:', res)
                    print('fact:', fact)

                    first_query = query_embeddings[0][sample_index]
                    first_fact = fact_embeddings[0][sample_index]
                    simis = cosine_similarity(first_query, first_fact)
                    print(simis.shape)
                    for q_index, q_words_f_words_simis in enumerate(simis):
                        if q_index >= len(ques_tokens):
                            continue
                        cur_ques_word = ques_tokens[q_index]

                        dic_sample = dict()
                        dic_sample['id'] = qa_id
                        dic_sample['label'] = cur_ques_word
                        dic_sample['prediction'] = cur_ques_word
                        dic_sample['text'] = []
                        dic_sample['attention'] = []

                        dic_tmp = dict()
                        for f_index, s in enumerate(q_words_f_words_simis):
                            score = round(s, 4)
                            score = score.tolist()
                            if f_index < len(fact_tokens):
                                f_word = fact_tokens[f_index].strip()
                                dic_sample['text'].append(f_word)
                                if f_word in dic_stop_words:
                                    dic_sample['attention'].append(0.0)
                                    continue
                                else:
                                    dic_sample['attention'].append(score)
                                dic_tmp[f_word] = s
                        vis_samples.append(dic_sample)
                        sort_word_simi = sorted(dic_tmp.items(), key=lambda x:x[1], reverse=True)
                        seq = []
                        seq.append(cur_ques_word + '\t')
                        for word, simi in sort_word_simi:
                            ss = word + ':' + str(simi)
                            seq.append(ss)
                        write = ' '.join(seq)
                        print(write)
        with open('case_study.json', 'w') as f:
            json.dump(vis_samples, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    auto_pointer_dssm_arguments(parser)
    args = parser.parse_args()
    print(args)

    trans = AutoPointerDssm(args)
    trans.train()
    trans.test()





