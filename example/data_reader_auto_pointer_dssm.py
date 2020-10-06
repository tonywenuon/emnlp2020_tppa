
import sys, os
project_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
if project_path not in sys.path:
    sys.path.append(project_path)

import random
import argparse
import numpy as np
from copy import deepcopy
from typing import Callable, Optional, Sequence, Iterable
from commonly_used_code import config, helper_fn
from run_script.args_parser import auto_pointer_dssm_arguments
from models.tokenizer import Tokenizer

class DataSet:
    def __init__(self, args):
        self.args = args
        self.__set_file_path()

        # get global token and ids 
        self.src_token_ids, self.src_id_tokens, self.src_vocab_size = self.__read_global_ids(self.src_global_token_path)
        self.train_sample_num = 0
        self.valid_sample_num = 0
        self.test_sample_num = 0
        self.__get_sample_numbers()

        self.use_sentence_piece = False
        #self.use_sentence_piece = True
        self.tokenizer = Tokenizer(os.path.join(args.xlnet_model_dir, 'spiece.model'))

        if self.use_sentence_piece == True:
            self.pad_id = self.tokenizer.SYM_PAD
        else:
            self.pad_id = self.src_token_ids.get(config.SYM_PAD)

    def __read_global_ids(self, token_path):
        f = open(token_path)
        token_ids = dict()
        id_tokens = dict()
        vocab_size = 0
        for line in f:
            elems = line.strip().split('\t')
            word = elems[0]
            index = int(elems[1])
            token_ids[word] = index
            id_tokens[index] = word 
            vocab_size += 1

        return token_ids, id_tokens, vocab_size

    def __set_file_path(self):
        if self.args.data_set == 'wizard':
            self.train_set_path = config.wizard_train_qa_path
            self.valid_set_path = config.wizard_valid_qa_path
            self.test_set_path = config.wizard_test_qa_path

            self.train_positive_path = config.wizard_train_response_retrieval_data_path
            self.train_negative_path = config.wizard_train_negative_data_path 
            self.train_question_retrieval_path = config.wizard_train_question_retrieval_data_path 

            self.valid_positive_path = config.wizard_valid_response_retrieval_data_path
            self.valid_negative_path = config.wizard_valid_negative_data_path
            self.valid_question_retrieval_path = config.wizard_valid_question_retrieval_data_path 

            self.test_positive_path = config.wizard_test_question_retrieval_data_path 
            # there is no negative fact for test set
            self.test_negative_path = config.wizard_test_question_retrieval_data_path 
            self.test_question_retrieval_path = config.wizard_test_question_retrieval_data_path 

            self.src_global_token_path = config.wizard_global_token_path


    def get_test_question_retrieval_file(self):
        ret = dict()
        with open(self.test_question_retrieval_path) as f:
        #with open(self.train_question_retrieval_path) as f:
            for line in f:
                elems = line.strip().split('\t')
                _id = elems[0]
                dic_facts = dict()
                for index, fact in enumerate(elems[1:]):
                    dic_facts[index] = fact
                ret[_id] = dic_facts
        return ret 

    def get_test_qa(self):
        ret = dict()
        with open(self.test_set_path) as f:
            for line in f:
                elems = line.strip().split('\t')
                _id = elems[0]
                ret[_id] = line.strip()
        return ret 

    def get_stop_word(self):
        ret = dict()
        with open(config.wizard_stop_word_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                word = line.strip()
                ret[word] = 1
        return ret 

    def __get_sample_numbers(self):
        print('Getting total samples numbers...')
        print(self.train_set_path)
        if os.path.exists(self.train_set_path):
            with open(self.train_set_path) as f:
                for line in f:
                    self.train_sample_num += 1
        if os.path.exists(self.valid_set_path):
            with open(self.valid_set_path) as f:
                for line in f:
                    self.valid_sample_num += 1
        if os.path.exists(self.test_set_path):
            with open(self.test_set_path) as f:
                for line in f:
                    self.test_sample_num += 1

        print('-' * 100)
        print('train sample number: ', self.train_sample_num)
        print('valid sample number: ', self.valid_sample_num)
        print('test sample number: ', self.test_sample_num)
        print('-' * 100)
                
    def generate_sequence(self, text, max_len):
        if self.use_sentence_piece == True:
            if text == self.tokenizer.SYM_PAD:
                encoded = [self.tokenizer.SYM_PAD] * max_len
            else:
                encoded = self.tokenizer.encode(text)[:max_len]
                encoded = [self.tokenizer.SYM_PAD] * (max_len - len(encoded)) + encoded
        else:
            # use ordinary vocabulary
            if text == self.tokenizer.SYM_PAD:
                pad_id = self.src_token_ids.get(config.SYM_PAD)
                encoded = [pad_id] * max_len
            else:
                encoded = []
                unk_id = self.src_token_ids.get(config.SYM_UNK)
                pad_id = self.src_token_ids.get(config.SYM_PAD)
                for token in text.strip().split(' '):
                    token_id = self.src_token_ids.get(token, unk_id)
                    encoded.append(token_id)
                encoded = encoded[:max_len]
                encoded = [pad_id] * (max_len - len(encoded)) + encoded
        return encoded

    def _deal_qa_sample(self, elems):
        ques = elems[1].strip()
        ans = elems[2].strip()
        encoded_ques = self.generate_sequence(ques, self.args.src_seq_length)
        return encoded_ques

    def _deal_positive_negative_samples(self, _type, elems, sample_number):
        encoded_samples = []
        for text in elems:
            encoded_text = self.generate_sequence(text, self.args.sample_seq_length)
            encoded_samples.append(encoded_text)
        encoded_samples = encoded_samples[:sample_number]
        if len(encoded_samples) < sample_number:
            if _type == 'pos':
                while len(encoded_samples) < sample_number:
                    index = random.randint(0, len(encoded_samples) - 1)
                    encoded_samples.append(encoded_samples[index])
            if _type == 'neg':
                pad_seq = self.generate_sequence(self.tokenizer.SYM_PAD, self.args.sample_seq_length)
                while len(encoded_samples) < sample_number:
                    encoded_samples.append(pad_seq)

        assert (len(encoded_samples) == sample_number)
        return encoded_samples

    def _deal_query_retrieval_samples(self, elems, sample_number):
        masks = []
        encoded_samples = []
        for text in elems:
            encoded_text = self.generate_sequence(text, self.args.src_seq_length)
            encoded_samples.append(encoded_text)
        encoded_samples = encoded_samples[:sample_number]
        for i in range(len(encoded_samples)):
            masks.append([1])

        if len(encoded_samples) < sample_number:
            pad_seq = self.generate_sequence(self.tokenizer.SYM_PAD, self.args.src_seq_length)
            pad_id = self.src_token_ids.get(config.SYM_PAD)
            if self.use_sentence_piece == True:
                pad_id = self.tokenizer.SYM_PAD
            while len(encoded_samples) < sample_number:
                encoded_samples.append(pad_seq)
                masks.append([pad_id])

        assert (len(encoded_samples) == sample_number)
        assert (len(masks) == sample_number)
        return encoded_samples, masks

    def _deal_query_retrieval_oracle_tags(self, elems, sample_number):
        tags = []
        for tag in elems:
            tag = int(tag)
            tags.append(tag)
        tags = tags[:sample_number]

        if len(tags) < sample_number:
            while len(tags) < sample_number:
                tags.append(0)

        assert (len(tags) == sample_number)
        return tags

    def fit_batch_size(self, seq_list):
        cur_count = len(seq_list[0])
        while cur_count < self.args.batch_size:
            index = random.randint(0, len(seq_list[0]) - 1)
            for i in range(len(seq_list)):
                seq_list[i].append(seq_list[i][index])
            cur_count = len(seq_list[0])
        return seq_list

    def feed_dict(self, file_type, q_tokens, pos_tokens, neg_tokens, qr_tokens, qr_masks, ids):
        if len(q_tokens) < self.args.batch_size:
            q_tokens, pos_tokens, neg_tokens, qr_tokens, qr_masks, ids = self.fit_batch_size(\
                [q_tokens, pos_tokens, neg_tokens, qr_tokens, qr_masks, ids])

        q_tokens = np.asarray(q_tokens)
        pos_tokens = np.asarray(pos_tokens)
        neg_tokens = np.asarray(neg_tokens)
        qr_tokens = np.asarray(qr_tokens)
        qr_masks = np.asarray(qr_masks)

        y = np.zeros((len(q_tokens), self.args.positive_number + self.args.negative_number))
        for i in range(self.args.positive_number):
            y[:, i] = 1
        
        ret_list = []
        ret_list.append(q_tokens)

        # output positive sample for training and validing 
        if file_type != 'test':
        #if file_type != 'train':
            for i in range(self.args.query_retrieval_number):
                ret_list.append(qr_masks[:, i, :])
            for i in range(self.args.query_retrieval_number):
                ret_list.append(qr_tokens[:, i, :])
            for i in range(self.args.positive_number):
                ret_list.append(pos_tokens[:,i,:])
            for i in range(self.args.negative_number):
                ret_list.append(neg_tokens[:, i, :])
            return ret_list, y
        else:
            ret_list.append(qr_masks)
            ret_list.append(qr_tokens)
            ret_list.append(neg_tokens)
            return ids, ret_list, y

    # This is a data generator, which is suitable for large-scale data set
    def data_generator(self, file_type):
        '''
        :param file_type: This is supposed to be: train, valid, or test
        '''
        #print('This is in data generator...')
        assert file_type == 'train' or file_type == 'valid' or file_type == 'test'
    
        if file_type == 'train':
            qa_path = self.train_set_path
            positive_path = self.train_positive_path 
            negative_path = self.train_negative_path 
            question_retrieval_path = self.train_question_retrieval_path 
        elif file_type == 'valid':
            qa_path = self.valid_set_path
            positive_path = self.valid_positive_path 
            negative_path = self.valid_negative_path 
            question_retrieval_path = self.valid_question_retrieval_path 
        elif file_type == 'test':
            qa_path = self.test_set_path
            positive_path = self.test_positive_path 
            # there is no negative for test set
            negative_path = self.test_question_retrieval_path 
            question_retrieval_path = self.test_question_retrieval_path 

        def _read_files():
            while True:
                f_qa = open(qa_path)
                f_pos = open(positive_path)
                f_neg = open(negative_path)
                f_qr = open(question_retrieval_path)
                print(qa_path)
                print(positive_path)
                print(negative_path)
                print(question_retrieval_path)

                ids = []
                q_tokens = []
                pos_tokens = []
                neg_tokens = []
                qr_tokens = []
                qr_masks = []
                for index, (qa_line, pos_line, neg_line, qr_line ) in enumerate(zip(f_qa, f_pos, f_neg, f_qr)):
                    qa_elems = qa_line.strip().split('\t')
                    pos_elems = pos_line.strip().split('\t')
                    neg_elems = neg_line.strip().split('\t')
                    qr_elems = qr_line.strip().split('\t')

                    qa_id = qa_elems[0]
                    pos_id = pos_elems[0]
                    neg_id = neg_elems[0]
                    qr_id = qr_elems[0]
                    assert(qa_id == pos_id)
                    assert(qa_id == neg_id)
                    assert(qa_id == qr_id)

                    encoded_ques = self._deal_qa_sample(qa_elems)
                    encoded_poses = self._deal_positive_negative_samples('pos', pos_elems[1:], self.args.positive_number)
                    encoded_qrs, qr_mask = self._deal_query_retrieval_samples(qr_elems[1:], self.args.query_retrieval_number)

                    if file_type == 'test':
                    #if file_type == 'train':
                        encoded_negs = self._deal_positive_negative_samples('neg', neg_elems[1:], self.args.test_top_k)
                    else:
                        encoded_negs = self._deal_positive_negative_samples('neg', neg_elems[1:], self.args.negative_number)
                    
                    ids.append(qa_id)
                    q_tokens.append(encoded_ques)
                    pos_tokens.append(encoded_poses)
                    neg_tokens.append(encoded_negs)
                    qr_tokens.append(encoded_qrs)
                    qr_masks.append(qr_mask)

                    if (len(q_tokens) % self.args.batch_size == 0):
                        ids_out = deepcopy(ids)
                        q_out = deepcopy(q_tokens)
                        pos_out = deepcopy(pos_tokens)
                        neg_out = deepcopy(neg_tokens)
                        qr_out = deepcopy(qr_tokens)
                        qr_mask_out = deepcopy(qr_masks)
                        ids = []
                        q_tokens = []
                        pos_tokens = []
                        neg_tokens = []
                        qr_tokens = []
                        qr_masks = []
                        yield self.feed_dict(file_type, q_out, pos_out, neg_out, qr_out, qr_mask_out, ids_out)

                if (len(q_tokens) != 0):
                    ids_out = deepcopy(ids)
                    q_out = deepcopy(q_tokens)
                    pos_out = deepcopy(pos_tokens)
                    neg_out = deepcopy(neg_tokens)
                    qr_out = deepcopy(qr_tokens)
                    qr_mask_out = deepcopy(qr_masks)
                    ids = []
                    q_tokens = []
                    pos_tokens = []
                    neg_tokens = []
                    qr_tokens = []
                    qr_masks = []
                    yield self.feed_dict(file_type, q_out, pos_out, neg_out, qr_out, qr_mask_out, ids_out)

        return _read_files()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    auto_pointer_dssm_arguments(parser)
    args = parser.parse_args()
    #args.batch_size = 1

    ds = DataSet(args)
    #data_generator = ds.data_generator('test')
    data_generator = ds.data_generator('train')
    ds.get_test_question_retrieval_file()

    for cur_text_batch in data_generator:
        #print(cur_text_batch)
        pass


