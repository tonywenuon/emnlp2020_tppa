# !/use/bin/env python

import os
import configparser
from commonly_used_code.helper_fn import *

parser = configparser.SafeConfigParser()
config_file_path = '../configuration/config.ini'
#config_file_path = 'configuration/config.ini'
parser.read(config_file_path)

#SYM_PAD = '<PAD>'
#SYM_UNK = '<UNK>'
#SYM_START = '<START>'
#SYM_END = '<END>'

SYM_PAD = '<pad>'
SYM_UNK = '<unk>'
SYM_START = '<start>'
SYM_END = '<end>'
SYM_CLS = '[CLS]'
SYM_SEP = '[SEP]'

# for TED model
# reserve <START> and <END> for source
src_reserved_pos = 2
# reserve <START> or <END> for target
tar_reserved_pos = 1
NO_FACT = 'no_fact'
NO_CONTEXT = 'no_context'

#SYM_PAD = '<PAD>'
#SYM_UNK = '<UNK>'

# original data set path
wizard_data_path = parser.get('FilePath', 'wizard_data_path')

train_path = parser.get('FilePath', 'train_data_path')
valid_path = parser.get('FilePath', 'valid_data_path')
test_path = parser.get('FilePath', 'test_data_path')

# generate symblic question answer and facts
src_global_token_path = parser.get('SymblicQAF', 'src_global_token_dict')
tar_global_token_path = parser.get('SymblicQAF', 'tar_global_token_dict')

pro_qa_data_path = parser.get('SymblicQAF', 'pro_qa_data')
negative_data_path = parser.get('SymblicQAF', 'negative_data')
question_retrieval_data_path = parser.get('SymblicQAF', 'question_retrieval_data')
response_retrieval_data_path = parser.get('SymblicQAF', 'response_retrieval_data')

# wizard train valid test data path
wizard_train_path = os.path.join(wizard_data_path, train_path)
wizard_valid_path = os.path.join(wizard_data_path, valid_path)
wizard_test_path = os.path.join(wizard_data_path, test_path)
makedirs(wizard_train_path)
makedirs(wizard_valid_path)
makedirs(wizard_test_path)

# used for wizard file path
wizard_global_token_path = os.path.join(wizard_train_path, src_global_token_path)

wizard_train_qa_path = os.path.join(wizard_train_path, pro_qa_data_path)
wizard_train_negative_data_path = os.path.join(wizard_train_path, negative_data_path)
wizard_train_question_retrieval_data_path = os.path.join(wizard_train_path, question_retrieval_data_path)
wizard_train_response_retrieval_data_path = os.path.join(wizard_train_path, response_retrieval_data_path)

wizard_valid_qa_path = os.path.join(wizard_valid_path, pro_qa_data_path)
wizard_valid_negative_data_path = os.path.join(wizard_valid_path, negative_data_path)
wizard_valid_question_retrieval_data_path = os.path.join(wizard_valid_path, question_retrieval_data_path)
wizard_valid_response_retrieval_data_path = os.path.join(wizard_valid_path, response_retrieval_data_path)

wizard_test_qa_path = os.path.join(wizard_test_path, pro_qa_data_path)
wizard_test_negative_data_path = os.path.join(wizard_test_path, negative_data_path)
wizard_test_question_retrieval_data_path = os.path.join(wizard_test_path, question_retrieval_data_path)
wizard_test_response_retrieval_data_path = os.path.join(wizard_test_path, response_retrieval_data_path)

