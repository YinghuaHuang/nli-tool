from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import sys

class ModelConfig():
    def __init__(self):
        self.__parser = argparse.ArgumentParser()
        self.arg = None
        self.__addArguments()
        self.__readConfig()

    # add config parameter
    def __addArguments(self):
        # training hyper-parameters
        self.__parser.add_argument('--num_epochs',
                                   '-ep',
                                   default=300,
                                   type=int,
                                   help='Number of epochs')
        self.__parser.add_argument('--batch_size',
                                   '-bs',
                                   default=32,
                                   type=int,
                                   help='Batch size')
        self.__parser.add_argument('--dropout_keep_prob',
                                   '-dkp',
                                   default=0.5,
                                   type=float,
                                   help='Dropout keep probability')
        self.__parser.add_argument('--learning_rate',
                                   '-lr',
                                   default=0.0004,
                                   type=float,
                                   help='Learning rate')
        self.__parser.add_argument('--l2',
                                   '-l2',
                                   default=0.0,
                                   type=float,
                                   help='L2 normalization constant')
        self.__parser.add_argument('--seq_length',
                                   '-sl',
                                   default=100,
                                   type=int,
                                   help='Max length of input sentence')
        self.__parser.add_argument('--early_stop_step',
                                   '-ess',
                                   default=50000,
                                   type=int,
                                   help='Early stop condition')
        # embeddings hyper-parameters
        self.__parser.add_argument('--embedding_size',
                                   '-es',
                                   default=300,
                                   type=int,
                                   help='Word embedding size')

        # layers hyper-parameters
        self.__parser.add_argument('--hidden_size',
                                   '-hs',
                                   default=300,
                                   type=int,
                                   help='Hidden layer size')

        # report hyper-parameters
        self.__parser.add_argument('--eval_batch',
                                   '-eb',
                                   default=1000,
                                   type=int,
                                   help='Number of batches between performance reports')

        # IO path
        ## embeddings
        self.__parser.add_argument('--vocab_path',
                                   '-vp',
                                   default='./data/word_sequence/vocab.txt',
                                   type=str,
                                   help='Vocabulary file')
        self.__parser.add_argument('--embedding_path',
                                   '-embp',
                                   default='./data/glove/filtered_glove_840B_300d.txt',
                                   type=str,
                                   help='Pre-trained word embeddings path')
        ## dataset
        self.__parser.add_argument('--train_premise_file',
                                   '-tpf',
                                   default='./data/word_sequence/premise_snli_1.0_train.txt',
                                   type=str,
                                   help='Train premise file')
        self.__parser.add_argument('--train_hypothesis_file',
                                   '-thf',
                                   default='./data/word_sequence/hypothesis_snli_1.0_train.txt',
                                   type=str,
                                   help='Train hypothesis file')
        self.__parser.add_argument('--train_label_file',
                                   '-tlf',
                                   default='./data/word_sequence/label_snli_1.0_train.txt',
                                   type=str,
                                   help='Train label file')
        self.__parser.add_argument('--dev_premise_file',
                                   '-dpf',
                                   default='./data/word_sequence/premise_snli_1.0_dev.txt',
                                   type=str,
                                   help='Valid premise file')
        self.__parser.add_argument('--dev_hypothesis_file',
                                   '-dhf',
                                   default='./data/word_sequence/hypothesis_snli_1.0_dev.txt',
                                   type=str,
                                   help='Valid hypothesis file')
        self.__parser.add_argument('--dev_label_file',
                                   '-dlf',
                                   default='./data/word_sequence/label_snli_1.0_dev.txt',
                                   type=str,
                                   help='Valid label file')

        ## model
        self.__parser.add_argument('--model',
                                   '-model',
                                   default='esim',
                                   type=str,
                                   help='Model type')

        ## reports
        self.__parser.add_argument('--save_path',
                                   '-sp',
                                   default='./ckpt/checkpoint',
                                   type=str,
                                   help='Directory to save checkpoint')
        self.__parser.add_argument('--best_path',
                                   '-bp',
                                   default='./ckpt/bestval',
                                   type=str,
                                   help='Directory to save the best model')
        self.__parser.add_argument('--log_path',
                                   '-lp',
                                   type=str,
                                   help='Log path')
        self.__parser.add_argument('--tfboard_path',
                                   '-tbp',
                                   default='./tensorboard',
                                   help='Directory to save the TensorBoard files')
        ## config
        self.__parser.add_argument('--config_path',
                                   '-cp',
                                   default='./config/esim_config.yaml',
                                   type=str,
                                   help='Config path')

    # read config information from config file
    def __readConfig(self):
        arg = self.__parser.parse_args()
        with open(arg.config_path) as conf:
            config_dict = yaml.load(conf)
            for key, value in config_dict.items():
                sys.argv.append('--' + key)
                sys.argv.append(str(value))
        self.arg = self.__parser.parse_args()

    # print config information
    def print_info(self):
        arg_dict = vars(self.arg)
        print('-' * 20 + ' Config Information ' + '-' * 20)
        for key, value in arg_dict.items():
            print('%-12s : %s' % (key, value))
        print('-' * 60)