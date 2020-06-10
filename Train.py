from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model import ESIM, Decomposable
from Utils import *
from data_helpers import *
import sys
import os
from datetime import datetime
import Config

def train_step(x_premise, x_hypothesis, x_premise_len, x_hypothesis_len, targets, sess):
    feed_dict = {
        model.premise: x_premise,
        model.hypothesis: x_hypothesis,
        model.premise_len: x_premise_len,
        model.hypothesis_len: x_hypothesis_len,
        model.target: targets,
        model.dropout_keep_prob: arg.dropout_keep_prob,
    }
    _, step, loss, accuracy, predicted_prob = sess.run(
        [model.train, model.global_step, model.mean_loss, model.accuracy, model.probs],
        feed_dict)

    time_str = datetime.now().isoformat()
    if step % 100 == 0:
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))


def check_step(dataset, sess, shuffle=False):
    num_test = 0
    num_correct = 0.0
    batches = batch_iter(dataset, arg.batch_size, 1, arg.seq_length, shuffle=shuffle)
    for batch in batches:
        x_premise, x_hypothesis, x_premise_len, x_hypothesis_len, targets = batch
        feed_dict = {
            model.premise: x_premise,
            model.hypothesis: x_hypothesis,
            model.premise_len: x_premise_len,
            model.hypothesis_len: x_hypothesis_len,
            model.target: targets,
            model.dropout_keep_prob: 1.0,
        }
        batch_accuracy, predicted_prob = sess.run([model.accuracy, model.probs], feed_dict)
        num_test += len(predicted_prob)
        num_correct += len(predicted_prob) * batch_accuracy

    # calculate Accuracy
    acc = num_correct / num_test
    print('num_test_samples: {}  accuracy: {}'.format(num_test, acc))
    return acc

# training
def train():
    # load data
    train_dataset = loadDataset(arg.train_premise_file, arg.train_hypothesis_file, arg.train_label_file, vocab_dict, arg.seq_length)
    print('train_dataset: {}'.format(len(train_dataset)))
    dev_dataset = loadDataset(arg.dev_premise_file, arg.dev_hypothesis_file, arg.dev_label_file,vocab_dict, arg.seq_length)
    print('dev_dataset: {}'.format(len(dev_dataset)))

    # model saving
    saver = tf.train.Saver(max_to_keep=5)
    save_file_dir, save_file_name = os.path.split(arg.save_path)
    if not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)
    # init
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    # training
    best_acc_val = 0.0
    batches = batch_iter(train_dataset, arg.batch_size, arg.num_epochs, arg.seq_length, shuffle=True)
    for batch in batches:
        x_premise, x_hypothesis, x_premise_len, x_hypothesis_len, targets = batch
        train_step(x_premise, x_hypothesis, x_premise_len, x_hypothesis_len, targets, sess)
        current_step = tf.train.global_step(sess, model.global_step)
        if current_step % arg.eval_batch == 0:
            acc_val = check_step(dev_dataset, sess, shuffle=True)
            # save model
            saver.save(sess = sess, save_path = arg.save_path, global_step = current_step)
            # save best model
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                saver.save(sess = sess, save_path = arg.best_path)

if __name__ == '__main__':
    # read config
    config = Config.ModelConfig()
    arg = config.arg

    vocab_dict = loadVocab(arg.vocab_path)
    embeddings = get_embeddings(vocab_dict, arg.embedding_size ,arg.embedding_path)
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    arg.log_path = 'config/log/log.{}'.format(dt)
    log = open(arg.log_path, 'w')
    print_log('CMD : python3 {0}'.format(' '.join(sys.argv)), file = log)
    print_log('Training with following options :', file = log)
    print_args(arg, log)

    ModelCollection = {"esim":ESIM, "decomposable":Decomposable}
    Model = ModelCollection.get(arg.model)
    model = Model(arg.seq_length, arg.hidden_size, arg.batch_size, arg.learning_rate, embeddings)
    train()
    log.close()