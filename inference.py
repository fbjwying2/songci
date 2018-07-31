import numpy as np
import tensorflow as tf

import codecs


import argparse

import os



def parse_args(check=True):
    parser = argparse.ArgumentParser()
    # train

    parser.add_argument('--title', type=str, default='海枯石烂')
    parser.add_argument('--vocab', type=str, default='/data/HataFeng/songci/sc.vocab')
    parser.add_argument('--train_dir', type=str, default='/output/sc.ckpt')
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_step', type=int, default=35)

    parser.add_argument('--environment', type=str, default='local') #tinymain

    FLAGS, unparsed = parser.parse_known_args()

    #print("FLAGS:", FLAGS)
    return FLAGS, unparsed

FLAGS, unparsed = parse_args()


TITLE = FLAGS.title
VOCAB = FLAGS.vocab
CHECKPOINT_PATH = FLAGS.train_dir

HIDDEN_SIZE = FLAGS.hidden_size
NUM_LAYERS = FLAGS.num_layers
TRAIN_BATCH_SIZE = FLAGS.batch_size
TRAIN_NUM_STEP = FLAGS.num_step

VOCAB_SIZE = 5000

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 1
LSTM_KEEP_PROB = 0.9
EMBEDDING_KEEP_PROB = 0.9
MAX_GRAD_NORM = 5
SHARE_EMB_AND_SOFTMAX = True



id_to_word = []
word_to_id = {}
line_no = 1
with codecs.open(VOCAB, "rb", "utf-8") as f:
    for line in f:
        id_to_word.append(line.strip())
        word_to_id[line.strip()] = line_no
        line_no += 1


class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0
        lstm_cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
                output_keep_prob=dropout_keep_prob)
            for _ in range(NUM_LAYERS)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        self.initial_state = cell.zero_state(batch_size, tf.float32)

        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training:
            inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)

        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        if SHARE_EMB_AND_SOFTMAX:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])

        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias

        self.predictions = tf.nn.softmax(logits, name='predictions')

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.targets, [-1]), logits=logits)
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state


def read_data(file_path):
    with open(file_path, "r") as fin:
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]
    return id_list


def make_batches(id_list, batch_size, num_step):
    num_batches = (len(id_list) - 1) // (batch_size * num_step)

    #
    data = np.array(id_list[: num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches * num_step])

    data_batches = np.split(data, num_batches, axis=1)

    #
    label = np.array(id_list[1: num_batches * batch_size * num_step + 1])
    label = np.reshape(label, [batch_size, num_batches * num_step])

    label_batches = np.split(label, num_batches, axis=1)

    return list(zip(data_batches, label_batches))


def run_epoch_test(session, model, state, batches, title, no_op, output_log, step):
    iters = 0
    state_ = session.run(model.initial_state)

    for x, y in batches:
        predictions, state_, _ = session.run(
            [model.predictions, model.final_state, no_op],
            {model.input_data: x, model.targets: y,
             model.initial_state: state_})

        iters += model.num_steps
        step += 1

    sentence = []
    input_words = predictions[0].argsort()[-3:]

    for input_word in input_words:
        print("last input_word:", id_to_word[input_word])
        _sentence = ""

        for i in range(512):
            predictions, state_, _ = session.run(
                [model.predictions, model.final_state, no_op],
                {model.input_data: [[input_word]], model.targets: y,
                 model.initial_state: state_})

            input_word = predictions[0].argsort()[-1]
            word = id_to_word[input_word]

            _sentence = _sentence + word

        print("%s%s" % (title, _sentence))
        sentence.append(_sentence)

    return sentence


def main():
    test_input = []
    title = TITLE
    print("title: ", title)
    for word in title:
        test_input.append(word_to_id[word])

    print(test_input)

    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    test_batches = make_batches(test_input, EVAL_BATCH_SIZE, EVAL_NUM_STEP)
    checkpoint_path = tf.train.latest_checkpoint(CHECKPOINT_PATH)
    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session, checkpoint_path)

        sentence = run_epoch_test(session, eval_model, train_model.final_state, test_batches, title, tf.no_op(), False,
                                  0)

    # print(sentence)


if __name__ == "__main__":
    if FLAGS.environment == "tinymine":
        print('current working dir [{0}]'.format(os.getcwd()))
        w_d = os.path.dirname(os.path.abspath(__file__))
        print('change wording dir to [{0}]'.format(w_d))
        os.chdir(w_d)

    main()