import numpy as np
import tensorflow as tf

import argparse

import os



def parse_args(check=True):
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--dataset', type=str, default='/data/HataFeng/songci/sc.train')
    parser.add_argument('--train_dir', type=str, default='/output/sc.ckpt')
    parser.add_argument('--learning_rate', type=float, default=0.4)
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_step', type=int, default=35)
    parser.add_argument('--Optimizer', type=str, default='SGD') # adam  SGD
    parser.add_argument('--environment', type=str, default='local') #tinymain

    FLAGS, unparsed = parser.parse_known_args()

    print("FLAGS:", FLAGS)
    return FLAGS, unparsed

FLAGS, unparsed = parse_args()

if FLAGS.environment == "tinymain":
    TRAIN_DATA = FLAGS.dataset
    CHECKPOINT_PATH = FLAGS.train_dir
    LEARNING_RATE = FLAGS.learning_rate
    HIDDEN_SIZE = FLAGS.hidden_size
    NUM_LAYERS = FLAGS.num_layers
    TRAIN_BATCH_SIZE = FLAGS.batch_size
    TRAIN_NUM_STEP = FLAGS.num_step

    VOCAB_SIZE = 5000

    EVAL_BATCH_SIZE = 1
    EVAL_NUM_STEP = 1
    NUM_EPOCH = 15
    LSTM_KEEP_PROB = 0.9
    EMBEDDING_KEEP_PROB = 0.9
    MAX_GRAD_NORM = 5
    SHARE_EMB_AND_SOFTMAX = True
else: # local
    TRAIN_DATA = "G:/test_data/songci/output/sc.train"
    TEST_DATA = "G:/test_data/songci/output/sc.test"

    CHECKPOINT_PATH = "G:/test_data/songci/output"

    #LEARNING_RATE = 0.4  #0.1 40.3 ; #0.5 38.5

    #HIDDEN_SIZE = 500
    #NUM_LAYERS = 10

    LEARNING_RATE = FLAGS.learning_rate
    HIDDEN_SIZE = FLAGS.hidden_size
    NUM_LAYERS = FLAGS.num_layers

    TRAIN_BATCH_SIZE = 20
    TRAIN_NUM_STEP = 35

    VOCAB_SIZE = 5000

    EVAL_BATCH_SIZE = 1
    EVAL_NUM_STEP = 1
    NUM_EPOCH = 15
    LSTM_KEEP_PROB = 0.9
    EMBEDDING_KEEP_PROB = 0.9
    MAX_GRAD_NORM = 5
    SHARE_EMB_AND_SOFTMAX = True


class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义使用LSTM结构为循环体结构且使用dropout的深层循环神经网络
        # NUM_LAYERS 网络深度层数
        # HIDDEN_SIZE 神经元数量
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0
        lstm_cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
                output_keep_prob=dropout_keep_prob)
            for _ in range(NUM_LAYERS)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        self.initial_state = cell.zero_state(batch_size, tf.float32)

        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        # 将输入单词转化为词向量
        #skip-gram模型
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # 对输入数据进行dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)

        # 收集LSTM不同时刻的输出
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

        # 分类输出
        self.predictions = tf.nn.softmax(logits, name='predictions')

        # 真实分布与预测分布的交叉熵
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]),
            logits=logits)
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        if not is_training: return

        trainable_variables = tf.trainable_variables()

        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)

        # 梯度优化
        print("FLAGS.Optimizer:", FLAGS.Optimizer)
        if FLAGS.Optimizer == "adam":
            print("use adma Optimizer  learning_rate:", LEARNING_RATE)
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)

        # 训练步骤
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=self.global_step)


def run_epoch(session, model, batches, train_op, output_log, saver):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    for x, y in batches:
        global_step, cost, state, _ = session.run(
            [model.global_step, model.cost, model.final_state, train_op],
            {model.input_data: x, model.targets: y,
             model.initial_state: state})

        total_costs += cost
        iters += model.num_steps

        if output_log and global_step % 100 == 0:
            print("After %d steps, perpelxity is %.3f" % (global_step, np.exp(total_costs / iters)))

        if global_step % 200 == 0:
            print("save ckpt  global_step:", global_step)
            saver.save(session, os.path.join(CHECKPOINT_PATH, "sc.ckpt"), global_step=global_step)

    return global_step, np.exp(total_costs / iters)

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


def main():
    #    train_batches = make_batches(read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        train_batches = make_batches(read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

        saver = tf.train.Saver(max_to_keep=10)

        try:
            checkpoint_path = tf.train.latest_checkpoint(CHECKPOINT_PATH)
            saver.restore(session, checkpoint_path)

        except Exception:
            print("no check point found....")

        for i in range(NUM_EPOCH):
            step, train_pplx = run_epoch(session, train_model, train_batches,
                                         train_model.train_op, True, saver)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_pplx))

if __name__ == "__main__":
    if FLAGS.environment == "tinymine":
        print('current working dir [{0}]'.format(os.getcwd()))
        w_d = os.path.dirname(os.path.abspath(__file__))
        print('change wording dir to [{0}]'.format(w_d))
        os.chdir(w_d)

    main()