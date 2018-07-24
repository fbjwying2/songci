# 代码运行：
>* 准备数据：sc.train  和  sc.vocab    (已上传到tinymind 地址：https://www.tinymind.com/HataFeng/datasets/songci)

>* 运行环境：
>> windows  或  tinymind  支持GPU或者CPU  Tensorflow 1.4  python3.6
>> 本地windows运行：
>>>* Cd code_path
>>>* python train.py --num_layers=2 --Optimizer="SGD" --learning_rate=0.1 --dataset="sc.train" 
>>>* 预测命令
>>>* python inference.py --train_dir="G:/test_data/songci/output/CT500_10_SGD"

# 神经网络的构成
>* 循环神经网络主体结构： 
词嵌入 + 多层LSTM

# 代码框架：
* 数据预处理模块
* 模型模块
* 预测模块

#### 模块说明：
> 数据预处理模块
>> * 1. 整理数据
>> * 2. 编码

> 训练模块
>> * 1.输入层：minibatch    词嵌入 + dropout
>> * 2.处理层：网络结构采用多层LSTM + dropout
>> * 3.输出层：softmax
>> * 4.Lost：交叉熵
>> * 5.优化：SGD或者adam
>> * 6.评价：复杂度。

> 预测结构：
>> * 1.输入层：词嵌入
>> * 2.处理层：网络结构采用多层LSTM + dropout
>> * 3.输出层：softmax
>> * 4.Lost：交叉熵
>> * 5.输出：top3 label 进行预测输出

# 代码说明文档
 * preprefile.ipynb    用空格对单词进行分割（切词后如下： 酒 泉 子 （ 十 之 一 ））
 * prefile.ipynb       按升序生成词表
 * encode.ipynb        将文本转化为单词编码（编码后如下：72 297 50 27 99 137 7 28 2）
 * train.py            训练程序
 * inference.py        预测模块
 
### 训练模块关键代码说明：
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
        
        .....
        
        # 将输入单词转化为词向量
        #skip-gram模型
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # 对输入数据进行dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)
            
            
        ....
        
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
        
        
        # 梯度优化
        print("FLAGS.Optimizer:", FLAGS.Optimizer)
        if FLAGS.Optimizer == "adam":
            print("use adma Optimizer  learning_rate:", LEARNING_RATE)
            optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)

        # 训练步骤
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=self.global_step)
