import tensorflow as tf
import numpy as np
from DataReader import DataReader


class RNN:
    def __init__(self, vocab_size, embedding_size, lstm_size, batch_size):
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._lstm_size = lstm_size
        self._batch_size = batch_size

        self._data = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, MAX_DOC_LENGTH])
        self._labels = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, ])
        self._sentence_lengths = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, ])
        self._final_token = tf.compat.v1.placeholder(tf.int32, shape=[batch_size, ])

    def embedding_layer(self, indices):
        pretrained_vector = []
        pretrained_vector.append(np.zeros(self._embedding_size))
        np.random.seed(2021)
        for _ in range(self._vocab_size + 1):
            pretrained_vector.append(np.random.normal(loc=0., scale=1., size=self._embedding_size))

        pretrained_vector = np.array(pretrained_vector)

        self._embedding_matrix = tf.compat.v1.get_variable(
            name='embedding',
            shape=(self._vocab_size + 2, self._embedding_size),
            initializer=tf.constant_initializer(pretrained_vector)
        )
        return tf.nn.embedding_lookup(self._embedding_matrix, indices)

    def LSTM_layer(self, embeddings):
        lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self._lstm_size)
        zero_state = tf.zeros(shape=(self._batch_size, self._lstm_size))
        initial_state = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(zero_state, zero_state)

        lstm_inputs = tf.unstack(tf.transpose(embeddings, perm=[1, 0, 2]))
        lstm_outputs, last_state = tf.compat.v1.nn.static_rnn(
            cell=lstm_cell,
            inputs=lstm_inputs,
            initial_state=initial_state,
            sequence_length=self._sentence_lengths
        )  # a length-500 list of [num_docs, lstm_size]

        lstm_outputs = tf.unstack(tf.transpose(lstm_outputs, perm=[1, 0, 2]), )
        lstm_outputs = tf.concat(lstm_outputs, axis=0)  # [num docs * MAX_SENT_LENGTH, lstm_size]

        # self.mask: [num_docs * MAX_SENT_LENGTH, ]
        mask = tf.sequence_mask(
            lengths=self._sentence_lengths,
            maxlen=MAX_DOC_LENGTH,
            dtype=tf.float32
        )  # [num_docs, MAX_SENTENCE_LENGTH]
        mask = tf.concat(tf.unstack(mask, axis=0), axis=0)
        mask = tf.expand_dims(mask, -1)

        lstm_outputs = mask * lstm_outputs
        lstm_outputs_split = tf.split(lstm_outputs, num_or_size_splits=self._batch_size)
        lstm_outputs_sum = tf.reduce_sum(lstm_outputs_split, axis=1)    # [num_docs, lstm_size]
        lstm_outputs_average = lstm_outputs_sum / tf.expand_dims(
            tf.cast(self._sentence_lengths, tf.float32), -1) # [num_docs, lstm_size]

        return lstm_outputs_average

    def build_graph(self):
        embeddings = self.embedding_layer(self._data)
        lstm_outputs = self.LSTM_layer(embeddings)

        weights = tf.compat.v1.get_variable(
            name='final_layer_weights',
            shape=(self._lstm_size, NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2021)
        )
        biases = tf.compat.v1.get_variable(
            name='final_layer_biases',
            shape=(NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2021)
        )
        logits = tf.matmul(lstm_outputs, weights) + biases

        labels_one_hot = tf.one_hot(
            indices=self._labels,
            depth=NUM_CLASSES,
            dtype=tf.float32
        )

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_one_hot,
            logits=logits
        )
        loss = tf.reduce_mean(loss)

        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axis=1)
        predicted_labels = tf.squeeze(predicted_labels)
        return predicted_labels, loss

    def trainer(self, loss, learning_rate):
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_op


def train_and_evaluate_RNN():
    with open('w2v/vocab-raw.txt') as f:
        vocab_size = len(f.read().splitlines())

    tf.disable_eager_execution()
    tf.random.set_seed(2021)
    rnn = RNN(
        vocab_size=vocab_size,
        embedding_size=300,
        lstm_size=50,
        batch_size=50
    )
    predicted_labels, loss = rnn.build_graph()
    train_op = rnn.trainer(loss=loss, learning_rate=0.01)

    with tf.compat.v1.Session as sess:
        train_data_reader = DataReader(
            data_path=train_data_path,
            batch_size=50
        )
        test_data_reader = DataReader(
            data_path=test_data_path,
            batch_size=50
        )
        step = 0
        MAX_STEP = 1000

        sess.run(tf.compat.v1.global_variables_initializer())
        while step < MAX_STEP:
            next_train_batch = train_data_reader.next_batch
            train_data, train_labels, train_sentence_lengths, train_final_tokens = next_train_batch
            plabels_eval, loss_eval, _ = sess.run(
                [predicted_labels, loss, train_op],
                feed_dict={
                    rnn._data: train_data,
                    rnn._labels: train_labels,
                    rnn._sentence_lengths: train_sentence_lengths,
                    rnn._final_token: train_final_tokens
                }
            )
            step += 1
            if step % 20 == 0:
                print('loss:', loss_eval)

            if train_data_reader._batch_id == 0:
                num_true_preds = 0
                while True:
                    next_test_batch = test_data_reader.next_batch
                    test_data, test_labels, test_sentence_lengths, test_final_tokens = next_test_batch

                    test_plabels_eval = sess.run(
                        predicted_labels,
                        feed_dict={
                            rnn._data: test_data,
                            rnn._labels: test_labels,
                            rnn._sequence_lengths: test_sentence_lengths,
                            rnn._final_token: test_final_tokens
                        }
                    )
                    matches = np.equal(test_plabels_eval, test_labels)
                    num_true_preds += np.sum(matches.astype(float))

                    if test_data_reader._batch_id == 0:
                        break
                print('Epoch:', train_data_reader._num_epoch)
                print('Accuracy on test data:', num_true_preds * 100. / len(test_data_reader._data))


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()

    MAX_DOC_LENGTH = 500
    NUM_CLASSES = 20

    vocab_path = 'w2v/vocab-raw.txt'
    train_data_path = 'w2v/20news-train-encoded.txt'
    test_data_path = 'w2v/20news-test-encoded.txt'

    train_and_evaluate_RNN()