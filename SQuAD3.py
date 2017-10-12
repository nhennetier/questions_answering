import sys
import time
import os
import numpy as np
from os.path import join as pjoin
from shutil import copyfile

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell, DropoutWrapper
from tensorflow.contrib.seq2seq import LuongAttention, AttentionWrapper
from tensorflow.contrib.layers import xavier_initializer

from data_processing import QA_Model

class Config(object):
    '''Hyperparameters'''
    embed_size = 300
    pretrained_embed = True
    hidden_size = 100
    nb_hidden_layers = 4
    proj_layers = [64, 64]
    type_cell = "LSTM"
    bidirectional = True
    max_epochs = 500
    early_stopping = 50
    lr = 1e-3
    dropout = 0.8


class CustomLSTMCell(LSTMCell):
    def __init__(self, *args, **kwargs):
        kwargs['state_is_tuple'] = False
        returns = super(CustomLSTMCell, self).__init__(*args, **kwargs)
        self._output_size = self._state_size
        return returns
    def __call__(self, inputs, state):
        output, next_state = super(CustomLSTMCell, self).__call__(inputs, state)
        return next_state, next_state


class SQuADModel(QA_Model):
    '''Questions Answering RNN model with LSTM cells.'''
    def __init__(self, config):
        self.config = config
        self.load_data()

        self.create_model()

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter("./summaries/train")
        self.test_writer = tf.summary.FileWriter("./summaries/test")

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def create_model(self):
        #Construction of the computational graph
        self.add_placeholders()
        self.embed_context, self.embed_questions = self.add_embedding()
        #Outputs for both context and questions
        self.encoded_questions, _ = self.add_encoder(self.embed_questions, 'Questions')
        _, self.encoded_sentences_context = self.add_encoder(self.embed_context, 'Sentences-Context')
        _, self.encoded_context = self.add_encoder(self.encoded_sentences_context, 'Context')
        self.outputs = self.add_projection(self.encoded_context)
        #Predictions of the model
        self.predictions = tf.cast(self.outputs, tf.float64)
        #Optimization step
        self.calculate_loss = self.add_loss_op(self.outputs)
        self.train_step = self.add_training_op(self.calculate_loss)


    def add_placeholders(self):
        #Placeholders for data: context, questions and answers
        self.context_placeholder = tf.placeholder(tf.int32,
            shape=[self.config.len_context, self.config.len_sent_context],
            name='Context')
        self.questions_placeholder = tf.placeholder(tf.int32,
            shape=[self.config.len_questions, self.config.len_sent_questions],
            name='Questions')
        self.answers_placeholder = tf.placeholder(tf.int32,
            shape=[None],
            name='Answers')
        
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
        
        self.context_len_placeholder = tf.placeholder(tf.int32,
            shape=[self.config.len_context],
            name='Len-Context')
        self.questions_len_placeholder = tf.placeholder(tf.int32,
            shape=[self.config.len_questions],
            name='Len-Questions')
        self.nb_sent_context_placeholder = tf.placeholder(tf.int32,
            shape=[self.config.len_questions],
            name='Nb-Sent-context')
        self.answers_selection_placeholder = tf.placeholder(tf.float32,
            shape=[None, self.config.len_questions],
            name='Answers-Selection')
        
    def add_embedding(self):
        with tf.device('/cpu:0'):
            if not self.config.pretrained_embed:
                self.embeddings = tf.get_variable('Embedding',
                                                  [len(self.vocab), self.config.embed_size],
                                                  trainable=True)
            embed_context = tf.nn.embedding_lookup(self.embeddings,
                                                   self.context_placeholder)
            embed_questions = tf.nn.embedding_lookup(self.embeddings,
                                                     self.questions_placeholder)
        return embed_context, embed_questions

    def create_single_cell(self, CellWrapper, args):
        cell =  DropoutWrapper(
            CellWrapper(**args),
            output_keep_prob=self.dropout_placeholder
        )
        return cell


    def create_cells(self, nb_layers, hidden_size, bidirectional=False, attention_inputs=None, num_proj=None, reuse=False):
        initializer = xavier_initializer()

        if self.config.type_cell == "LSTM":
            CellWrapper = LSTMCell
            args = {"num_units": hidden_size, "num_proj": num_proj, "initializer": initializer, "reuse": reuse}
        elif self.config.type_cell == "GRU":
            CellWrapper = GRUCell
            args = {"num_units": hidden_size, "num_proj": num_proj, "kernel_initializer": initializer, "reuse": reuse}
        else:
            raise NotImplementedError

        cell_bw = None
        cell_fw = [self.create_single_cell(CellWrapper, args) for _ in range(nb_layers)]
        if bidirectional:
            cell_bw = [self.create_single_cell(CellWrapper, args) for _ in range(nb_layers)]

        if nb_layers > 1:
            cell_fw = MultiRNNCell(cell_fw)
            if bidirectional:
                cell_bw = MultiRNNCell(cell_bw)
        else:
            cell_fw = cell_fw[0]
            if bidirectional:
                cell_bw = cell_bw[0]

        return cell_fw, cell_bw

    def fit_rnn(self, inputs, nb_layers, hidden_size, batch_size, sequence_length, bidirectional=False, reuse=False):
        cell_fw, cell_bw = self.create_cells(nb_layers, hidden_size, bidirectional=bidirectional, reuse=reuse)
        
        initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)
        if bidirectional:
            initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
            hidden_states, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                         cell_bw,
                                                                         inputs,
                                                                         initial_state_fw=initial_state_fw,
                                                                         initial_state_bw=initial_state_bw,
                                                                         sequence_length=sequence_length)
        else:
            hidden_states, final_state = tf.nn.dynamic_rnn(cell_fw,
                                                           inputs,
                                                           initial_state=initial_state_fw,
                                                           sequence_length=sequence_length)

        return hidden_states, final_state

    def reshape_inputs(self, inputs, type_layer):
        if type_layer == "Sentences-Context":
            inputs = tf.tile(inputs, [self.config.len_questions, 1, 1])
        elif type_layer == "Questions":
            inputs = tf.split(inputs, self.config.len_questions, axis=0)
            inputs = [tf.tile(x, [self.config.len_context, 1, 1]) for x in inputs]
            inputs = tf.concat(inputs, axis=0)
        elif type_layer == "Context":
            inputs = tf.split(inputs, self.config.len_questions, axis=0)
            inputs = [tf.expand_dims(x, 1) for x in inputs]
            inputs = tf.concat(inputs, axis=1)
            inputs = tf.transpose(inputs, [1, 0, 2])
        return inputs

    def reshape_placeholders(self, inputs, type_layer):
        if type_layer == "Sentences-Context":
            inputs = tf.tile(inputs, [self.config.len_questions])
        elif type_layer == "Questions":
            inputs = tf.split(inputs, self.config.len_questions, axis=0)
            inputs = [tf.tile(x, [self.config.len_context]) for x in inputs]
            inputs = tf.concat(inputs, axis=0)
        return inputs

    def add_encoder(self, inputs, type_layer):
        '''Construction of the RNN model with LSTM cells.
        Arguments:
            - type_layer: should be 'Context' or 'Questions'
        '''
        if type_layer == "Sentences-Context":
            batch_size = self.config.len_context * self.config.len_questions
            sequence_length = self.reshape_placeholders(self.context_len_placeholder, type_layer)
            inputs = self.reshape_inputs(inputs, type_layer)
            attention_inputs = self.reshape_inputs(self.encoded_questions, "Questions")
        elif type_layer == "Context":
            batch_size = self.config.len_questions
            sequence_length = self.nb_sent_context_placeholder
            inputs = self.reshape_inputs(inputs, type_layer)
            attention_inputs = self.encoded_questions
        elif type_layer == "Questions":
            batch_size = self.config.len_questions
            sequence_length = self.questions_len_placeholder
            attention_inputs = None

        with tf.variable_scope('Encoding-Layer-{}'.format(type_layer)) as scope:
            hidden_states, final_state = self.fit_rnn(inputs,
                                                      self.config.nb_hidden_layers,
                                                      self.config.hidden_size,
                                                      batch_size,
                                                      sequence_length,
                                                      bidirectional=self.config.bidirectional)

            if self.config.bidirectional:
                hidden_states = tf.concat(hidden_states, axis=2)
                if self.config.nb_hidden_layers > 1:
                    final_state = (final_state[0][-1], final_state[1][-1])
                if self.config.type_cell == "LSTM":
                    final_state = (final_state[0].h, final_state[1].h)
                final_state = tf.concat(final_state, 1)
            else:
                if self.config.nb_hidden_layers > 1:
                    final_state = final_state[-1]
                if self.config.type_cell == "LSTM":
                    final_state = final_state.h

            if type_layer == "Sentences-Context" or type_layer == "Context":
                expanded_final_state = tf.expand_dims(final_state, 1)
                expanded_final_state = tf.tile(expanded_final_state, [1, self.config.len_sent_questions, 1])
                alignements = tf.reduce_sum(tf.multiply(expanded_final_state, attention_inputs), axis=2)
                alignements = tf.nn.softmax(alignements, dim=1)
                alignements = tf.expand_dims(alignements, 2)
                context_vector = tf.multiply(attention_inputs, alignements)
                context_vector = tf.reduce_sum(context_vector, axis=1)

                final_state = tf.concat([final_state, context_vector], axis=1)
                size = self.config.hidden_size
                if self.config.bidirectional:
                    size *= 2
                Wc = tf.Variable(tf.random_normal([2*size, size]))
                final_state = tf.nn.tanh(tf.matmul(final_state, Wc))

        print("Encoding layer for {} ready".format(type_layer))
        return hidden_states, final_state

    def add_projection(self, outputs):
        if self.config.bidirectional:
            size = 2*self.config.hidden_size
        else:
            size = self.config.hidden_size

        proj_layers = [size] + self.config.proj_layers + []

        with tf.variable_scope('Projection-Layer') as scope:
            for i in range(len(proj_layers)-1):
                W = tf.get_variable('W{}'.format(i), [proj_layers[i], proj_layers[i+1]])
                b = tf.get_variable('b{}'.format(i), [proj_layers[i+1]])
                outputs = tf.tanh(tf.matmul(outputs, W) + b)

            W = tf.get_variable('W{}'.format(len(proj_layers)),
                [proj_layers[-1], len(self.vocab)])
            b = tf.get_variable('b{}'.format(len(proj_layers)),
                [len(self.vocab)])
            outputs = tf.matmul(outputs, W) + b

        print("Projection layer ready")
        return outputs

    def add_loss_op(self, logits):
        '''Computation of mean squared error.'''
        labels = tf.one_hot(self.answers_placeholder, depth=len(self.vocab))
        logits = tf.matmul(self.answers_selection_placeholder, logits)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.RMSPropOptimizer(self.config.lr,
                                              decay=0.99,
                                              momentum=0.,
                                              centered=True)
        train_op = optimizer.minimize(loss)

        return train_op
    
    def run_epoch(self, data, data_type="train", train_op=None, verbose=1, compute_pred=False):
        '''Runs the model for an entire dataset.
        Arguments:
            - data_type: 0 for training, 1 for dev and 2 for test
            - train_op: None for no backprop, self.train_step for backprop
        Returns:
            - Mean squared error
            - Accuracy
        '''
        dp = self.config.dropout
        if not train_op:
            train_op = tf.no_op()
            dp = 1

        total_steps = len(data)
        total_loss = []
        pos_preds = 0
        num_answers = 0

        #Runs the model for each whole context and its q&a
        for step, paragraph in enumerate(data):
            context = paragraph['context']
            questions = paragraph['questions']
            answers = paragraph['answers']

            lengths_sent_context = self.config.lengths_sent_context[data_type][step]
            lengths_sent_questions = self.config.lengths_sent_questions[data_type][step]
            nb_sent_context = [self.config.lenghts_context[data_type][step] for _ in range(self.config.len_questions)]
            answers_selection = np.concatenate([
                np.identity(answers.shape[0]),
                np.zeros((answers.shape[0], self.config.len_questions - answers.shape[0]))
            ], axis=1)
            
            feed = {self.context_placeholder: context,
                    self.questions_placeholder: questions,
                    self.answers_placeholder: answers,
                    self.context_len_placeholder: lengths_sent_context,
                    self.questions_len_placeholder: lengths_sent_questions,
                    self.nb_sent_context_placeholder: nb_sent_context,
                    self.answers_selection_placeholder: answers_selection,
                    self.dropout_placeholder: dp}

            #Runs the model with forward pass (and backprop if train_op)
            loss, _ = self.session.run(
                [self.calculate_loss, train_op], feed_dict=feed)
            total_loss.append(loss)

            if compute_pred:
                predictions = self.session.run(self.predictions, feed_dict=feed)
                #Predictions and accuracy
                for i in range(len(predictions)):
                    pred = predictions[i]
                    ans = answers[i]
                    pred = np.argmin(np.sum((pred - self.embeddings)**2, 1))
                    pos_preds += (pred == ans)
                    print("Question: " + ' '.join([self.vocab.decode(word) for word in questions[i]]))
                    print("Answer: " + self.vocab.decode(ans))
                    print("Prediction: " + self.vocab.decode(pred))
                    print("\n")

            num_answers += answers.shape[0]

            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(
                    step, total_steps, np.sum(total_loss) / num_answers))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')

        return np.sum(total_loss) / num_answers, pos_preds / num_answers

def weights_saver(model_name, is_best):
    print("Saving weights")
    if not os.path.exists("./weights/current"):
        os.makedirs("./weights/current")
    if not os.path.exists("./weights/best"):
        os.makedirs("./weights/best")

    self.saver.save(self.session, "./weights/current/" + model_name + ".ckpt")

    if is_best:
        for f in os.listdir("./weights/current"):
            copyfile(pjoin("./weights/current", f), pjoin("./weights/best", f))
        

def train_SQuAD(model_name):
    config = Config()
    with tf.variable_scope('SQuAD') as scope:
        model = SQuADModel(config)

    best_val_mse = float('inf')
    best_val_epoch = 0

    try:    
        model.saver.restore(model.session, "./weights/current/" + model_name + ".ckpt")
    except:
        pass

    for epoch in range(config.max_epochs):
        print('Epoch {}'.format(epoch))
        start = time.time()

        train_mse, _ = model.run_epoch(model.encoded_train,
                                       data_type="train",
                                       train_op=model.train_step)
        print('Training mse: {}'.format(train_mse))

        valid_mse, _ = model.run_epoch(model.encoded_valid,
                                       data_type="valid")
        print('Validation mse: {}'.format(valid_mse))
        
        #Run additional epochs while mse improving on dev dataset
        if valid_mse < best_val_mse:
            best_val_mse = valid_mse
            best_val_epoch = epoch
            weights_saver(model_name, True)
        else:
            weights_saver(model_name, False)

        if epoch - best_val_epoch > config.early_stopping:
            break
        print('Total time: {}'.format(time.time() - start))

def test_SQuAD(model_name):
    config = Config()
    with tf.variable_scope('SQuAD') as scope:
        model = SQuADModel(config)
        
    #Run on test dataset
    try:
        model.saver.restore(model.session, "./weights/best/" + model_name + ".ckpt")
    except:
        raise ValueError("Model has no saved weights to restore.")

    test_mse, test_accuracy = model.run_epoch(model.encoded_test,
                                              data_type="test",
                                              compute_pred=True)
    print('=-=' * 5)
    print('Test mse: {}'.format(test_mse))
    print('Test accuracy: {}'.format(test_accuracy))
    print('=-=' * 5)


if __name__ == "__main__":
    model_name = "Model3"
    train_SQuAD(model_name)