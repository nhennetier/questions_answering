import sys
import time
import os
import numpy as np
from os.path import join as pjoin
from shutil import copyfile

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell, DropoutWrapper

from data_processing import QA_Model

class Config(object):
    '''Hyperparameters'''
    embed_size = 300
    pretrained_embed = False
    hidden_size = 100
    nb_hidden_layers = 3
    nb_selection_layers = 3
    nb_proj_layers = 2
    type_cell = "LSTM"
    hidden_bidirectional = True
    selection_bidirectional = True
    output_type = "hs"
    max_epochs = 500
    early_stopping = 50
    lr = 1e-3
    dropout = 0.8


class SQuADModel2(QA_Model):
    '''Questions Answering RNN model with LSTM cells.'''
    def __init__(self, config):
        self.config = config
        self.load_data()

        #Construction of the computational graph
        self.add_placeholders()
        self.inputs = self.add_embedding()
        #Outputs for both context and questions
        self.encoding_outputs = (self.add_encoder(self.inputs[0], 'Context'),
                                 self.add_encoder(self.inputs[1], 'Questions'))
        self.selection_outputs = self.add_selection(self.encoding_outputs)
        self.outputs = self.add_projection(self.selection_outputs)
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
            name='Len-Questions')
        self.output_placeholder = tf.placeholder(tf.float32,
            shape=[None, self.config.len_questions],
            name='Output')
        
    def add_embedding(self):
        '''Replace tokens (ids) by their vector representations'''
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

    def create_cells(self, nb_layers, reuse=False):
        initializer = tf.random_uniform_initializer(-1,1)

        cell_bw = None            
        if nb_layers > 1:
            if self.config.type_cell == "LSTM":
                cell_fw = MultiRNNCell([LSTMCell(self.config.hidden_size, initializer=initializer, reuse=reuse)
                                        for _ in range(nb_layers)])
                if self.config.hidden_bidirectional:
                    cell_bw = MultiRNNCell([LSTMCell(self.config.hidden_size, initializer=initializer, reuse=reuse)
                                            for _ in range(nb_layers)])
            elif self.config.type_cell == "GRU":
                cell_fw = MultiRNNCell([GRUCell(self.config.hidden_size, kernel_initializer=initializer, reuse=reuse)
                                        for _ in range(nb_layers)])
                if self.config.hidden_bidirectional:
                    cell_bw = MultiRNNCell([GRUCell(self.config.hidden_size, kernel_initializer=initializer, reuse=reuse)
                                            for _ in range(nb_layers)])
            else:
                raise NotImplementedError
        else:
            if self.config.type_cell == "LSTM":
                cell_fw = LSTMCell(self.config.hidden_size, initializer=initializer, reuse=reuse)
                if self.config.hidden_bidirectional:
                    cell_bw = LSTMCell(self.config.hidden_size, initializer=initializer, reuse=reuse)
            elif self.config.type_cell == "GRU":
                cell_fw = GRUCell(self.config.hidden_size, kernel_initializer=initializer, reuse=reuse)
                if self.config.hidden_bidirectional:
                    cell_bw = GRUCell(self.config.hidden_size, kernel_initializer=initializer, reuse=reuse)
            else:
                raise NotImplementedError

        return cell_fw, cell_bw

    def add_encoder(self, inputs, type_layer):
        '''Construction of the RNN model with LSTM cells.
        Arguments:
            - type_layer: should be 'Context' or 'Questions'
        '''

        with tf.variable_scope('Encoding-Layer', initializer=tf.contrib.layers.xavier_initializer()) as scope:
            reuse = type_layer == "Questions"
            initializer = tf.random_uniform_initializer(-1,1)

            cell_fw, cell_bw = self.create_cells(self.config.nb_hidden_layers, reuse)

            if type_layer == "Context":
                batch_size = self.config.len_context
                sequence_length = self.context_len_placeholder
            elif type_layer == "Questions":
                batch_size = self.config.len_questions
                sequence_length = self.questions_len_placeholder

            cell_fw = DropoutWrapper(cell_fw, output_keep_prob=self.dropout_placeholder)
            initial_state_fw = cell_fw.zero_state(batch_size, tf.float32)

            if self.config.hidden_bidirectional:
                cell_bw = DropoutWrapper(cell_bw, output_keep_prob=self.dropout_placeholder)
                initial_state_bw = cell_bw.zero_state(batch_size, tf.float32)
                outputs, hidden_states = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                         cell_bw,
                                                                         inputs,
                                                                         initial_state_fw=initial_state_fw,
                                                                         initial_state_bw=initial_state_bw,
                                                                         sequence_length=sequence_length)
            else:
                outputs, hidden_states = tf.nn.dynamic_rnn(cell_fw,
                                                           inputs,
                                                           initial_state=initial_state_fw,
                                                           sequence_length=sequence_length)

            if self.config.output_type == "output":
                encoding_outputs = tf.transpose(outputs, [1, 0, 2])
                encoding_outputs = tf.gather(encoding_outputs, self.config.len_questions - 1)

            elif self.config.output_type == "hs":
                if self.config.hidden_bidirectional:
                    encoding_outputs = (hidden_states[0], hidden_states[1])
                    if self.config.nb_hidden_layers > 1:
                        encoding_outputs = (encoding_outputs[0][-1], encoding_outputs[1][-1])
                    if self.config.type_cell == "LSTM":
                        encoding_outputs = (encoding_outputs[0].h, encoding_outputs[1].h)
                else:
                    encoding_outputs = hidden_states
                    if self.config.nb_hidden_layers > 1:
                        encoding_outputs = encoding_outputs[-1]
                    if self.config.type_cell == "LSTM":
                        encoding_outputs = encoding_outputs.h

        return encoding_outputs

    def add_selection(self, encoding_outputs):
        '''Compute the probabilities of answers for each token from vocabulary.'''
        h_context = encoding_outputs[0]
        h_questions = encoding_outputs[1]

        if self.config.hidden_bidirectional:
            h_context = tf.concat([h_context[0], h_context[1]], 1)
            h_questions = tf.concat([h_questions[0], h_questions[1]], 1)

        h_questions = tf.split(h_questions, self.config.len_questions, 0)
        
        if self.config.hidden_bidirectional:
            size = 2*self.config.hidden_size
        else:
            size = self.config.hidden_size

        with tf.variable_scope('Selection-Layer', initializer=tf.contrib.layers.xavier_initializer()) as scope:
            batches = []
            for question in h_questions:
                question = tf.tile(question, [self.config.len_context, 1])
                h_context_question = tf.concat([h_context, question], 1)
                h_context_question = tf.expand_dims(h_context_question, 0)
                batches.append(h_context_question)

            inputs = tf.concat(batches, 0)

            cell_fw, cell_bw = self.create_cells(self.config.nb_selection_layers)

            initial_state_fw = cell_fw.zero_state(self.config.len_questions, tf.float32)

            if self.config.selection_bidirectional:
                initial_state_bw = cell_bw.zero_state(self.config.len_questions, tf.float32)
                outputs, hidden_states = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                         cell_bw,
                                                                         inputs,
                                                                         initial_state_fw=initial_state_fw,
                                                                         initial_state_bw=initial_state_bw,
                                                                         sequence_length=self.nb_sent_context_placeholder)
            else:
                outputs, hidden_states = tf.nn.dynamic_rnn(cell_fw,
                                                           inputs,
                                                           initial_state=initial_state_fw,
                                                           sequence_length=self.nb_sent_context_placeholder)

            if self.config.selection_bidirectional:
                outputs = [hidden_states[0], hidden_states[1]]
                if self.config.nb_selection_layers > 1:
                    outputs = [outputs[0][-1], outputs[1][-1]]
                if self.config.type_cell == "LSTM":
                    outputs = [outputs[0].h, outputs[1].h]
                outputs = tf.concat(outputs, 1)

            else:
                outputs = hidden_states
                if self.config.nb_selection_layers > 1:
                    outputs = outputs[-1]
                if self.config.type_cell == "LSTM":
                    outputs = outputs.h

            selection_outputs = tf.matmul(self.output_placeholder, outputs)

        return selection_outputs

    def add_projection(self, selection_outputs):
        if self.config.selection_bidirectional:
            size = 2*self.config.hidden_size
        else:
            size = self.config.hidden_size

        outputs = selection_outputs
        with tf.variable_scope('Projection-Layer', initializer=tf.contrib.layers.xavier_initializer()) as scope:
            for i in range(self.config.nb_proj_layers - 1):
                W = tf.get_variable('W{}'.format(i), [size, size])
                b = tf.get_variable('b{}'.format(i), [size])
                outputs = tf.tanh(tf.matmul(outputs, W) + b)

            W = tf.get_variable('W{}'.format(self.config.nb_proj_layers - 1),
                [size, self.config.embed_size])
            b = tf.get_variable('b{}'.format(self.config.nb_proj_layers - 1),
                [self.config.embed_size])
            outputs = tf.matmul(outputs, W) + b

        return outputs

    def add_loss_op(self, output):
        '''Computation of mean squared error.'''
        embed_answers = tf.nn.embedding_lookup(self.embeddings,
                                               self.answers_placeholder)
        loss = tf.nn.l2_loss(output - embed_answers)
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.RMSPropOptimizer(self.config.lr,
                                              decay=0.99,
                                              momentum=0.,
                                              centered=True)
        train_op = optimizer.minimize(loss)

        return train_op
    
    def run_epoch(self, session, data, data_type=0, train_op=None, verbose=1, compute_pred=False):
        '''Runs the model for an entire dataset.
        Arguments:
            - data_type: 0 for training, 1 for dev and 2 for test
            - train_op: None for no backprop, self.train_step for backprop
        Returns:
            - Mean squared error
            - Accuracy
        '''
        config = self.config
        dp = config.dropout
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

            output_padding = np.concatenate([np.identity(len(answers)),
                                             np.zeros((len(answers), config.len_questions - len(answers)))],
                                            axis=1)
            
            lengths_sent_context = config.lengths_sent_context[data_type][step]
            lengths_sent_questions = config.lengths_sent_questions[data_type][step]
            nb_sent_context = [config.lenghts_context[data_type][step] for _ in range(config.len_questions)]
            
            feed = {self.context_placeholder: context,
                    self.questions_placeholder: questions,
                    self.answers_placeholder: answers,
                    self.context_len_placeholder: lengths_sent_context,
                    self.questions_len_placeholder: lengths_sent_questions,
                    self.nb_sent_context_placeholder: nb_sent_context,
                    self.output_placeholder: output_padding,
                    self.dropout_placeholder: dp}

            #Runs the model with forward pass (and backprop if train_op)
            loss, _ = session.run(
                [self.calculate_loss, train_op], feed_dict=feed)
            total_loss.append(loss)

            if compute_pred:
                predictions = session.run(self.predictions, feed_dict=feed)
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

            num_answers += len(answers)

            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(
                    step, total_steps, np.sum(total_loss) / num_answers))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')

        return np.sum(total_loss) / num_answers, pos_preds / num_answers

def weights_saver(saver, session, model_name, is_best):
    print("Saving weights")
    if not os.path.exists("./weights/current"):
        os.makedirs("./weights/current")
    if not os.path.exists("./weights/best"):
        os.makedirs("./weights/best")

    saver.save(session, "./weights/current/" + model_name + ".ckpt")

    if is_best:
        for f in os.listdir("./weights/current"):
            copyfile(pjoin("./weights/current", f), pjoin("./weights/best", f))
        

def train_SQuAD2(model_name):
    config = Config()
    with tf.variable_scope('SQuAD') as scope:
        model = SQuADModel2(config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        best_val_mse = float('inf')
        best_val_epoch = 0
    
        session.run(init)
        for epoch in range(config.max_epochs):
            if epoch == 0:
                try:    
                    saver.restore(session, "./weights/current/" + model_name + ".ckpt")
                except:
                    pass

            print('Epoch {}'.format(epoch))
            start = time.time()

            train_mse, _ = model.run_epoch(session,
                                           model.encoded_train,
                                           data_type=0,
                                           train_op=model.train_step)
            print('Training mse: {}'.format(train_mse))

            valid_mse, _ = model.run_epoch(session,
                                           model.encoded_valid,
                                           data_type=1)
            print('Validation mse: {}'.format(valid_mse))
            
            #Run additional epochs while mse improving on dev dataset
            if valid_mse < best_val_mse:
                best_val_mse = valid_mse
                best_val_epoch = epoch
                weights_saver(saver, session, model_name, True)
            else:
                weights_saver(saver, session, model_name, False)

            if epoch - best_val_epoch > config.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))

def test_SQuAD2(model_name):
    config = Config()
    with tf.variable_scope('SQuAD') as scope:
        model = SQuADModel2(config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)
        
        #Run on test dataset
        try:
            saver.restore(session, "./weights/best/" + model_name + ".ckpt")
        except:
            raise ValueError("Model has no saved weights to restore.")

        test_mse, test_accuracy = model.run_epoch(session,
                                                  model.encoded_test,
                                                  data_type=2,
                                                  compute_pred=True)
        print('=-=' * 5)
        print('Test mse: {}'.format(test_mse))
        print('Test accuracy: {}'.format(test_accuracy))
        print('=-=' * 5)