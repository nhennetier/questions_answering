import sys
import time
import json
from os.path import join as pjoin
from shutil import copyfile

import numpy as np
from math import floor
from nltk.tokenize import sent_tokenize, word_tokenize

from utils import Vocab

import tensorflow as tf
from tensorflow.contrib.rnn import static_rnn, static_bidirectional_rnn, GRUCell, LSTMCell, MultiRNNCell


class Config(object):
    '''Hyperparameters'''
    embed_size = 300
    pretrained_embed = True
    hidden_size = 100
    num_hidden_layers = 1
    max_epochs = 500
    early_stopping = 50
    lr = 1e-3
    dropout = 0.8
  

class QA_Model():
    '''Preprocessing of raw data from the SQuAD.'''
    def context_parser(self, old_context):
        new_context = []
        old_context = sent_tokenize(old_context)
        for sent in old_context:
            new_context += sent.replace(':', ';').split(";")
        return new_context
    
    def preprocess(self, data):
        '''First preprocessing: sentence tokenization and word tokenization with nltk.'''
        dataset = []
        words = []
        for article in data['data']:
            paragraphs = article['paragraphs']
            for par in paragraphs:
                qas = [{'question': word_tokenize(question['question']),
                        'answer': question['answers'][0]['text']}
                       for question in par['qas']
                       if len(word_tokenize(question['answers'][0]['text']))==1]
                context = self.context_parser(par['context'])
                context = [word_tokenize(sent) for sent in context]
                words += sum([question['question'] + [question['answer']] for question in qas], [])
                words += sum(context, [])
                dataset.append({'context': context, 'qas': qas})
        words = [word.lower() for word in words]
        dataset = [par for par in dataset if len(par['qas'])>0]
        return words, dataset

    def encode_dataset(self, dataset):
        '''Encode words (tokens to their ids) and sort sentences by decreasing length.'''
        encoded_dataset = [{'context': sorted([[self.vocab.encode(word.lower()) 
                                                for word in sent]
                                               for sent in par['context']],
                                              key=len,
                                              reverse=True),
                            'questions': sorted([[self.vocab.encode(word.lower()) 
                                                  for word in question['question']]
                                                 for question in par['qas']],
                                                 key=len,
                                                 reverse=True),
                            'answers': [self.vocab.encode(question['answer'].lower())
                                        for question in par['qas']]}
                           for par in dataset]
        return encoded_dataset

    def padding_dataset(self, dataset):
        '''Addition of 0s at the end of sentences to fit in tf placeholders.
        NB: these 0s will not be considered by the model'''
        encoded_dataset = [{'context': np.array([sent + [0 for i in range(self.config.len_sent_context - len(sent))]
                                                 for sent in par['context']] \
                                                + [[0 for i in range(self.config.len_sent_context)]
                                                   for j in range(self.config.len_context - len(par['context']))]),
                            'questions': np.array([sent + [0 for i in range(self.config.len_sent_questions - len(sent))]
                                                   for sent in par['questions']] \
                                                  + [[0 for i in range(self.config.len_sent_questions)]
                                                     for j in range(self.config.len_questions - len(par['questions']))]),
                            'answers': np.array(par['answers'])
                           } for par in dataset]
        return encoded_dataset

    def variable_len_sent_context(self, encoded_data):
        '''Array of sentences' lengths from the different contexts.'''
        return [[len(sent) for sent in par['context']] \
                + [0 for _ in range(self.config.len_context - len(par['context']))] \
                for par in encoded_data]

    def variable_len_sent_questions(self, encoded_data):
        '''Array of sentences' lengths from the different questions.'''
        return [[len(sent) for sent in par['questions']] \
                + [0 for _ in range(self.config.len_questions - len(par['questions']))] \
                for par in encoded_data]

    def max_len_sent(self, encoded_data, type_data):
        '''Maximum length of a sentence within the contexts.'''
        return max([max([len(sent) for sent in par[type_data]]) for par in encoded_data])
        
    def max_len(self, encoded_data, type_data):
        return max([len(par[type_data]) for par in encoded_data])
        
    def load_data(self):
        #Loading of datasets from files
        with open('./data/train.json') as data_file:
            train = json.load(data_file)
        with open('./data/dev.json') as data_file:
            dev = json.load(data_file)

        words_train, dataset_train = self.preprocess(train)
        words_dev, dataset_dev = self.preprocess(dev)

        #Construction of vocabulary from both train and dev datasets
        self.vocab = Vocab()
        self.vocab.construct(words_train + words_dev)

        #Mapping from tokens to vector representations in CommonCrawl glove. 
        if self.config.pretrained_embed:
            glove_vecs = {}
            with open('/Users/nicolashennetier/pretrained_embeddings/glove.840B.300d.txt') as glove_file:
                for line in glove_file:
                    vec = line.split()
                    if len(vec) == 301 and vec[0] in self.vocab.word_to_index.keys():
                        glove_vecs[vec[0]] = [float(x) for x in vec[1:]]

            #Creation of embedding matrix
            self.embeddings = np.zeros((len(self.vocab), 300))
            for ind, word in self.vocab.index_to_word.items():
                try:
                    self.embeddings[ind,:] = glove_vecs[word]
                except:
                    self.embeddings[ind,:] = np.zeros(300)
            self.embeddings = self.embeddings.astype(np.float32)

        self.encoded_train = self.encode_dataset(dataset_train)
        self.encoded_valid = self.encode_dataset(dataset_dev)

        #Constants of the datasets
        self.config.len_sent_context = max(self.max_len_sent(self.encoded_train, type_data='context'),
                                           self.max_len_sent(self.encoded_valid, type_data='context'))
        self.config.len_sent_questions = max(self.max_len_sent(self.encoded_train, type_data='questions'),
                                             self.max_len_sent(self.encoded_valid, type_data='questions'))
        self.config.len_context = max(self.max_len(self.encoded_train, type_data='context'),
                                       self.max_len(self.encoded_valid, type_data='context'))
        self.config.len_questions = max(self.max_len(self.encoded_train, type_data='questions'),
                                         self.max_len(self.encoded_valid, type_data='questions'))

        self.config.lengths_sent_context = [self.variable_len_sent_context(self.encoded_train),
                                            self.variable_len_sent_context(self.encoded_valid)]
        self.config.lengths_sent_questions = [self.variable_len_sent_questions(self.encoded_train),
                                              self.variable_len_sent_questions(self.encoded_valid)]
        
        self.encoded_train = self.padding_dataset(self.encoded_train)
        self.encoded_valid = self.padding_dataset(self.encoded_valid)
        
        #Dev/Test split
        n = floor(len(self.encoded_valid)/2)
        self.encoded_test = self.encoded_valid[n:]
        self.config.lengths_sent_context.append(self.config.lengths_sent_context[1][n:])
        self.config.lengths_sent_questions.append(self.config.lengths_sent_questions[1][n:])
        self.encoded_valid = self.encoded_valid[:n]
        self.config.lengths_sent_context[1] = self.config.lengths_sent_context[1][:n]
        self.config.lengths_sent_questions[1] = self.config.lengths_sent_questions[1][:n]


class RNN_QAModel(QA_Model):
    '''Questions Answering RNN model with LSTM cells.'''
    def __init__(self, config):
        self.config = config
        self.load_data()

        #Construction of the computational graph
        self.add_placeholders()
        self.inputs = self.add_embedding()
        #Outputs for both context and questions
        self.rnn_outputs = (self.add_model(self.inputs[0], 'Context'),
                            self.add_model(self.inputs[1], 'Questions'))
        self.outputs = self.add_projection(self.rnn_outputs)
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
            context = [tf.squeeze(x, [1]) for x in tf.split(embed_context, self.config.len_sent_context, 1)]
            questions = [tf.squeeze(x, [1]) for x in tf.split(embed_questions, self.config.len_sent_questions, 1)]
        
        return context, questions

    def add_model(self, inputs, type_layer):
        '''Construction of the RNN model with LSTM cells.
        Arguments:
            - type_layer: should be 'Context' or 'Questions'
        '''
        #Dropout for inputs
        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in inputs]

        with tf.variable_scope('Hidden-Layers', initializer=tf.contrib.layers.xavier_initializer()) as scope:
            if type_layer == 'Questions':
                scope.reuse_variables()

            initializer = tf.random_uniform_initializer(-1,1) 
            cell = LSTMCell(self.config.hidden_size, initializer=initializer)

            if type_layer == 'Context':
                initial_state = cell.zero_state(self.config.len_context, tf.float32)
                outputs, hidden_state = static_rnn(cell,
                                                   inputs,
                                                   initial_state=initial_state,
                                                   sequence_length=self.context_len_placeholder)
            elif type_layer == "Questions":
                initial_state = cell.zero_state(self.config.len_questions, tf.float32)
                outputs, hidden_state = static_rnn(cell,
                                                   inputs,
                                                   initial_state=initial_state,
                                                   sequence_length=self.questions_len_placeholder)

        return outputs[-1]

    def add_projection(self, rnn_outputs):
        '''Compute the probabilities of answers for each token from vocabulary.'''
        h_context = rnn_outputs[0]
        h_questions = rnn_outputs[1]

        h_questions = tf.split(h_questions, self.config.len_questions, 0)
        
        with tf.variable_scope('Projection-Layer', initializer=tf.contrib.layers.xavier_initializer()) as scope:
            Uc = tf.get_variable('Uc',
              [self.config.hidden_size, self.config.hidden_size])
            Uq = tf.get_variable('Uq',
              [self.config.hidden_size, self.config.hidden_size])
            b = tf.get_variable('b',
              [self.config.hidden_size])
            Wo = tf.get_variable('Wo',
              [self.config.hidden_size, self.config.embed_size])
            bo = tf.get_variable('bo',
              [self.config.embed_size])

            outputs = []
            for question in h_questions:
                output_gates = tf.sigmoid(tf.matmul(h_context, Uc) + tf.matmul(question, Uq) + b)
                gated_context = tf.multiply(h_context, output_gates)
                outputs.append(tf.reduce_sum(gated_context, axis=0))
            
            outputs = tf.stack(outputs, axis=0)
            outputs = tf.matmul(self.output_placeholder, outputs)
            outputs = tf.matmul(outputs, Wo) + bo

        return outputs

    def add_loss_op(self, output):
        '''Computation of cross-entropy error.'''
        embed_answers = tf.nn.embedding_lookup(self.embeddings,
                                               self.answers_placeholder)
        loss = tf.reduce_sum(tf.nn.l2_loss(output - embed_answers))
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.RMSPropOptimizer(self.config.lr,
                                              decay=0.99,
                                              momentum=0.5,
                                              centered=True)
        train_op = optimizer.minimize(loss)

        return train_op
    
    def run_epoch(self, session, data, data_type=0, train_op=None, verbose=1):
        '''Runs the model for an entire dataset.
        Arguments:
            - data_type: 0 for training, 1 for dev and 2 for test
            - train_op: None for no backprop, self.train_step for backprop
        Returns:
            - Cross-entropy loss
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
            
            lengths_sent_context = self.config.lengths_sent_context[data_type][step]
            lengths_sent_questions = self.config.lengths_sent_questions[data_type][step]
            
            feed = {self.context_placeholder: context,
                    self.questions_placeholder: questions,
                    self.answers_placeholder: answers,
                    self.context_len_placeholder: lengths_sent_context,
                    self.questions_len_placeholder: lengths_sent_questions,
                    self.output_placeholder: output_padding,
                    self.dropout_placeholder: dp}

            #Runs the model with forward pass (and backprop if train_op)
            loss, _, predictions = session.run(
                [self.calculate_loss, train_op, self.predictions], feed_dict=feed)
            total_loss.append(loss)

            #Predictions and accuracy
            for i in range(len(predictions)):
                pred = predictions[i]
                ans = answers[i]
                pred = np.argmin(np.sum((pred - self.embeddings)**2, 1))
                pos_preds += (pred == ans)
                num_answers += 1

            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(
                    step, total_steps, np.sum(total_loss) / num_answers))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
        return np.sum(total_loss) / num_answers, pos_preds / num_answers

def weights_saver(saver, session, is_best):
    print("Saving weights")
    if not os.path.exists("./weights/current"):
        os.makedirs("./weights/current")
    if not os.path.exists("./weights/best"):
        os.makedirs("./weights/best")

    saver.save(session, "./weights/current/model.ckpt")

    if is_best:
        for f in os.listdir("./weights/current"):
            copyfile(pjoin("./weights/current", f), pjoin("./weights/best", f))
        

def test_RNNQA():
    config = Config()
    with tf.variable_scope('RNNLM') as scope:
        model = RNN_QAModel(config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        best_val_ce = float('inf')
        best_val_epoch = 0
    
        session.run(init)
        for epoch in range(config.max_epochs):
            if epoch == 0:
                try:    
                    saver.restore(session, "./weights/current/model.ckpt")
                except:
                    pass

            print('Epoch {}'.format(epoch))
            start = time.time()

            train_ce, train_accuracy = model.run_epoch(
                session, model.encoded_train,
                data_type=0, train_op=model.train_step)
            print('Training cross-entropy: {}'.format(train_ce))
            print('Training accuracy: {}'.format(train_accuracy))

            valid_ce, valid_accuracy = model.run_epoch(
                session, model.encoded_valid,
                data_type=1)
            print('Validation cross-entropy: {}'.format(valid_ce))
            print('Validation accuracy: {}'.format(valid_accuracy))
            
            #Run additional epochs while cross-entropy improving on dev dataset
            if valid_ce < best_val_ce:
                best_val_ce = valid_ce
                best_val_epoch = epoch
                weights_saver(saver, session, True)
            else:
                weights_saver(saver, session, False)

            if epoch - best_val_epoch > config.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))
        
        #Final run on test dataset
        saver.restore(session, 'ptb_rnnlm.weights')
        test_ce, test_accuracy = model.run_epoch(
            session, model.encoded_test,
            data_type=2)
        print('=-=' * 5)
        print('Test cross-entropy: {}'.format(test_ce))
        print('Test accuracy: {}'.format(test_accuracy))
        print('=-=' * 5)

if __name__ == "__main__":
    test_RNNQA()