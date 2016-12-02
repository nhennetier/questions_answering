import getpass
import sys
import time
import json

import numpy as np
from math import floor
from nltk.tokenize import sent_tokenize, word_tokenize

from utils import Vocab

import tensorflow as tf


class Config(object):
    embed_size = 300
    pretrained_embed = True
    hidden_size = 100
    num_hidden_layers = 1
    max_epochs = 500
    early_stopping = 20
    lr = 1e-3
    dropout = 0.8
  

class QA_Model():
    def preprocess(self, data):
        dataset = []
        words = []
        for article in data['data']:
            paragraphs = article['paragraphs']
            for par in paragraphs:
                qas = [{'question': word_tokenize(question['question']),
                        'answer': question['answers'][0]['text']}
                       for question in par['qas']
                       if len(word_tokenize(question['answers'][0]['text']))==1]
                context = sent_tokenize(par['context'])
                context = [word_tokenize(sent) for sent in context]
                words += sum([question['question'] + [question['answer']] for question in qas], [])
                words += sum(context, [])
                dataset.append({'context': context, 'qas': qas})
        words = [word.lower() for word in words]
        dataset = [par for par in dataset if len(par['qas'])>0]
        return words, dataset

    def encode_dataset(self, dataset):
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
        encoded_dataset = [{'context': np.array([sent + [0 for _ in range(self.config.len_sent_context - len(sent))]
                                                 for sent in par['context']]),
                            'questions': np.array([sent + [0 for _ in range(self.config.len_sent_questions - len(sent))]
                                                   for sent in par['questions']]),
                            'answers': np.array(par['answers'])
                           } for par in dataset]
        return encoded_dataset

    def variable_len_sent_context(self, encoded_data):
        return [[len(sent) for sent in par['context']] for par in encoded_data]

    def variable_len_sent_questions(self, encoded_data):
        return [[len(sent) for sent in par['questions']] for par in encoded_data]

    def max_len_sent_context(self, encoded_data):
        return max([max([len(sent) for sent in par['context']]) for par in encoded_data])
        
    def max_len_sent_questions(self, encoded_data):
        return max([max([len(sent) for sent in par['questions']]) for par in encoded_data])
        
    def load_data(self, debug=False):
        with open('./data/train.json') as data_file:
            train = json.load(data_file)
        with open('./data/dev.json') as data_file:
            dev = json.load(data_file)

        words_train, dataset_train = self.preprocess(train)
        words_dev, dataset_dev = self.preprocess(dev)

        self.vocab = Vocab()
        self.vocab.construct(words_train + words_dev)

        if self.config.pretrained_embed:
            glove_vecs = {}
            with open('./data/glove.840B.300d.txt') as glove_file:
                for line in glove_file:
                    vec = line.split()
                    if len(vec) == 301 and vec[0] in self.vocab.word_to_index.keys():
                        glove_vecs[vec[0]] = [float(x) for x in vec[1:]]

            self.embeddings = np.zeros((len(self.vocab), 300))
            for ind, word in self.vocab.index_to_word.items():
                try:
                    self.embeddings[ind,:] = glove_vecs[word]
                except:
                    self.embeddings[ind,:] = np.zeros(300)
            self.embeddings = self.embeddings.astype(np.float32)

        self.encoded_train = self.encode_dataset(dataset_train)
        self.encoded_valid = self.encode_dataset(dataset_dev)

        self.config.len_sent_context = max(self.max_len_sent_context(self.encoded_train),
                                           self.max_len_sent_context(self.encoded_valid))
        self.config.len_sent_questions = max(self.max_len_sent_questions(self.encoded_train),
                                           self.max_len_sent_questions(self.encoded_valid))
        self.config.lengths_sent_context = [self.variable_len_sent_context(self.encoded_train),
                                            self.variable_len_sent_context(self.encoded_valid)]
        self.config.lengths_sent_questions = [self.variable_len_sent_questions(self.encoded_train),
                                              self.variable_len_sent_questions(self.encoded_valid)]
        
        self.encoded_train = self.padding_dataset(self.encoded_train)
        self.encoded_valid = self.padding_dataset(self.encoded_valid)
        
        n = floor(len(self.encoded_valid)/2)
        self.encoded_test = self.encoded_valid[n:]
        self.config.lengths_sent_context.append(self.config.lengths_sent_context[1][n:])
        self.config.lengths_sent_questions.append(self.config.lengths_sent_questions[1][n:])
        self.encoded_valid = self.encoded_valid[:n]
        self.config.lengths_sent_context[1] = self.config.lengths_sent_context[1][:n]
        self.config.lengths_sent_questions[1] = self.config.lengths_sent_questions[1][:n]


class RNN_QAModel(QA_Model):
    def __init__(self, config):
        self.config = config
        self.load_data(debug=False)
        self.add_placeholders()
        self.inputs = self.add_embedding()
        self.rnn_outputs = (self.add_model(self.inputs[0], 'Context'),
                            self.add_model(self.inputs[1], 'Questions'))
        self.outputs = self.add_projection(self.rnn_outputs)
        
        self.predictions = tf.nn.softmax(tf.cast(self.outputs, tf.float64))
        self.calculate_loss = self.add_loss_op(self.outputs)
        self.train_step = self.add_training_op(self.calculate_loss)


    def add_placeholders(self):
        self.context_placeholder = tf.placeholder(tf.int32,
            shape=[None, self.config.len_sent_context],
            name='Context')
        self.questions_placeholder = tf.placeholder(tf.int32,
            shape=[None, self.config.len_sent_questions],
            name='Questions')
        self.answers_placeholder = tf.placeholder(tf.int32,
            shape=[None],
            name='Answers')
        self.output_gates_placeholder1 = tf.placeholder(tf.float32,
            shape=[None, None],
            name='OutputGates1')
        self.output_gates_placeholder2 = tf.placeholder(tf.float32,
            shape=[None, None],
            name='OutputGates2')
        self.output_placeholder = tf.placeholder(tf.float32,
            shape=[None, None],
            name='Output')
        self.context_padding = []
        for i in range(self.config.len_sent_context):
            self.context_padding.append(tf.placeholder(tf.float32,
                shape=[None, None],
                name='ContextPadding%s' % i))
        self.questions_padding = []
        for i in range(self.config.len_sent_questions):
            self.questions_padding.append(tf.placeholder(tf.float32,
                shape=[None, None],
                name='QuestionsPadding%s' % i))
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
        
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
            context = [tf.squeeze(x, [1]) for x in tf.split(1, self.config.len_sent_context, embed_context)]
            questions = [tf.squeeze(x, [1]) for x in tf.split(1, self.config.len_sent_questions, embed_questions)]
          
        return context, questions

    def add_model(self, inputs, type_layer):
        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x, self.dropout_placeholder) for x in inputs]

        hidden_states_old = inputs
        for hidden_step in range(self.config.num_hidden_layers):
            hidden_states_new = []
            with tf.variable_scope('RNN_'+type_layer, initializer=tf.contrib.layers.xavier_initializer()) as scope:
                h = tf.zeros([tf.shape(hidden_states_old[0])[0], self.config.hidden_size])
                c = h
                for tstep, current_sent in enumerate(hidden_states_old):
                    if tstep > 0:
                        scope.reuse_variables()
                    if hidden_step == 0:
                        if type_layer == 'Context':
                            current_sent = tf.matmul(self.context_padding[tstep], current_sent)
                            ht = tf.matmul(self.context_padding[tstep], h)
                            ct = tf.matmul(self.context_padding[tstep], c)
                        elif type_layer == 'Questions':
                            current_sent = tf.matmul(self.questions_padding[tstep], current_sent)
                            ht = tf.matmul(self.questions_padding[tstep], h)
                            ct = tf.matmul(self.questions_padding[tstep], c)
                    else:
                        ht = h
                        ct = c

                    Ui = tf.get_variable(
                        'Ui_l%s' % hidden_step, [self.config.hidden_size, self.config.hidden_size])
                    bi = tf.get_variable(
                        'bi_l%s' % hidden_step, [self.config.hidden_size])
                    pi = tf.get_variable(
                        'pi_l%s' % hidden_step, [self.config.hidden_size])
                    Uf = tf.get_variable(
                        'Uf_l%s' % hidden_step, [self.config.hidden_size, self.config.hidden_size])
                    bf = tf.get_variable(
                        'bf_l%s' % hidden_step, [self.config.hidden_size])
                    pf = tf.get_variable(
                        'pf_l%s' % hidden_step, [self.config.hidden_size])
                    Uo = tf.get_variable(
                        'Uo_l%s' % hidden_step, [self.config.hidden_size, self.config.hidden_size])
                    bo = tf.get_variable(
                        'bo_l%s' % hidden_step, [self.config.hidden_size])
                    po = tf.get_variable(
                        'po_l%s' % hidden_step, [self.config.hidden_size])
                    Uz = tf.get_variable(
                        'Uz_l%s' % hidden_step, [self.config.hidden_size, self.config.hidden_size])
                    bz = tf.get_variable(
                        'bz_l%s' % hidden_step, [self.config.hidden_size])
                    if hidden_step == 0:
                        Wi = tf.get_variable(
                            'Wi_l%s' % hidden_step, [self.config.embed_size, self.config.hidden_size])
                        Wf = tf.get_variable(
                            'Wf_l%s' % hidden_step, [self.config.embed_size, self.config.hidden_size])
                        Wo = tf.get_variable(
                            'Wo_l%s' % hidden_step, [self.config.embed_size, self.config.hidden_size])
                        Wz = tf.get_variable(
                            'Wz_l%s' % hidden_step, [self.config.embed_size, self.config.hidden_size])
                    else:
                        Wi = tf.get_variable(
                            'Wi_l%s' % hidden_step, [self.config.hidden_size, self.config.hidden_size])
                        Wf = tf.get_variable(
                            'Wf_l%s' % hidden_step, [self.config.hidden_size, self.config.hidden_size])
                        Wo = tf.get_variable(
                            'Wo_l%s' % hidden_step, [self.config.hidden_size, self.config.hidden_size])
                        Wz = tf.get_variable(
                            'Wz_l%s' % hidden_step, [self.config.hidden_size, self.config.hidden_size])
                    z = tf.nn.tanh(tf.matmul(ht, Uz) + tf.matmul(current_sent, Wz) + bz)
                    i = tf.nn.sigmoid(tf.matmul(ht, Ui) + tf.matmul(current_sent, Wi) + tf.mul(pi, ct) + bi)
                    f = tf.nn.sigmoid(tf.matmul(ht, Uf) + tf.matmul(current_sent, Wf) + tf.mul(pf, ct) + bf)
                    ct = tf.mul(f,ct) + tf.mul(i,z)
                    o = tf.nn.sigmoid(tf.matmul(ht, Uo) + tf.matmul(current_sent, Wo) + tf.mul(po, ct) + bo)
                    ht = tf.mul(o, tf.nn.tanh(ct))

                    if hidden_step == 0:
                        if type_layer == 'Context':
                            h = tf.concat(0, [ht, h[tf.shape(self.context_padding[tstep])[0]:,:]])
                            c = tf.concat(0, [ct, c[tf.shape(self.context_padding[tstep])[0]:,:]])
                        elif type_layer == 'Questions':
                            h = tf.concat(0, [ht, h[tf.shape(self.questions_padding[tstep])[0]:,:]])
                            c = tf.concat(0, [ct, c[tf.shape(self.questions_padding[tstep])[0]:,:]])
                    else:
                        h = ht
                        c = ct

                    hidden_states_new.append(h)
            hidden_states_old = hidden_states_new

        with tf.variable_scope('OutputDropout'):
            hidden_states_new = [tf.nn.dropout(x, self.dropout_placeholder) for x in hidden_states_new]
        return hidden_states_new[-1]

    def add_projection(self, rnn_outputs):
        h_context = rnn_outputs[0]
        h_questions = rnn_outputs[1]
        
        with tf.variable_scope('Projection-Layer', initializer=tf.contrib.layers.xavier_initializer()) as scope:
            Uc = tf.get_variable('Uc',
              [self.config.hidden_size, self.config.hidden_size])
            bc = tf.get_variable('bc',
              [self.config.hidden_size])
            Uq = tf.get_variable('Uq',
              [self.config.hidden_size, self.config.hidden_size])
            bq = tf.get_variable('bq',
              [self.config.hidden_size])
            W = tf.get_variable('W',
              [self.config.hidden_size, len(self.vocab)])
            og1 = tf.matmul(h_questions, Uq) + bq
            og2 = tf.matmul(h_context, Uc) + bc
            og1 = tf.matmul(self.output_gates_placeholder1, og1)
            og2 = tf.matmul(self.output_gates_placeholder2, og2)
            output_gates = tf.sigmoid(og1 + og2)
            outputs = tf.mul(tf.matmul(self.output_gates_placeholder2, h_context), output_gates)
            outputs = tf.matmul(self.output_placeholder, outputs)
            outputs = tf.matmul(outputs, W)

        return outputs

    def add_loss_op(self, output):
        labels = tf.one_hot(self.answers_placeholder,
            len(self.vocab),
            on_value=1,
            off_value=0,
            axis=-1)
        cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(output, labels))
        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))

        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)

        return train_op
    
    def run_epoch(self, session, data, data_type=0, train_op=None, verbose=1):
        config = self.config
        dp = config.dropout
        if not train_op:
            train_op = tf.no_op()
            dp = 1
        total_steps = len(data)
        total_loss = []
        pos_preds = 0
        num_answers = 0
        for step, paragraph in enumerate(data):
            context = paragraph['context']
            questions = paragraph['questions']
            answers = paragraph['answers']
            output_gates1 = np.concatenate([np.identity(len(questions)) for _ in range(len(context))])
            output_gates2 = [np.zeros((len(questions), len(context))) for _ in range(len(context))]
            for i in range(len(context)):
                output_gates2[i][:,i] = 1
            output_gates2 = np.concatenate(output_gates2)
            output = np.concatenate([np.identity(len(questions)) for _ in range(len(context))], 1)
            lengths_sent_context = self.config.lengths_sent_context[data_type][step]
            lengths_sent_questions = self.config.lengths_sent_questions[data_type][step]
            padding_lengths_context = [len([x for x in lengths_sent_context if x>=t])
                               for t in range(1,self.config.len_sent_context+1)]
            padding_lengths_questions = [len([x for x in lengths_sent_questions if x>=t])
                               for t in range(1,self.config.len_sent_questions+1)]
            context_padding = [np.concatenate([np.identity(padding_lengths_context[i]),
                                               np.zeros((padding_lengths_context[i],
                                                         len(context) - padding_lengths_context[i]))],
                                              1)
                               for i in range(self.config.len_sent_context)]
            questions_padding = [np.concatenate([np.identity(padding_lengths_questions[i]),
                                                 np.zeros((padding_lengths_questions[i],
                                                           len(questions) - padding_lengths_questions[i]))],
                                                1)
                               for i in range(self.config.len_sent_questions)]
            
            feed = {self.context_placeholder: context,
                    self.questions_placeholder: questions,
                    self.answers_placeholder: answers,
                    self.output_gates_placeholder1: output_gates1,
                    self.output_gates_placeholder2: output_gates2,
                    self.output_placeholder: output,
                    self.dropout_placeholder: dp}
            feed.update({k:v for k,v in [(self.context_padding[i], context_padding[i])
                                         for i in range(self.config.len_sent_context)]})
            feed.update({k:v for k,v in [(self.questions_padding[i], questions_padding[i])
                                         for i in range(self.config.len_sent_questions)]})
            loss, _, pred = session.run(
                [self.calculate_loss, train_op, self.predictions], feed_dict=feed)
            total_loss.append(loss)
            pred = np.argmax(pred, 1)
            pos_preds += np.sum(pred==answers)
            num_answers += len(answers)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(
                    step, total_steps, np.sum(total_loss) / num_answers))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        return np.sum(total_loss) / num_answers, pos_preds / num_answers

def test_RNNQA():
    config = Config()
    with tf.variable_scope('RNNLM') as scope:
        model = RNN_QAModel(config)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as session:
        best_val_ce = float('inf')
        best_val_epoch = 0
    
        session.run(init)
        for epoch in range(config.max_epochs):
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
            saver.save(session, './ptb_rnnlm.weights')
            
            if valid_ce < best_val_ce:
                best_val_ce = valid_ce
                best_val_epoch = epoch
            if epoch - best_val_epoch > config.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))
        
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