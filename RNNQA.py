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
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    embed_size = 300
    pretrained_embed = False
    hidden_size = 100
    max_epochs = 16
    early_stopping = 2
    lr = 1e-3
  

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

    def reencode_dataset(self, dataset):
        encoded_dataset = [{'context': np.array([[0 for _ in range(self.config.len_sent_context - len(sent))] + sent
                                                 for sent in par['context']]),
                            'questions': np.array([[0 for _ in range(self.config.len_sent_questions - len(sent))] + sent
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
        """Loads starter word-vectors and train/dev/test data."""
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
        
        self.encoded_train = self.reencode_dataset(self.encoded_train)
        self.encoded_valid = self.reencode_dataset(self.encoded_valid)
        
        n = floor(len(self.encoded_valid)/2)
        self.encoded_test = self.encoded_valid[n:]
        self.encoded_valid = self.encoded_valid[:n]


class RNNContext_Model(QA_Model):
    def add_placeholders(self):
        """Generate placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building
        code and will be fed data during training.  Note that when "None" is in a
        placeholder's shape, it's flexible

        context_placeholder: Input placeholder tensor of shape
                           (None, num_steps), type tf.int32
        answers_placeholder: Labels placeholder tensor of shape
                            (None, num_steps), type tf.float32
        dropout_placeholder: Dropout value placeholder (scalar),
                             type tf.float32
        """
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
    
    def add_embedding(self):
        """Add embedding layer.

        Returns:
          inputs: List of length num_steps, each of whose elements should be
                  a tensor of shape (batch_size, embed_size).
        """
        # The embedding lookup is currently only implemented for the CPU
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

    def add_projection(self, rnn_outputs):
        """Adds a projection layer.

        The projection layer transforms the hidden representation to a distribution
        over the vocabulary.

        Args:
          rnn_outputs: List of length num_steps, each of whose elements should be
                       a tensor of shape (batch_size, embed_size).
        Returns:
          outputs: List of length num_steps, each a tensor of shape
                   (batch_size, len(vocab)
        """
        h_context = rnn_outputs[0]
        h_questions = rnn_outputs[1]
        
        with tf.variable_scope('Projection-Layer', initializer=tf.contrib.layers.xavier_initializer()) as scope:
            Uc = tf.get_variable('Uc',
              [self.config.hidden_size, self.config.hidden_size])
            Uq = tf.get_variable('Uq',
              [self.config.hidden_size, self.config.hidden_size])
            bc = tf.get_variable('bc',
              [self.config.hidden_size])
            W = tf.get_variable('W',
              [self.config.hidden_size, len(self.vocab)])
            og1 = tf.matmul(h_questions, Uq)
            og2 = tf.matmul(h_context, Uc)
            og1 = tf.matmul(self.output_gates_placeholder1, og1)
            og2 = tf.matmul(self.output_gates_placeholder2, og2)
            output_gates = tf.sigmoid(og1 + og2)
            outputs = tf.mul(tf.matmul(self.output_gates_placeholder2, h_context), output_gates)
            outputs = tf.matmul(self.output_placeholder, outputs)
            outputs = tf.matmul(outputs, W)

        return outputs

    def add_loss_op(self, output):
        """Adds loss ops to the computational graph.

        Args:
          output: A tensor of shape (None, self.vocab)
        Returns:
          loss: A 0-d tensor (scalar)
        """
        labels = tf.one_hot(self.answers_placeholder,
            len(self.vocab),
            on_value=1,
            off_value=0,
            axis=-1)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, labels))
        tf.add_to_collection('total_loss', cross_entropy)
        loss = tf.add_n(tf.get_collection('total_loss'))

        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See 

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Hint: Use tf.train.AdamOptimizer for this model.
              Calling optimizer.minimize() will return a train_op object.

        Args:
          loss: Loss tensor, from cross_entropy_loss.
        Returns:
          train_op: The Op for training.
        """
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(self.calculate_loss)

        return train_op
    
    def __init__(self, config):
        self.config = config
        self.load_data(debug=False)
        self.add_placeholders()
        self.inputs = self.add_embedding()
        self.rnn_outputs = self.add_model(self.inputs)
        self.outputs = self.add_projection(self.rnn_outputs)
        
        # We want to check how well we correctly predict the next word
        # We cast o to float64 as there are numerical issues at hand
        # (i.e. sum(output of softmax) = 1.00000298179 and not 1)
        self.predictions = tf.nn.softmax(tf.cast(self.outputs, tf.float64))
        # Reshape the output into len(vocab) sized chunks - the -1 says as many as
        # needed to evenly divide
        self.calculate_loss = self.add_loss_op(self.outputs)
        self.train_step = self.add_training_op(self.calculate_loss)


    def add_model(self, inputs):
        """Creates the RNN LM model.

        In the space provided below, you need to implement the equations for the
        RNN LM model. Note that you may NOT use built in rnn_cell functions from
        tensorflow.

        Args:
          inputs: List of length num_steps, each of whose elements should be
                  a tensor of shape (batch_size, embed_size).
        Returns:
          outputs: List of length num_steps, each of whose elements should be
                   a tensor of shape (batch_size, hidden_size)
        """
        context = inputs[0]
        questions = inputs[1]
        
        with tf.variable_scope('RNN_Context', initializer=tf.contrib.layers.xavier_initializer()) as scope:
            self.initial_state = tf.zeros([tf.shape(context[0])[0], self.config.hidden_size])
            h = self.initial_state
            c = self.initial_state
            for tstep, current_sent in enumerate(context):
                if tstep > 0:
                    scope.reuse_variables()
                Ui = tf.get_variable(
                    'Ui', [self.config.hidden_size, self.config.hidden_size])
                Wi = tf.get_variable(
                    'Wi', [self.config.embed_size, self.config.hidden_size])
                Uf = tf.get_variable(
                    'Uf', [self.config.hidden_size, self.config.hidden_size])
                Wf = tf.get_variable(
                    'Wf', [self.config.embed_size, self.config.hidden_size])
                Uo = tf.get_variable(
                    'Uo', [self.config.hidden_size, self.config.hidden_size])
                Wo = tf.get_variable(
                    'Wo', [self.config.embed_size, self.config.hidden_size])
                Uc = tf.get_variable(
                    'Uc', [self.config.hidden_size, self.config.hidden_size])
                Wc = tf.get_variable(
                    'Wc', [self.config.embed_size, self.config.hidden_size])
                i = tf.nn.sigmoid(tf.matmul(h, Ui) + tf.matmul(current_sent, Wi))
                f = tf.nn.sigmoid(tf.matmul(h, Uf) + tf.matmul(current_sent, Wf))
                o = tf.nn.sigmoid(tf.matmul(h, Uo) + tf.matmul(current_sent, Wo))
                ct = tf.nn.tanh(tf.matmul(h, Uc) + tf.matmul(current_sent, Wc))
                c = tf.mul(f,c) + tf.mul(i,ct)
                h = tf.mul(o, tf.nn.tanh(c))
            self.final_state_context = h

        with tf.variable_scope('RNN_Questions', initializer=tf.contrib.layers.xavier_initializer()) as scope:
            self.initial_state = tf.zeros([tf.shape(questions[0])[0], self.config.hidden_size])
            h = self.initial_state
            c = self.initial_state
            for tstep, current_sent in enumerate(questions):
                if tstep > 0:
                    scope.reuse_variables()
                Ui = tf.get_variable(
                    'Ui', [self.config.hidden_size, self.config.hidden_size])
                Wi = tf.get_variable(
                    'Wi', [self.config.embed_size, self.config.hidden_size])
                Uf = tf.get_variable(
                    'Uf', [self.config.hidden_size, self.config.hidden_size])
                Wf = tf.get_variable(
                    'Wf', [self.config.embed_size, self.config.hidden_size])
                Uo = tf.get_variable(
                    'Uo', [self.config.hidden_size, self.config.hidden_size])
                Wo = tf.get_variable(
                    'Wo', [self.config.embed_size, self.config.hidden_size])
                Uc = tf.get_variable(
                    'Uc', [self.config.hidden_size, self.config.hidden_size])
                Wc = tf.get_variable(
                    'Wc', [self.config.embed_size, self.config.hidden_size])
                i = tf.nn.sigmoid(tf.matmul(h, Ui) + tf.matmul(current_sent, Wi))
                f = tf.nn.sigmoid(tf.matmul(h, Uf) + tf.matmul(current_sent, Wf))
                o = tf.nn.sigmoid(tf.matmul(h, Uo) + tf.matmul(current_sent, Wo))
                ct = tf.nn.tanh(tf.matmul(h, Uc) + tf.matmul(current_sent, Wc))
                c = tf.mul(f,c) + tf.mul(i,ct)
                h = tf.mul(o, tf.nn.tanh(c))
            self.final_state_questions = h

        return self.final_state_context, self.final_state_questions


    def run_epoch(self, session, data, train_op=None, verbose=10):
        config = self.config
        if not train_op:
            train_op = tf.no_op()
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
            
            feed = {self.context_placeholder: context,
                    self.questions_placeholder: questions,
                    self.answers_placeholder: answers,
                    self.output_gates_placeholder1: output_gates1,
                    self.output_gates_placeholder2: output_gates2,
                    self.output_placeholder: output}
            loss, _, pred = session.run(
                [self.calculate_loss, train_op, self.predictions], feed_dict=feed)
            total_loss.append(loss)
            pred = np.argmax(pred, 1)
            pos_preds += np.sum(pred==answers)
            num_answers += len(answers)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        return np.mean(total_loss), pos_preds / num_answers

def test_RNNLM():
    config = Config()
    with tf.variable_scope('RNNLM') as scope:
        model = RNNContext_Model(config)

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
                train_op=model.train_step)
            print('Training cross-entropy: {}'.format(train_ce))
            print('Training accuracy: {}'.format(train_accuracy))

            valid_ce, valid_accuracy = model.run_epoch(
                session, model.encoded_valid)
            print('Validation cross-entropy: {}'.format(valid_ce))
            print('Validation accuracy: {}'.format(valid_accuracy))

            if valid_ce < best_val_ce:
                best_val_ce = valid_ce
                best_val_epoch = epoch
                saver.save(session, './ptb_rnnlm.weights')
            if epoch - best_val_epoch > config.early_stopping:
                break
            print('Total time: {}'.format(time.time() - start))
        
        saver.restore(session, 'ptb_rnnlm.weights')    
        test_ce, test_accuracy = model.run_epoch(
            session, model.encoded_test)
        print('=-=' * 5)
        print('Test cross-entropy: {}'.format(test_ce))
        print('Test accuracy: {}'.format(test_accuracy))
        print('=-=' * 5)

if __name__ == "__main__":
    test_RNNLM()
