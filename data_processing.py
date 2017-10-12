import numpy as np
import json
from math import floor

from utils import Vocab
from nltk.tokenize import sent_tokenize, word_tokenize


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

    def variable_len_context(self, encoded_data):
        '''Array of sentences' lengths from the different contexts.'''
        return [len(par['context']) for par in encoded_data]

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
            with open('/home/nhennetier/pretrained_embeddings/glove.840B.300d.txt') as glove_file:
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
        
        self.config.lengths_sent_context = {"train": self.variable_len_sent_context(self.encoded_train),
                                            "valid": self.variable_len_sent_context(self.encoded_valid)}
        self.config.lengths_sent_questions = {"train": self.variable_len_sent_questions(self.encoded_train),
                                              "valid": self.variable_len_sent_questions(self.encoded_valid)}

        self.config.lenghts_context = {"train": self.variable_len_context(self.encoded_train),
                                       "valid": self.variable_len_context(self.encoded_valid)}
        
        self.encoded_train = self.padding_dataset(self.encoded_train)
        self.encoded_valid = self.padding_dataset(self.encoded_valid)
        
        #Dev/Test split
        n = int(len(self.encoded_valid) / 2)
        self.encoded_test = self.encoded_valid[n:]
        self.config.lengths_sent_context["test"] = self.config.lengths_sent_context["valid"][n:]
        self.config.lengths_sent_questions["test"] = self.config.lengths_sent_questions["valid"][n:]
        self.config.lenghts_context["test"] = self.config.lenghts_context["valid"][n:]
        self.encoded_valid = self.encoded_valid[:n]
        self.config.lengths_sent_context["valid"] = self.config.lengths_sent_context["valid"][:n]
        self.config.lengths_sent_questions["valid"] = self.config.lengths_sent_questions["valid"][:n]
        self.config.lenghts_context["valid"] = self.config.lenghts_context["valid"][:n]