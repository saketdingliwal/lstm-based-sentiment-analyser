
# coding: utf-8

# In[1]:

import torch
import json
import sys
import numpy as np
import pickle
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk.sentiment
import random
import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
negate_list = ["not","never","no"]
tokenizer = RegexpTokenizer(r'\w+')
en_stop = set([ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ])

use_gpu = torch.cuda.is_available()


def add_adj(documents):
    new_docs = []
    for doc in documents:
        doc = doc.lower()
        new_words = []
        words = doc.split()
        for word in words:
            new_words.append(word)
            if word in positive_set or word in negative_set:
                new_words.append(word)
                new_words.append(word)
        new_doc = ' '.join(new_words)
        new_docs.append(new_doc)
    return new_docs



# In[ ]:


test_documents = []
filepath = sys.argv[1]
test_summary = []
with open(filepath,'r') as fp:
    line = fp.readline()
    while line:
        input_data = (json.loads(line))
        test_documents.append(input_data["reviewText"]+input_data["summary"]+input_data["summary"])
        line = fp.readline()


# In[1]:


def cleaning2(docs):
    new_docs = []
    for document in docs:
        raw = document.lower()
        raw = raw.replace("<br /><br />", " ")
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [token for token in tokens if token not in en_stop]
        documentWords = ' '.join(stopped_tokens)
        new_docs.append(documentWords)
    return new_docs


# In[2]:


def not_clear(tokens):
    i =0
    for token in tokens:
        if token in negate_list or token[-3:]=="n't":
            if i+1 < len(tokens):
                tokens[i+1] =  tokens[i+1] + "_NEG"
            if i+2 < len(tokens):
                tokens[i+2] = tokens[i+2] + "_NEG"
        i+=1
    return tokens


# In[3]:


def negate(documents):
    new_documents = []
    for doc in documents:
#         doc = doc.lower()
        words = doc.split()
        new_words = not_clear(words)
        newdocument = ' '.join(new_words)
        new_documents.append(newdocument)
    return new_documents


# In[ ]:


# with open('clf.pkl', 'rb') as f:
#     clf = pickle.load(f)
# with open('dict.pkl', 'rb') as f:
#     bigram_vect = pickle.load(f)



class LSTM_MODEL(torch.nn.Module) :
    def __init__(self,vocabsize,embedding_dim,hidden_dim,num_layers,drop_layer,num_classes):
        super(LSTM_MODEL,self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocabsize, embedding_dim)
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)
        self.linearOut = nn.Linear(2*hidden_dim,num_classes)
    def forward(self,inputs):
        x = self.embeddings(inputs).view(len(inputs),batch_size,-1)
        hidden = self.init_hidden()
        lstm_out,lstm_h = self.lstm(x,hidden)
        x = lstm_out[-1]
        x = self.linearOut(x)
        x = F.log_softmax(x)
        return x
    def init_hidden(self):
        h0 = Variable(torch.zeros(2*self.num_layers,batch_size,self.hidden_dim).cuda())
        c0 = Variable(torch.zeros(2*self.num_layers,batch_size,self.hidden_dim).cuda())
        return (h0,c0)


def clip_doc(doc):
    sent_vect = []
    words = doc.split()
    for word in words:
        if word not in word_to_idx:
            sent_vect.append(word_to_idx['unch'])
        else:
            sent_vect.append(word_to_idx[word])
    if len(sent_vect) > sen_len:
        sent_vect = sent_vect[0:sen_len]
    else:
        diff = sen_len - len(sent_vect)
        for i in range(diff):
            sent_vect.append(word_to_idx['padd'])
    return sent_vect




def batchify(label_doc,start_index):
    label_batch = []
    doc_batch = []
    for i in range(start_index,start_index+batch_size):
        label_batch.append(int(label_doc[i][1])-1)
        doc_batch.append(label_doc[i][0])
    return (label_batch,doc_batch)




with open('simple_dict.pkl','rb') as f :
    word_to_idx = pickle.load(f)

documents = test_documents

documents = cleaning2(documents)
vocabsize = len(word_to_idx)
sen_len = 300
doc_label_pair = []
ind = 0
batch_size = 250
num_batch = len(documents)//batch_size
for ind in range(len(documents)):
    doc_label_pair.append((clip_doc(documents[ind]),0))

vocabsize = len(word_to_idx)
emebed_dim = 400
hidden_dim = 100
model = LSTM_MODEL(vocabsize,emebed_dim,hidden_dim,1,0.5,5)
model = model.cuda()
model.load_state_dict(torch.load('simple4.pth'))
predicted_labels = []
uptill=0
for iterr in range(num_batch-1):
    _,batch_data = batchify(doc_label_pair,iterr*batch_size)
    batch_data = Variable(torch.LongTensor(batch_data).cuda())
    y_pred = model(batch_data.t())
    _, predicted = torch.max(y_pred.data, 1)
    for i in range(len(predicted)):
        predicted_labels.append(predicted[i])
        uptill+=1


batch_size = 1
for i in range(uptill,len(documents)):
    input_data = clip_doc(documents[i])
    input_data = Variable(torch.LongTensor(input_data).cuda())
    y_pred = model(input_data)
    _, predicted = torch.max(y_pred.data, 1)
    predicted_labels.append(predicted)


for i in range(len(predicted_labels)):
    if int(predicted_labels[i]) == 1:
        predicted_labels[i] = 0
    if int(predicted_labels[i]) == 3:
        predicted_labels[i] = 4
    predicted_labels[i] = int(predicted_labels[i])


# In[ ]:


output_file = open(sys.argv[2],'w')
for i in range(len(predicted_labels)):
    if predicted_labels[i]==0:
        output_file.write("1\n")
    elif predicted_labels[i]==2:
        output_file.write("3\n")
    else:
        output_file.write("5\n")
