import os
import re
import sys
import codecs
from nltk.parse import stanford
from nltk.tree import ParentedTree
import dill as pickle
import time
import configparser
import csv
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas
import random
import numpy

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Concatenate, concatenate
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import keras.backend






class CustomLabelEncoder:

    def __init__(self):

        self.l2i = defaultdict(int)
        self.i2l = defaultdict(str)
        self.i = 0

    def encode(self, m):

        em = []
        for row in m:
            er = []
            for val in row:
                if val in self.l2i:
                    er.append(self.l2i[val])
                else:
                    self.i += 1
                    self.l2i[val] = self.i
                    self.i2l[self.i] = val
                    er.append(self.i)
            em.append(er)
        return em

    def vocabIndex(self, m):

        self.w2i = defaultdict(int)
        self.i2w = defaultdict(str)
        self.wi = 1 # 0 is reserved for unknown words
        for row in m:
            w = row[0]
            if not w in self.w2i:
                self.w2i[w] = self.wi
                self.i2w[self.wi] = w
                self.wi += 1
        

class CustomCONLLToken:

    def __init__(self, uid, sid, stid, token, pos_coarse, pos_fine, segmentStarter, head, deprel):
        self.uid = int(uid)
        self.sid = int(sid)
        self.stid = int(stid)
        self.token = token
        self.pos_coarse = pos_coarse
        self.pos_fine = pos_fine
        self.head = int(head)
        self.deprel = deprel
        self.segmentStarter = segmentStarter

    def addFullSentence(self, sentaslist):
        self.fullSentence = sentaslist
        

config = configparser.ConfigParser()
config.read('config.ini')
os.environ['JAVAHOME'] = config['lexparser']['javahome']
os.environ['STANFORD_PARSER'] = config['lexparser']['stanfordParser']
os.environ['STANFORD_MODELS'] = config['lexparser']['stanfordModels']
os.environ['CLASSPATH'] = config['lexparser']['path']
lexParser = stanford.StanfordParser(model_path=config['lexparser']['germanModel'])



def RNN_internalEmbeddings(dim, voc_size, bsize):
    """
    inputs = Input(name='inputs',shape=[dim])
    layer = Embedding(voc_size,50,input_length=dim)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    """
    print('debugging dim:', dim)
    print('debugging voc_size:', voc_size)
    print('debugging bsize:', bsize)

    #inputs1 = Input(name='i1',shape=[bsize,dim])
    inputs1 = Input(name='i1',shape=[bsize,dim])# this one is oh encoding dim (without the word index), play around with the 20 (bsize) afterward
    inputs2 = Input(name='i2',shape=[bsize,1]) # 1, because just one int for word index in ...
    # voc_size is just that now: nr of unique words
    
    emblayer = Embedding(voc_size,50)(inputs2) # 50 is prediction size # inputs2 because this one is about words only
    

    layer = Concatenate()([inputs1, emblayer])
    #layer = concatenate([inputs1, emblayer])
    layer = LSTM(64,return_sequences=True)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('softmax')(layer)
    model = Model(inputs=[inputs1, inputs2],outputs=layer)
    return model


def RNN_externalEmbeddings(dim):
    inputs = Input(name='inputs',shape=[dim])
    layer = LSTM(64)(inputs)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model



def addFullSentences(conlltokens, sid2tokens):

    for ct in conlltokens:
        fullsent = sid2tokens[ct.sid]
        ct.addFullSentence(fullsent)

        

def customConllParser(cfile):

    """
    https://stackoverflow.com/questions/27416164/what-is-conll-data-format
    ID (index in sentence, starting at 1)
    FORM (word form itself)
    LEMMA (word's lemma or stem)
    POS (part of speech)
    FEAT (list of morphological features separated by |)
    HEAD (index of syntactic parent, 0 for ROOT)
    DEPREL (syntactic relationship between HEAD and this word)
    """
    
    tokens = []
    reader = csv.reader(codecs.open(cfile), delimiter='\t')
    sentences = []
    sent = []
    uid2sid = defaultdict(str)
    sid2tokens = defaultdict(list)
    sid = 1
    uid = 0
    nrrows = len(codecs.open(cfile).readlines())
    for row in reader:
        if len(row) > 5:
            sTokenId = row[0]
            token = row[1]
            lemma = row[2]
            pos_coarse = row[3]
            pos_fine = row[4]
            morph_feats = row[5]
            head_tokenid = row[6]
            deprel = row[7]
            #secondary_edges = row[8] # not using this for now
            #projective_heads = row[9] # not using this for now
            if lemma.startswith('$') or lemma == 'PUNCT': # for some tab chars, there was no surface form and lemma
                lemma = token
                pos_coarse = row[2]
                pos_fine = row[3]
                morph_feats = row[4]
                head_tokenid = row[5]
                deprel = row[6]
            segmentStarter = False
            if re.search('BeginSeg=Yes', ' '.join(row[8:])):
                segmentStarter = True
            cct = CustomCONLLToken(uid, sid, sTokenId, token, pos_coarse, pos_fine, segmentStarter, head_tokenid, deprel)
            uid2sid[uid] = sid
            sent.append(token)
            tokens.append(cct)
        else:
            sentences.append(sent)
            sid2tokens[sid] = sent
            sent = []
            sid += 1
        uid += 1
        if uid == nrrows:
            sentences.append(sent)
            sid2tokens[sid] = sent
            sent = []
            sid += 1
        
    return tokens, sentences, sid2tokens # redundancy here for sentences and sid2tokens
            
def parseCorpus(sentences):

    pm = {}
    sl = len(sentences)
    for i, sentence in enumerate(sentences):
        try:
            sentence = ' '.join(sentence)
            sentence = re.sub('\)', ']', re.sub('\(', '[', sentence)) # remember to do this again at run/eval time!
            if i % 100 == 0:
                sys.stderr.write("INFO: Parsing %i of %i...\n" % (i+1, sl))
            tokens = sentence.split()
            ptree = None
            tree = lexParser.parse(tokens)
            ptreeiter = ParentedTree.convert(tree)
            for t in ptreeiter:
                ptree = t
                break
            pm[sentence] = ptree
        except:
            sys.stderr.write('WARNING: Skipped sentence during preparsing...\n')
    return pm

def getId2CTdict(conlltokens):

    uid2ct = defaultdict()
    sidtid2ct = defaultdict()
    for ct in conlltokens:
        uid2ct[ct.uid] = ct
        sidtid2ct[(ct.sid, ct.stid)] = ct
    return uid2ct, sidtid2ct

def getTopN(conlltokens, n, lowercase=False):

    fd = defaultdict(int)
    for ct in conlltokens:
        if lowercase:
            fd[ct.token.lower()] += 1
        else:
            fd[ct.token] += 1
    return set([x[0] for x in sorted(fd.items(), key = lambda x: x[1], reverse=True)[:n]])
        

def getMatrix(conlltokens, nonfilterLowFreq=False):

    uid2ct, sidtid2ct = getId2CTdict(conlltokens)
    top_n = getTopN(conlltokens, 100, True)
    m = []
    for ct in conlltokens:
        row = []
        #('word', '=', 0.018278717735402652) # this word lowercased if it’s in the top 100 most freq items, else its pos
        word = ct.token
        if not nonfilterLowFreq: # nice double negation in the functioning of this flag... 
            if ct.token.lower() in top_n: # lowercasing here corresponds to lc flag used in getTopN above!!!
                word = ct.token.lower()
            else:
                word = ct.pos_coarse
        #word = ct.token # switch this line and the few above to change between taking all words or only the top 100 (and pos otherwise)
        row.append(word)

        #('last', '=', 0.730843358367783)  # Is this the last word in the sentence? (Clearly the most important feature)
        last = False
        if ct.stid == len(ct.fullSentence):
            last = True
        row.append(last)
        #('dist2par', '=', 0.03922734913136085)  # distance in tokens to dependency parent
        dist2par = ct.stid - ct.head # could experiment with absolute distance here
        row.append(dist2par)
        #('parent_func', '=', 0.03496516497209843)  # gram. func of parent
        parent_func = None
        if (ct.sid, ct.head) in sidtid2ct:
            parent_func = sidtid2ct[(ct.sid, ct.head)].deprel
        row.append(parent_func)
        #(‘next_func’, ‘=’, 0.027653679478951993)  # func of next word
        next_func = None
        if (ct.sid, ct.stid+1) in sidtid2ct:
            next_func = sidtid2ct[(ct.sid, ct.stid+1)].deprel
        row.append(next_func)
        #('next_pos', '=', 0.02664229467168697)  # pos of next word
        next_pos = None
        if (ct.sid, ct.stid+1) in sidtid2ct:
            next_pos = sidtid2ct[(ct.sid, ct.stid+1)].pos_fine
        row.append(next_pos)
        #('func', '=', 0.02635001365803647)  # func of this word
        func = ct.deprel
        row.append(func)
        #('prev_func', '=', 0.024408972670520945) …
        prev_func = None
        if (ct.sid, ct.stid-1) in sidtid2ct:
            prev_func = sidtid2ct[(ct.sid, ct.stid-1)].deprel
        row.append(prev_func)
        #('parent_pos', '=', 0.021090340795278512)
        parent_pos = None
        if (ct.sid, ct.head) in sidtid2ct:
            parent_pos = sidtid2ct[(ct.sid, ct.head)].pos_fine
        row.append(parent_pos)
        #('prev_pos', '=', 0.020115833651870505)
        prev_pos = None
        if (ct.sid, ct.stid-1) in sidtid2ct:
            prev_pos = sidtid2ct[(ct.sid, ct.stid-1)].pos_fine
        row.append(prev_pos)
        #('pos', '=', 0.014981602331863216)
        pos = ct.pos_fine
        row.append(pos)
        #('next_upper', '=', 0.005338427363756266)  # next word is upper case
        next_upper = False
        if (ct.sid, ct.stid+1) in sidtid2ct:
            if sidtid2ct[(ct.sid, ct.stid+1)].token[0].isupper():
                next_upper = True
        row.append(next_upper)
        #('parent_upper', '=', 0.004352975814376736)  # parent is upper case
        parent_upper = False
        if (ct.sid, ct.head) in sidtid2ct:
            if sidtid2ct[(ct.sid, ct.head)].token[0].isupper():
                parent_upper = True
        row.append(parent_upper)
        #('prev_upper', '=', 0.003879340368824429) …
        prev_upper = False
        if (ct.sid, ct.stid-1) in sidtid2ct:
            if sidtid2ct[(ct.sid, ct.stid-1)].token[0].isupper():
                prev_upper = True
        row.append(prev_upper)
        #('word_upper', '=', 0.0018719289881911527)
        word_upper = False
        if ct.token[0].isupper():
            word_upper = True
        row.append(word_upper)
        label = ct.segmentStarter
        row.append(label)

        m.append(row)
    return m
        

def trainClassifier(matrix, headers):
    
    df = pandas.DataFrame(matrix, columns=headers)
    Y = df.class_label
    labels = list(set(Y))
    Y = numpy.array([labels.index(x) for x in Y])
    X = df.iloc[:,:len(headers)-1]
    X = numpy.array(X)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, Y)

    return clf

def evalClassifier(clf, testmatrix, le, headers):

    tdf = pandas.DataFrame(testmatrix, columns=headers)
    Y = tdf.class_label
    labels = list(set(Y))
    Y = numpy.array([labels.index(x) for x in Y])
    X = tdf.iloc[:,:len(headers)-1]
    X = numpy.array(X)
    results = clf.predict(X)
    p, r, f, tp, fp, fn, tn = getNumbers(X, Y, results, le)
    #tp, fp, fn, tn = getNumbers(X, Y, results, le)

    cp = tp/(tp+fp)
    cr = tp/(tp+fn)
    cf = 2 * ((cp * cr) / (cp + cr))

    tn, fp, fn, tp = confusion_matrix(Y, results).ravel()
    
    sys.stderr.write('\nTEST RESULTS:\n\n')
    sys.stderr.write('sklearn precision:'+ str(p) + '\n')
    sys.stderr.write('sklearn recall:' + str(r) + '\n')
    sys.stderr.write('sklearn f:' + str(f) + '\n')
    sys.stderr.write('custom precision:' + str(cp) + '\n')
    sys.stderr.write('custom recall:' + str(cr) + '\n')
    sys.stderr.write('custom f:' + str(cf) + '\n')
    
    

def getNumbers(test_features, test_labels, results, le):
        
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for inp, pred, label in zip(test_features, results, test_labels):
        reconstr = None
        if le:
            reconstr = [le.i2l[x] for x in inp]
        if label == 0:
            if pred == 0:
                tn += 1
            elif pred == 1:
                #if debugFlag:
                    #print('False Positive for:', reconstr)
                fp += 1
        elif label == 1:
            if pred == 1:
                tp += 1
            elif pred == 0:
                fn += 1
                #if debugFlag:
                    #print('False Negative for:', reconstr)


    precision = 0
    recall = 0
    f1 = 0
    precision = precision_score(test_labels, results)
    recall = recall_score(test_labels, results)
    f1 = f1_score(test_labels, results)
        
    return precision, recall, f1, tp, fp, fn, tn
    #return tp, fp, fn, tn

def loadExternalEmbeddings(embfile):

    starttime = time.time()
    sys.stderr.write('INFO: Loading external embeddings from %s.\n' % embfile)
    ed = defaultdict()
    #"""
    with open(embfile, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            values = line.split()
            ed[values[0]] = numpy.array([float(x) for x in values[1:]])
    #"""
    endtime = time.time()
    sys.stderr.write('INFO: Done loading embeddings. Took %s seconds.\n' % (str(endtime - starttime)))
    return ed


def getFeaturesForTestData(matrix, headers, le, f2ohvlen, f2ohvpos, embd=False):

    df = pandas.DataFrame(matrix, columns=headers)
    Y = df.class_label
    labels = list(set(Y))
    Y = numpy.array([labels.index(x) for x in Y])

    dim = 300
    X1 = []
    X2 = []
    for row in matrix:
        token = None
        if le:
            token = le.i2l[row[0]]
        else:
            token = row[0]
        nrow = []
        if embd:
            if token in embd:
                for item in embd[token]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                    nrow.append(item)
        else:
            #nextrows = [0] * f2ohvlen[0]
            #if token in f2ohvpos[0]:
                #nextrows[f2ohvpos[0][token]] = 1
            #nrow += nextrows
            if row[0] in le.w2i:
                X1.append([le.w2i[row[0]]])
            else:
                X1.append([0]) # unknown words get a 0
            
        for index, val in enumerate(row[1:-1]):
            nextrows = [0] * f2ohvlen[index+1]
            if val in f2ohvpos[index+1]:
                nextrows[f2ohvpos[index+1][val]] = 1
            else: # feature not seen during training (play around with UNK feature (dedicated pos for unknowns, or just set all to 0)
                nextrows[nextrows[random.randint(0, len(nextrows)-1)]] = 1
            nrow += nextrows
        X2.append(nrow)

    X1 = numpy.array(X1)
    X2 = numpy.array(X2)

    return X1, X2, Y

def createEmbeddingMatrix(matrix, headers, le, embd=False):

    df = pandas.DataFrame(matrix, columns=headers)
    Y = df.class_label
    labels = list(set(Y))
    Y = numpy.array([labels.index(x) for x in Y])


    f2ohvpos = defaultdict(lambda : defaultdict(int))
    f2 = defaultdict(set)
    f2ohvlen = defaultdict()
    rowsize = 0
    for row in matrix:
        rowsize = len(row)
        for pos, val in enumerate(row):
            f2[pos].add(val)
    for i in f2:
        f2ohvlen[i] = len(f2[i])
        for c, i2 in enumerate(f2[i]):
            f2ohvpos[i][i2] = c

    dim = 300

    
    X1 = []
    X2 = []
    input_dim = 0
    for row in matrix:
        # first n are reserved for word embeddings:
        token = None
        if le:
            token = le.i2l[row[0]]
        else:
            token = row[0]
        nrow = []
        if embd:
            if token in embd:
                for item in embd[token]:
                    nrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                    nrow.append(item)
        else:
            #nextrows = [0] * f2ohvlen[0]
            #nextrows[f2ohvpos[0][token]] = 1
            #nrow += nextrows
            # not taking ohv for word, but just plain index instead
            X1.append([le.w2i[row[0]]])
            
        for index, val in enumerate(row[1:-1]):
            nextrows = [0] * f2ohvlen[index+1] # index+1, because the first value (word itself) I'm doing with embeddings
            nextrows[f2ohvpos[index+1][val]] = 1
            nrow += nextrows
        input_dim = len(nrow)
        X2.append(nrow)

    X1 = numpy.array(X1)
    X2 = numpy.array(X2)
        
    return X1, X2, Y, f2ohvlen, f2ohvpos, input_dim

def randomforest(le, e_train, e_test, headers):

    clf = trainClassifier(e_train, headers)
    evalClassifier(clf, e_test, le, headers)

    #TEST RESULTS:
    
    #sklearn precision:0.9029850746268657
    #sklearn recall:0.8231292517006803
    #sklearn f:0.8612099644128115
    #custom precision:0.9029850746268657
    #custom recall:0.8231292517006803
    #custom f:0.8612099644128115

def prepareDataInBatchSize(X, bsize, step_size):

    # what this does, is cutting the data up in pieces with some step_size, such that:
    # [[33], [44], [58] ,[222]] becomes
    # [[[33], [44]], [[44, 58]], [[58, 222]]] # this would be for the scenario bsize 2, step size 1...
    # and this should not be 0 to 20, 20 to 40, 40 to 60, but instead 0 to 20, 5 to 25, 10 to 30, etc.
    n = []
    for i in range(0, len(X), step_size):
        n.append(X[i:i+bsize]) # this may go wrong because the len of sequences are probably not the same (the last one may differ, because the nr of features may not be divisible by the bsize)
    n = numpy.array(n)
    # or, just trying with keras:
    #n = keras.backend.reshape(X, bsize)
    
    return n

def internalEmbeddings(train, test, headers, epochs):

    #embd = loadExternalEmbeddings('/home/peter/phd/PhdPlayground/cc.de.300.vec')
    le = CustomLabelEncoder()
    le.vocabIndex(train)
    X1, X2, Y, f2ohvlen, f2ohvpos, dim = createEmbeddingMatrix(train, headers, le, False) # there's probably a whole bunch of keras utils to do exactly this, but for understanding how it works, want to build it myself
    sys.stderr.write('INFO: Number of training samples: %i\n' % len(X1))
    # with embeddings and one-hot vectors for categorical features:
    X1_test, X2_test, Y_test = getFeaturesForTestData(test, headers, le, f2ohvlen, f2ohvpos, False)
    sys.stderr.write('INFO: Number of test samples: %i\n' % len(X1_test))

    # NOTE: X1 are the words, X2 are the oh features
    
    #dim = len(headers)-1 # use this for same feature set as randomforest
    voc_size = (X1.max()+1).astype('int64')
    bsize = 20
    step_size = 5
    X1 = prepareDataInBatchSize(X1, bsize, step_size)
    X2 = prepareDataInBatchSize(X2, bsize, step_size)
    print('debug X1 shape:', numpy.shape(X1))
    print('debug X2 shape:', numpy.shape(X2))

    
    model = RNN_internalEmbeddings(dim, voc_size, bsize)
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

    # preparing data in right dimension
    #split up X1 and X2 in, such that X2 is [[[33, 44, 56], [1,45,7], ... til 20], [[], [], []], [[], [], [],[], [], [],[], [], [],[], [], [],[], [], [],[], [], [],[], []]] # last one should be len of 20, corresponding to 20 in rnn definition
    # and this should not be 0 to 20, 2o to 40, 40 to 60, but instead 0 to 20, 5 to 25, 10 to 30, etc.
    
    #X1 is now [[0,0,0,1,0,0,1,0,0,1,1],....]
    #X2 is [[33], 56, 111] # find out if this should be [33,44,56] or [[33],[44], etc
    model.fit((X2, X1), Y, batch_size=128,epochs=epochs,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

    X1_test = prepareDataInBatchSize(X1_test, bsize, step_size)
    X2_test = prepareDataInBatchSize(X2_test, bsize, step_size)
    results = model.predict((X2_test, X1_test))
    binary_results = [0 if x  < 0.5 else 1 for x in results]
    p, r, f, tp, fp, fn, tn = getNumbers(X1_test, Y_test, binary_results, le)

    cp = tp/(tp+fp)
    cr = tp/(tp+fn)
    cf = 2 * ((cp * cr) / (cp + cr))

    tn, fp, fn, tp = confusion_matrix(Y_test, binary_results).ravel()
    print('sklearn confusion matrix:')
    print('tn:', tn)
    print('fp:', fp)
    print('tp:', tp)
    print('fn:', fn)

    tn, fp, fn, tp = confusion_matrix(Y_test, results).ravel()
    print('sklearn confusion matrix:')
    print('tn:', tn)
    print('fp:', fp)
    print('tp:', tp)
    print('fn:', fn)

    
    sys.stderr.write('\nTEST RESULTS WITH INTERNAL EMBEDDINGS:\n\n')
    sys.stderr.write('sklearn precision:'+ str(p) + '\n')
    sys.stderr.write('sklearn recall:' + str(r) + '\n')
    sys.stderr.write('sklearn f:' + str(f) + '\n')
    sys.stderr.write('custom precision:' + str(cp) + '\n')
    sys.stderr.write('custom recall:' + str(cr) + '\n')
    sys.stderr.write('custom f:' + str(cf) + '\n')


def externalEmbeddings(le, e_train, e_test, headers, epochs):

    embd = loadExternalEmbeddings('/home/peter/phd/PhdPlayground/cc.de.300.vec')
    # NOTE: the feature part needs work anyway to get it working with external embeddings, fix things like vocab size etc.
    X, Y, f2ohvlen, f2ohvpos, dim = createEmbeddingMatrix(e_train, headers, le, embd) # there's probably a whole bunch of keras utils to do exactly this, but for understanding how it works, want to build it myself

    sys.stderr.write('INFO: Number of training samples: %i\n' % len(X))

    model = RNN_externalEmbeddings(dim)
    #model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])

    model.fit(X, Y, batch_size=128,epochs=epochs,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
    
    # with embeddings and one-hot vectors for categorical features:
    X_test, Y_test = getFeaturesForTestData(e_test, headers, le, f2ohvlen, f2ohvpos, embd)

    sys.stderr.write('INFO: Number of test samples: %i\n' % len(X_test))
    results = model.predict(X_test)
    binary_results = [0 if x  < 0.5 else 1 for x in results]
    print('results:', results)

    p, r, f, tp, fp, fn, tn = getNumbers(X_test, Y_test, binary_results, le)

    cp = tp/(tp+fp)
    cr = tp/(tp+fn)
    cf = 2 * ((cp * cr) / (cp + cr))

    tn, fp, fn, tp = confusion_matrix(Y_test, binary_results).ravel()

    
    sys.stderr.write('\nTEST RESULTS WITH EXTERNAL EMBEDDINGS:\n\n')
    sys.stderr.write('sklearn precision:'+ str(p) + '\n')
    sys.stderr.write('sklearn recall:' + str(r) + '\n')
    sys.stderr.write('sklearn f:' + str(f) + '\n')
    sys.stderr.write('custom precision:' + str(cp) + '\n')
    sys.stderr.write('custom recall:' + str(cr) + '\n')
    sys.stderr.write('custom f:' + str(cf) + '\n')
    






if __name__ == '__main__':

    """
    tokens, sentences = customConllParser('dev.conll')
    memorymap = parseCorpus(sentences)
    memname = 'dev.parses.pickle'
    with open(memname, 'wb') as handle:
        pickle.dump(memorymap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """
    
    """
    tokens, sentences = customConllParser('test.conll')
    memorymap = parseCorpus(sentences)
    memname = 'test.parses.pickle'
    with open(memname, 'wb') as handle:
        pickle.dump(memorymap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """

    """
    tokens, sentences = customConllParser('train.conll')
    memorymap = parseCorpus(sentences)
    memname = 'train.parses.pickle'
    with open(memname, 'wb') as handle:
        pickle.dump(memorymap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """

    traintokens, sentences, sid2tokens =  customConllParser('train.conll')
    addFullSentences(traintokens, sid2tokens)
    trainmatrix = getMatrix(traintokens)
    testtokens, sentences, sid2tokens =  customConllParser('test.conll')
    addFullSentences(testtokens, sid2tokens)
    testmatrix = getMatrix(testtokens)
    le = CustomLabelEncoder()
    e_train = le.encode(trainmatrix)
    e_test = le.encode(testmatrix)
    headers = ['word', 'last', 'dist2par', 'parent_func', 'next_func', 'next_pos', 'func', 'prev_func', 'parent_pos', 'prev_pos', 'pos', 'next_upper', 'parent_upper', 'prev_upper', 'word_upper', 'class_label']


    randomforest(le, e_train, e_test, headers)
    
    
    """
    Amir's baseline
    ('last', '=', 0.730843358367783)  # Is this the last word in the sentence? (Clearly the most important feature)
    ('dist2par', '=', 0.03922734913136085)  # distance in tokens to dependency parent
    ('parent_func', '=', 0.03496516497209843)  # gram. func of parent
    (‘next_func’, ‘=’, 0.027653679478951993)  # func of next word
    ('next_pos', '=', 0.02664229467168697)  # pos of next word
    ('func', '=', 0.02635001365803647)  # func of this word
    ('prev_func', '=', 0.024408972670520945) …
    ('parent_pos', '=', 0.021090340795278512)
    ('prev_pos', '=', 0.020115833651870505)
    ('word', '=', 0.018278717735402652) # this word lowercased if it’s in the top 100 most freq items, else its pos
    ('pos', '=', 0.014981602331863216)
    ('next_upper', '=', 0.005338427363756266)  # next word is upper case
    ('parent_upper', '=', 0.004352975814376736)  # parent is upper case
    ('prev_upper', '=', 0.003879340368824429) …
    ('word_upper', '=', 0.0018719289881911527)
    """

    ### neural part starts here
    # based on https://www.kaggle.com/kredy10/simple-lstm-for-text-classification

    trainmatrix = getMatrix(traintokens, True) # re-generating matrices, because not taking POS tag for non-top-n words here
    testmatrix = getMatrix(testtokens, True)
    epochs = 10
    internalEmbeddings(trainmatrix, testmatrix, headers, epochs)

    #externalEmbeddings(le, e_train, e_test, headers, epochs)
