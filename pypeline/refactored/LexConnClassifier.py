#!/usr/bin/python3

import os
import sys
import numpy
import pandas
import utils
import codecs
import json
import dill as pickle
from tqdm import tqdm
from nltk.parse import stanford
from nltk.tree import ParentedTree
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import f1_score, precision_score, recall_score
from nltk import word_tokenize, sent_tokenize
import DimLexParser
import PCCParser

def getDimlexCandidates(parser):

    dimlex = DimLexParser.parseXML(parser.config['dimlex']['dimlex'])
    dimlextuples = {}
    for entry in dimlex:
        altdict = entry.alternativeSpellings
        for item in altdict: # canonical form is always in list of alt spellings
            if altdict[item]['phrasal'] == 'cont':
                tupl = tuple(word_tokenize(item))
                dimlextuples[tupl] = 'cont'
            elif altdict[item]['single'] == 'cont':
                tupl = tuple(word_tokenize(item))
                dimlextuples[tupl] = 'cont'
            elif altdict[item]['single'] == 'discont':
                tupl = tuple(word_tokenize(item))
                dimlextuples[tupl] = 'discont'

    return dimlextuples


class LexConnClassifier:

    mainclassdict = {'APPR':'prep','KOKOM':'cco','APART':'prep','KOUS':'csu','KON':'cco','ADV':'adv','KOUI':'csu'}
    
    def __init__(self, parser):
        self.customEncoder = defaultdict(int)
        self.customDecoder = defaultdict(str)
        self.maxEncoderId = 1
        self.dimlexconnectives = getDimlexCandidates(parser)

    def getCategoricalLength(self, l):

        if l < 5:
            return 0
        elif l < 10:
            return 1
        elif l < 15:
            return 2
        elif l < 20:
            return 3
        elif l < 25:
            return 4
        elif l < 30:
            return 6
        elif l < 35:
            return 7
        elif l < 40:
            return 8
        else:
            return 9
    def isSInitial(self, firsttoken, ptree):

        for s in ptree.subtrees(lambda t: t.label().startswith('S')):
            if s.leaves():
                if s.leaves()[0] == firsttoken: # sloppy coding; if a candidate occurs multiple times in a sentence, all its instances get True if this is true for one instance... (should not happen too often though)
                    return True


    def classify(self, parser, _input):

        sys.stderr.write('INFO: Identifying connectives in input...\n')
        sentences = sent_tokenize(_input)
        out = []
        for sentenceno, sentence in enumerate(tqdm(sentences)):
            tokens = word_tokenize(sentence)
            sdict = {'sentence_id':sentenceno}
            sdict['tokens'] = {}
            sdict['connectives'] = []
            for tid, t in enumerate(tokens):
                sdict['tokens'][tid] = t
            ptree = None
            tree = parser.lexParser.parse(tokens)
            ptreeiter = ParentedTree.convert(tree)
            for t in ptreeiter:
                ptree = t
                break # always taking the first, assuming that this is the best scoring tree.
            for dc in self.dimlexconnectives:
                feat = None
                feat2positions = {}
                if self.dimlexconnectives[dc] == 'cont': ## continuous case (easy)
                    if utils.contains_sublist(tokens, list(dc)):
                        match_positions = utils.get_match_positions(tokens, list(dc))
                        if len(dc) == 1: # single token
                            for position in match_positions:
                                feat = self.getFeaturesFromTreeCont(ptree, position, dc[0])
                                feat2positions[tuple(feat)] = [position]
                        elif len(dc) > 1: # phrasal
                            for startposition in match_positions:
                                positions = list(range(startposition, startposition+len(dc)))
                                feat = self.getFeaturesFromTreeCont(ptree, list(positions), dc)
                                feat2positions[tuple(feat)] = positions

                elif self.dimlexconnectives[dc] == 'discont': ## discontinuous case (not so easy)
                    if utils.contains_discont_sublist(tokens, list(dc)):
                        match_positions = utils.get_discont_match_positions(tokens, list(dc))
                        feat = self.getFeaturesFromTreeDiscont(ptree, match_positions, dc)
                        feat2positions[tuple(feat)] = match_positions
                            
                if feat:
                    mainclass = self.mainclassdict[feat[1]] if feat[1] in self.mainclassdict else 'other'
                    catlen = self.getCategoricalLength(len(tokens))
                    sinit = 1 if self.isSInitial(dc[0], ptree) else 0

                    row = feat + [mainclass, catlen, sinit]
                    encoded = []
                    for x in row:
                        if x in self.customEncoder:
                            encoded.append(self.customEncoder[x])
                        else:
                            self.maxEncoderId += 1
                            encoded.append(self.maxEncoderId)
                            self.customEncoder[x] = self.maxEncoderId
                            self.customDecoder[self.maxEncoderId] = x
                    df = pandas.DataFrame([encoded], columns=None)
                    test_features = df.iloc[:,:]
                    test_features = numpy.array(test_features)
                    connective_prediction = self.customDecoder[self.classifier.predict(test_features)[0]]
                    if [tokens[x] for x in feat2positions[tuple(feat)]] != list(dc): # sanity check...
                        sys.stderr.write('ERROR: Indices are off. Connective is "%s" while indices point at "%s" in sentence "%s". Dying now.\n' % (list(dc), [tokens[x] for x in match_positions], sentence))
                        sys.exit()
                    if connective_prediction:
                        # with an entweder oder example, entweder...oder was discovered first, then oder was still classified as individual connective. Filter these sublist cases out here...
                        pre_existing = False
                        for existing in sdict['connectives']:
                            if utils.contains_sublist(existing, feat2positions[tuple(feat)]):
                                pre_existing = True
                        if not pre_existing:
                            sdict['connectives'].append(feat2positions[tuple(feat)])
            out.append(sdict)

        return out#json.dumps(out, ensure_ascii=False, indent=2)
                

    def train(self, parser, train_fileids=None):

        sys.stderr.write('INFO: Starting training of classifier...\n')
        if train_fileids:
            sys.stderr.write('INFO: Training LexConnClassifier with subset of files.\n')
        matrix, headers = self.getPCCFeatures(parser, train_fileids)
        df = pandas.DataFrame(matrix, columns=headers)
        y = df.class_label
        X = numpy.array(df.iloc[:,1:len(headers)-1])
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.classifier.fit(X, y)
        sys.stderr.write('INFO: Done training.\n')

    def evaluate(self, parser):

        matrix, headers = self.getPCCFeatures(parser)
        df = pandas.DataFrame(matrix, columns=headers)
        train_labels = df.class_label
        labels = list(set(train_labels)) # this just converts the labels to 0 and 1 (since my labels are binary), can also comment it out (in fact; have commented it out in the train function)
        train_labels = numpy.array([labels.index(x) for x in train_labels])
        train_features = df.iloc[:,1:len(headers)-1]
        train_features = numpy.array(train_features)
        
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.classifier.fit(train_features, train_labels)
        
        weighted_scoring = {'precision': 'precision_weighted',
                            'recall': 'recall_weighted',
                            'f': 'f1_weighted'}
        crossvalscores_weighted = cross_validate(self.classifier, train_features, train_labels, cv=10, scoring=weighted_scoring)
        
        print('Training and evaluating on PCC (10-fold cv)')
        avg_weighted_precision = numpy.mean([x for x in crossvalscores_weighted['test_precision']])
        avg_weighted_recall = numpy.mean([x for x in crossvalscores_weighted['test_recall']])
        avg_weighted_f = numpy.mean([x for x in crossvalscores_weighted['test_f']])
        print('\tavg weighted precision %f' % avg_weighted_precision)
        print('\tavg weighted recall    %f' % avg_weighted_recall)
        print('\tavg weighted f         %f' % avg_weighted_f)
    
        
        

    def getPCCFeatures(self, parser, train_fileIds=None):

        connectivefiles = utils.getInputfiles(os.path.join(parser.config['PCC']['rootfolder'], parser.config['PCC']['standoffConnectives']))
        syntaxfiles = utils.getInputfiles(os.path.join(parser.config['PCC']['rootfolder'], parser.config['PCC']['syntax']))
        fdict = defaultdict(lambda : defaultdict(str))
        fdict = utils.addAnnotationLayerToDict(connectivefiles, fdict, 'connectives')
        fdict = utils.addAnnotationLayerToDict(syntaxfiles, fdict, 'syntax') # it's using syntax layer only because this is the only layer aware of sentences (not using gold syntax trees themselves)

        # parsing corpus for stanford const trees first (using memorymap if provided)
        file2sentences = {}
        for basename in fdict:
            if train_fileIds:
                if os.path.splitext(basename)[0] in train_fileIds:
                    pccTokens, discourseRelations, tid2dt = PCCParser.parseStandoffConnectorFile(fdict[basename]['connectives'])
                    pccTokens = PCCParser.parseSyntaxFile(fdict[basename]['syntax'], pccTokens)
                    sentences = PCCParser.wrapTokensInSentences(pccTokens)
                    file2sentences[basename] = sentences
            else:
                pccTokens, discourseRelations, tid2dt = PCCParser.parseStandoffConnectorFile(fdict[basename]['connectives'])
                pccTokens = PCCParser.parseSyntaxFile(fdict[basename]['syntax'], pccTokens)
                sentences = PCCParser.wrapTokensInSentences(pccTokens)
                file2sentences[basename] = sentences
        
        if not os.path.exists(parser.config['PCC']['memorymap']):
            sys.stderr.write('WARNING: Pickled parse trees not found. Starting to parse sentences. This may take a while.\n')
            parsermemorymap = {}
            allsentences = set()
            for f in file2sentences:
                sd = file2sentences[f]
                for sid in sd:
                    sentence = ' '.join([t.token for t in sd[sid]]) # joining and splitting back and forth, a bit inefficient, but since lists are unhashable...
                    allsentences.add(sentence)
            for sent in tqdm(allsentences):
                tokens = sentence.split()
                ptree = None
                tree = parser.lexParser.parse(tokens)
                ptreeiter = ParentedTree.convert(tree)
                for t in ptreeiter:
                    ptree = t
                    break # always taking the first, assuming that this is the best scoring tree.
                parsermemorymap[sent] = ptree
            pickle.dump(parsermemorymap, codecs.open(parser.config['PCC']['memorymap'], 'wb'))
            sys.stderr.write('INFO: Parse trees pickled to "%s"\n' % parser.config['PCC']['memorymap'])
            
        parsermemorymap = pickle.load(codecs.open(parser.config['PCC']['memorymap'], 'rb'))
        matrix = []
        mid = 0
        for f in tqdm(file2sentences):
            sd = file2sentences[f]
            for sid in sd:
                sentlist = [t.token for t in sd[sid]]
                sentence = ' '.join(sentlist)
                if sentence in parsermemorymap:
                    ptree = parsermemorymap[sentence]
                    for dc in self.dimlexconnectives:
                        feat = None
                        isConnective = False
                        if self.dimlexconnectives[dc] == 'cont': ## continuous case (easy)
                            if utils.contains_sublist(sentlist, list(dc)):
                                match_positions = utils.get_match_positions(sentlist, list(dc))
                                if all([sd[sid][x].isConnective for x in match_positions]):
                                    isConnective = True
                                if len(dc) == 1: # single token
                                    for position in match_positions:
                                        feat = self.getFeaturesFromTreeCont(ptree, position, dc[0])
                                elif len(dc) > 1: # phrasal
                                    for startposition in match_positions:
                                        positions = list(range(startposition, startposition+len(dc)))
                                        feat = self.getFeaturesFromTreeCont(ptree, list(positions), dc)
                                        
                        elif self.dimlexconnectives[dc] == 'discont': ## discontinuous case (not so easy)
                            if utils.contains_discont_sublist(sentlist, list(dc)):
                                match_positions = utils.get_discont_match_positions(sentlist, list(dc))
                                if all([sd[sid][x].isConnective for x in match_positions]):
                                    isConnective = True
                                feat = self.getFeaturesFromTreeDiscont(ptree, match_positions, dc)

                        if feat:
                            mainclass = self.mainclassdict[feat[1]] if feat[1] in self.mainclassdict else 'other'
                            catlen = self.getCategoricalLength(len(sentlist))
                            sinit = 1 if self.isSInitial(dc[0], ptree) else 0
                            row = [mid] + feat + [mainclass, catlen, sinit] + [isConnective]
                            encoded = []
                            for x in row:
                                if x in self.customEncoder:
                                    encoded.append(self.customEncoder[x])
                                else:
                                    self.maxEncoderId += 1
                                    encoded.append(self.maxEncoderId)
                                    self.customEncoder[x] = self.maxEncoderId
                                    self.customDecoder[self.maxEncoderId] = x
                            matrix.append(encoded)
                            mid += 1
                            
        headers = ['id','token','pos','leftbigram','leftpos','leftposbigram','rightbigram','rightpos','rightposbigram','selfCategory','parentCategory','leftsiblingCategory','rightsiblingCategory','rightsiblingContainsVP','pathToRoot','compressedPath', 'mainclass', 'sentencelength', 'sinitial','class_label']

        return matrix, headers


    def getFeaturesFromTreeDiscont(self, ptree, positions, reftoken):

        features = []
        parentedTree = ParentedTree.convert(ptree)
        for i, node in enumerate(parentedTree.pos()):
            if i == positions[0] and node[0] == reftoken[0]:
                # getting left and right from leftmost and rightmost, ignoring things in between (think this leads to too much different feature types for classifier anyway)
                # currently this code is almost completely duplicated (only minor diffs) from the getFeaturesFromTreeCont with phrasal conns. Keeping it separate in case I want to treat discont ones more different though.
                currWord = '_'.join(reftoken)
                currPos = '_'.join([x[1] for i2, x in enumerate(parentedTree.pos()) if i2 in positions])
                features.append(currWord)
                features.append(currPos)
                ln = "SOS" if i == 0 else parentedTree.pos()[i-1]
                rn = 'EOS' if positions[-1] == len(parentedTree.pos()) else parentedTree.pos()[positions[-1]+1]
                lpos = "_" if ln == "SOS" else ln[1]
                rpos = "_" if rn == "EOS" else rn[1]
                lstr = ln if ln == "SOS" else ln[0]
                rstr = rn if rn == "EOS" else rn[0]
                lbigram = lstr + '_' + currWord
                rbigram = currWord + '_' + rstr
                lposbigram = lpos + '_' + currPos
                rposbigram = currPos + '_' + rpos
                lpos = "_" if ln == "SOS" else ln[1]
                rpos = "_" if rn == "EOS" else rn[1]
                lstr = ln if ln == "SOS" else ln[0]
                rstr = rn if rn == "EOS" else rn[0]
                lbigram = lstr + '_' + currWord
                rbigram = currWord + '_' + rstr
                lposbigram = lpos + '_' + currPos
                rposbigram = currPos + '_' + rpos
                features.append(lbigram)
                features.append(lpos)
                features.append(lposbigram)
                features.append(rbigram)
                features.append(rpos)
                features.append(rposbigram)
                parent = utils.get_parent(parentedTree, i)
                selfnode = utils.find_lowest_embracing_node_discont(parent, reftoken)
                selfcat = selfnode.label()
                features.append(selfcat)
                parentcat = 'ROOT'
                if not selfnode.label() == 'ROOT':
                    parentnode = selfnode.parent()
                    parentcat = parentnode.label()
                ls = selfnode.left_sibling()
                rs = selfnode.right_sibling()
                lsCat = False if not ls else ls.label()
                rsCat = False if not rs else rs.label() 
                features.append(parentcat)
                features.append(lsCat)
                features.append(rsCat)
                rsContainsVP = False
                if rs:
                    if list(rs.subtrees(filter=lambda x: x.label()=='VP')):
                        rsContainsVP = True
                features.append(rsContainsVP)
                rootRoute = utils.getPathToRoot(selfnode, [])
                cRoute = utils.compressRoute([x for x in rootRoute])
                features.append('_'.join(rootRoute))
                features.append('_'.join(cRoute))

        return features

    
    def getFeaturesFromTreeCont(self, ptree, position, reftoken):

        features = []
        parentedTree = ParentedTree.convert(ptree)
        if isinstance(position, int): # single token
            for i, node in enumerate(parentedTree.pos()):
                if i == position and node[0] == reftoken:
                    features.append(node[0]) # surface form/word
                    features.append(node[1]) # pos tag
                    currWord = node[0]
                    currPos = node[1]
                    ln = "SOS" if i == 0 else parentedTree.pos()[i-1]
                    rn = "EOS" if i == len(parentedTree.pos())-1 else parentedTree.pos()[i+1]
                    lpos = "_" if ln == "SOS" else ln[1]
                    rpos = "_" if rn == "EOS" else rn[1]
                    lstr = ln if ln == "SOS" else ln[0]
                    rstr = rn if rn == "EOS" else rn[0]
                    lbigram = lstr + '_' + currWord
                    rbigram = currWord + '_' + rstr
                    lposbigram = lpos + '_' + currPos
                    rposbigram = currPos + '_' + rpos
                    features.append(lbigram)
                    features.append(lpos)
                    features.append(lposbigram)
                    features.append(rbigram)
                    features.append(rpos)
                    features.append(rposbigram)
                    selfcat = currPos # always POS for single words
                    features.append(selfcat)
                    nodePosition = parentedTree.leaf_treeposition(i)
                    parent = parentedTree[nodePosition[:-1]].parent()
                    parentCategory = parent.label()
                    features.append(parentCategory)
                    ls = parent.left_sibling()
                    lsCat = False if not ls else ls.label()
                    rs = parent.right_sibling()
                    rsCat = False if not rs else rs.label()
                    features.append(lsCat)
                    features.append(rsCat)
                    rsContainsVP = False
                    if rs:
                        if list(rs.subtrees(filter=lambda x: x.label()=='VP')):
                            rsContainsVP = True
                    features.append(rsContainsVP)
                    rootRoute = utils.getPathToRoot(parent, [])
                    features.append('_'.join(rootRoute))
                    cRoute = utils.compressRoute([x for x in rootRoute])
                    features.append('_'.join(cRoute))
                    
                                        
        elif isinstance(position, list): # phrasal
            for i, node in enumerate(parentedTree.pos()):
                if i == position[0] and node[0] == reftoken[0]:
                    currWord = '_'.join([x[0] for x in parentedTree.pos()[i:i+len(reftoken)]])
                    currPos = '_'.join([x[1] for x in parentedTree.pos()[i:i+len(reftoken)]])
                    features.append(currWord)
                    features.append(currPos)
                    ln = "SOS" if i == 0 else parentedTree.pos()[i-1]
                    rn = "EOS" if i == len(parentedTree.pos())-len(reftoken)-1 else parentedTree.pos()[i+len(reftoken)]
                    lpos = "_" if ln == "SOS" else ln[1]
                    rpos = "_" if rn == "EOS" else rn[1]
                    lstr = ln if ln == "SOS" else ln[0]
                    rstr = rn if rn == "EOS" else rn[0]
                    lbigram = lstr + '_' + currWord
                    rbigram = currWord + '_' + rstr
                    lposbigram = lpos + '_' + currPos
                    rposbigram = currPos + '_' + rpos
                    lpos = "_" if ln == "SOS" else ln[1]
                    rpos = "_" if rn == "EOS" else rn[1]
                    lstr = ln if ln == "SOS" else ln[0]
                    rstr = rn if rn == "EOS" else rn[0]
                    lbigram = lstr + '_' + currWord
                    rbigram = currWord + '_' + rstr
                    lposbigram = lpos + '_' + currPos
                    rposbigram = currPos + '_' + rpos
                    features.append(lbigram)
                    features.append(lpos)
                    features.append(lposbigram)
                    features.append(rbigram)
                    features.append(rpos)
                    features.append(rposbigram)
                    parent = utils.get_parent(parentedTree, i)
                    selfnode = utils.find_lowest_embracing_node(parent, reftoken)
                    selfcat = selfnode.label() # cat of lowest level node containing all tokens of connective
                    features.append(selfcat)
                    parentcat = 'ROOT'
                    if not selfnode.label() == 'ROOT':
                        parentnode = selfnode.parent()
                        parentcat = parentnode.label()
                    ls = selfnode.left_sibling()
                    rs = selfnode.right_sibling()
                    lsCat = False if not ls else ls.label()
                    rsCat = False if not rs else rs.label() 
                    features.append(parentcat)
                    features.append(lsCat)
                    features.append(rsCat)
                    rsContainsVP = False
                    if rs:
                        if list(rs.subtrees(filter=lambda x: x.label()=='VP')):
                            rsContainsVP = True
                    features.append(rsContainsVP)
                    rootRoute = utils.getPathToRoot(selfnode, [])
                    cRoute = utils.compressRoute([x for x in rootRoute])
                    features.append('_'.join(rootRoute))
                    features.append('_'.join(cRoute))
                    
        return features
