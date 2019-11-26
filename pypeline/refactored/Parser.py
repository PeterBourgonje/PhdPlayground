#!/usr/bin/python3
import LexConnClassifier
import PCCParser
import configparser
import sys
import time
import codecs
import json
import os
from nltk.parse import stanford


class Parser:

    def __init__(self):

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        os.environ['JAVAHOME'] = self.config['lexparser']['javahome']
        os.environ['STANFORD_PARSER'] = self.config['lexparser']['stanfordParser']
        os.environ['STANFORD_MODELS'] = self.config['lexparser']['stanfordModels']
        os.environ['CLASSPATH'] = self.config['lexparser']['path']
        self.lexParser = stanford.StanfordParser(model_path=self.config['lexparser']['germanModel'])


    def preParse(self, sentences):
        runtimeparsermemory = {}
        for sentence in sentences:
            tokens = sentence.split()
            ptree = None
            tree = self.lexParser.parse(tokens)
            ptreeiter = ParentedTree.convert(tree)
            for t in ptreeiter:
                ptree = t
                break # always taking the first, assuming that this is the best scoring tree.
            runtimeparsermemory[sentence] = ptree
            
        return runtimeparsermemory

                              
    def getSentencesFromPCC(self):

        connectivefiles = utils.getInputfiles(os.path.join(self.config['PCC']['rootfolder'], self.config['PCC']['standoffConnectives']))
        syntaxfiles = utils.getInputfiles(os.path.join(self.config['PCC']['rootfolder'], self.config['PCC']['syntax']))
        rstfiles = utils.getInputfiles(os.path.join(self.config['PCC']['rootfolder'], self.config['PCC']['rst']))
        tokenfiles = utils.getInputfiles(os.path.join(self.config['PCC']['rootfolder'], self.config['PCC']['tokens']))
        fdict = defaultdict(lambda : defaultdict(str))
        fdict = utils.addAnnotationLayerToDict(connectivefiles, fdict, 'connectors')
        fdict = utils.addAnnotationLayerToDict(syntaxfiles, fdict, 'syntax') # not using the gold syntax, but this layer is needed to extract full sentences, as it's (I think) the only layer that knows about this.

        file2sentences = {}
        for basename in fdict:
            pccTokens, discourseRelations, tid2dt = MinimalPCCParser.parseStandoffConnectorFile(fdict[basename]['connectors'])
            pccTokens = MinimalPCCParser.parseSyntaxFile(fdict[basename]['syntax'], pccTokens)
            sentences = MinimalPCCParser.wrapTokensInSentences(pccTokens)
            file2sentences[basename] = sentences

        return file2sentences

if __name__ == '__main__':

    p = Parser()
    lcf = LexConnClassifier.LexConnClassifier(p)
    lcf.evaluate(p)
