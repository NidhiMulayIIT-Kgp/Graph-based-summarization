''' ASSIGNMENT 3 : CONTENT SUMMARIZATION
    Submitted by: Archie Mittal(17CS60R82)
                  Nidhi Mulay(17CS60R75)'''

import matplotlib.pyplot as plt
import networkx as nx
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import codecs
from nltk.corpus import stopwords
import nltk.data
import string
from rouge import Rouge
import sys


'''
    * Function Name: calculatebckwrdPR
    * Input: sentences, bckwrdVariant
            bckwrdVariant -- backward variant of the directed graph
    * Output: bckwrdPR
            bckwrdPR -- stores the list returned by the function calculatePR that stores page rank of all the sentences
    * Logic : reverses the sentences and calculates page rank for the graph using formula
'''
def calculatebckwrdPR(sentences, bckwrdVariant):

    #It arranges the sentences in reverse order
    sentences.reverse()

    #'bckwrdPR' stores the list returned by the function calculatePR that stores page rank of all the sentences
    bckwrdPR = calculatePR(sentences, bckwrdVariant);
    bckwrdPR.reverse();
    return bckwrdPR;

'''
    * Function Name: calculatefrwdPR
    * Input: sentences, frwdVariant
            frwdVariant -- forward variant of the directed graph 
    * Output: frwdPR
            frwdPR -- stores the list returned by the function calculatePR that stores page rank of all the sentences
    * Logic : calculates page rank for the graph using formula
'''    
def calculatefrwdPR(sentences, frwdVariant): 
    frwdPR = calculatePR(sentences, frwdVariant);
    return frwdPR;

'''
    * Function Name: calculatePR
    * Input: sentences, graph
            graph -- the directed graph having sentences as nodes
    * Output: PRlist
            PRList -- the list storing the page rank of the sentences
    * Logic : calculates page rank for the graph using formula
'''   
def calculatePR(sentences, graph):
    PRlist = [];

    #'d' represents the probability that an imaginary surfer who is randomly clicking on links will continue called as damping factor
    d = 0.85;
 
    #calculates the page rank using the formula given in paper by Rada Mihalcea
    for sentence in sentences:
        inSum = 0.0;
##        print "sentence = ", sentence
        #'incomingNodes' stores the incoming edges of all nodes in the graph
        incomingNodes = getIncomingNodes(sentence, graph);
##        print "incoming nodes = ", incomingNodes
        #'outgoingNodes' stores the outgoing edges of all nodes in the graph
        for node in incomingNodes:
            outgoingNodes = getOutgoingNodes(node[0], graph);
##            print "outGoing nodes = ", outgoingNodes
            outSum = 0.0;
            for outNode in outgoingNodes:
##                print outNode
                outSum = outSum + getWeight(outNode);
##                print "getWeight = ", getWeight(outNode)
##
##                print "out sum = ",outSum
            if outSum != 0:
                #outSum = 0.0000001;
                inSum = (inSum + (float(getWeight(node)) * float(PRlist[sentences.index(node[0])]))/float(outSum));
##        print "inSum =", inSum
        PR = float(1 - d) + (d * inSum);
##        print "PR + ", PR
        PRlist.append(PR);
        PR = 0.0;
        
    
##        print
##    print "PRList = ", PRlist
    return PRlist;

def calculateRougeScore(hypothesis, rfrncSummry):
    rouge = Rouge()
    score = rouge.get_scores(rfrncSummry, hypothesis)
    print score[0]['rouge-2']
    
    
'''
    * Function Name: constructBckwrdVariant
    * Input: sentences
    * Output: G
            G -- graph having sentences as nodes 
    * Logic : process sentences to form graph having nodes as sentences directed edges from the later sentences to the previous sentences and weights according to the degree of similarity between the sentences
'''  
def constructBckwrdVariant(sentences):
    G = nx.DiGraph()
    for i in range(len(sentences)):
        G.add_node(sentences[i])
        for j in range (i + 1, len(sentences)):
            G.add_edge(sentences[j], sentences[i], weight= findSimilarity(sentences[j],sentences[i]))
    return G;

'''
    * Function Name: constructDigraph
    * Input: sentences
    * Output: frwrdVariant, bckwrdVariant
            frwrdVariant -- forward variant of graph
            bckwrdVariant -- backward variant of graph
    * Logic : constructs both the variants of the graph by calling the functions constructBckwrdVariant() and contructFrwrdDigraph()
'''  
def constructDigraph(sentences):
    frwrdVariant = contructFrwrdDigraph(sentences);
    bckwrdVariant = constructBckwrdVariant(sentences);
    return frwrdVariant, bckwrdVariant;

'''
    * Function Name: contructFrwrdDigraph
    * Input: sentences
    * Output: G
            G -- graph having sentences as nodes 
    * Logic : This function takes sentences as input and constructs a graph having nodes as sentences, directed edges from the earlier sentences to the later sentences and weights according to the degree of similarity between the sentences and returns that graph
'''  
def contructFrwrdDigraph(sentences):
    G = nx.DiGraph()
    for i in range(len(sentences)):
        G.add_node(sentences[i])
        for j in range(i+1,len(sentences)):
            G.add_edge(sentences[i],sentences[j],weight= findSimilarity(sentences[i],sentences[j]));
    nx.draw_networkx(G)
    return G;

#This function identifies different sentences from the input text and returns them 
def detectSentences(text):
    sentences = [];
    sentences = sent_tokenize(text);
    return sentences

#This function takes sentence and graph 'G' as input and returns the set of incoming edges to that sentence node
def getIncomingNodes(sentence, G):
    return G.in_edges(sentence, data = True);

#This function takes sentence and graph 'G' as input and returns the set of outgoing edges to that sentence node
def getOutgoingNodes(sentence, G):
    return G.out_edges(sentence, data = True);

#This function takes edge as an input and returns its weight
def getWeight(edge):
    return edge[2]['weight'];

'''
    * Function Name: findSimilarity
    * Input: sentence1, sentence2
    * Output: similarity
            similarity -- score showing similarity between two sentences 
    * Logic : This function takes 2 sentences as input and calculates similarity between them by counting the common words
''' 
def findSimilarity(sentence1, sentence2):
    list1 = sentence1.split();
    list2 = sentence2.split();
    wk = (set(list1).intersection(set(list2)))
    si = len(list1)
    sj = len(list2)
    similarity = (float(len(wk))/float(si*sj))
    return similarity;

'''
    * Function Name: performStemming
    * Input: sentences
    * Output: modifiedSentences
            modifiedSentences -- sentences after stemming
    * Logic :performs stemming on sentences using the PorterStemmer() and returns the modified sentencesThis function takes 2 sentences as input and calculates similarity between them by counting the common words
''' 
def performStemming(sentences):
    modifiedSentences = [];
    currentSentence ="";
    ps = PorterStemmer()
    for sentence in sentences:
        
        for word in word_tokenize(sentence.lower()):
            currentSentence = currentSentence + ps.stem(word) + " ";
            
        modifiedSentences.append(currentSentence);
        currentSentence = "";
    return modifiedSentences

'''
    * Function Name: preprocessSentences
    * Input: sentences
    * Output: modifiedSentences
            modifiedSentences -- sentences after stemming and stop word removal
    * Logic : This function removes stop words and performs stemming on sentences using the removeStopWords() and performStemming() returns the modified sentences
'''    
def preprocessSentences(sentences):
    stopWordFreeSentences = removeStopWords(sentences);
    stemmedSentences = performStemming(stopWordFreeSentences);
    return stemmedSentences

'''
    * Function Name: getBckwrdSummary
    * Input: bckwrdPR, sentences
             bckwrdPR -- list of page ranks of sentences in backward variant of graph
    * Output: sentences[i]
            sentences[i] -- summary of 'n' lines
    * Logic : sorts sentences in descending order of their page ranks and selects displays top 'n' lines as summary
'''
def getBckwrdSummary(bckwrdPR, sentences, m):
    indices = [];
    summary = "";
    for i in range(m):
        pr = max(bckwrdPR);
        index = bckwrdPR.index(pr);
        indices.append(index);
        bckwrdPR[index] = -1;
    indices.sort();
    for index in indices:
        summary = summary + sentences[index];
    return summary
    
'''
    * Function Name: printFrwdSummary
    * Input: frwdPR, sentences
             frwdPR -- list of page ranks of sentences in backward variant of graph
    * Output: sentences[i]
            sentences[i] -- summary of 'n' lines
    * Logic : sorts sentences in descending order of their page ranks and selects displays top 'n' lines as summary
'''    
def getFrwdSummary(frwdPR, sentences, m):
    indices = [];
    summary = "";
    for i in range(m):
        pr = max(frwdPR);
        index = frwdPR.index(pr);
        indices.append(index);
        frwdPR[index] = -1;
    indices.sort();
    for index in indices:
        summary = summary + sentences[index];
    return summary

'''
    * Function Name: readFile
    * Input: fileInput
             fileInput -- text file
    * Output: text
            text -- text file as it is
    * Logic : This function takes a text file 'fileInput' as input and reads the whole text and returns it
'''        
def readFile(fileInput):
    text = fileInput.read();
    return text

'''
    * Function Name: removeEscapeSequences
    * Input: sentences
    * Output: modifiedSentences
            modifiedSentences -- sentences after removing escape sequences
    * Logic : This function removes escape sequences fron the sentences and returns the modified sentences
'''
def removeEscapeSequences(sentences):
    modifiedSentences = [];
    previousLetter = ' ';
    for sentence in sentences:
        currentSentence = ""
        for letter in sentence:
            if(letter == ' ' and previousLetter == ' '):
                continue
            if(letter != '\n' and letter != '\t'):
                currentSentence = currentSentence + letter;
                previousLetter = letter;
        modifiedSentences.append(currentSentence)
    return modifiedSentences

'''
    * Function Name: removePunctuations
    * Input: sentences
    * Output: modifiedSentences
            modifiedSentences -- sentences after removing punctuations
    * Logic : This function removes punctuations fron the sentences and returns the modified sentences
'''
def removePunctuations(sentences):
    modifiedSentences = [];
    for sentence in sentences:
        currentSentence = "".join([ch for ch in sentence if ch not in string.punctuation])
        modifiedSentences.append(currentSentence);
    return modifiedSentences

'''
    * Function Name: removeStopWords
    * Input: sentences
    * Output: modifiedSentences
            modifiedSentences -- sentences after removing stop words
    * Logic : This function removes stop words using nltk tools fron the sentences and returns the modified sentences
'''   
def removeStopWords(sentences):
    modifiedSentences = [];
    currentSentence ="";
    stopWords = set(stopwords.words('english'))
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word not in stopWords:
                currentSentence = currentSentence + word + " ";
        modifiedSentences.append(currentSentence);
        currentSentence = "";
    return modifiedSentences
        
if __name__ == "__main__":
##    fileName = raw_input("Enter file name: ")
##    m = int(raw_input("Enter no of lines of summary required: "))
##    refFile = raw_input("Enter reference summary file name: ")
    fileName = sys.argv[1];
    m = int(sys.argv[2]);
    refFile = sys.argv[3];
    refFileInput = open(refFile)
    fileInput = codecs.open(fileName);
    text = readFile(fileInput)
    text=text.decode('ascii', 'ignore')
    rfrncSummry = readFile(refFileInput)
    sentences = detectSentences(text);
    sentences = removeEscapeSequences(sentences);
    unpunctuatedSentences = removePunctuations(sentences);
    preProcessedSentences = preprocessSentences(unpunctuatedSentences);
    frwdVariant, bckwrdVariant = constructDigraph(preProcessedSentences);
    frwdPR = calculatefrwdPR(preProcessedSentences, frwdVariant);
    print "Forward Summary -->"
    frwdSummry = getFrwdSummary(frwdPR, sentences, m);
    print frwdSummry
    calculateRougeScore(frwdSummry, rfrncSummry);
    bckwrdPR = calculatebckwrdPR(preProcessedSentences, bckwrdVariant);
    print
    print "Backward Summary-->"
    bckwrdSummry = getBckwrdSummary(bckwrdPR, sentences, m);
    print bckwrdSummry
    calculateRougeScore(bckwrdSummry, rfrncSummry);

    
