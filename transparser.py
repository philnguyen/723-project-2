import networkx as nx
import sys

for arg in sys.argv:
    inputTrainSet = sys.argv[1]
    inputTestSet = sys.argv[2]
    outputTestSet = sys.argv[3]

#TODO
# create a random weight matrix
#iterate over all sentences in the training set
    # parse a sentence into a buffer
    # determine transitions for a particular sentence
    # train a classifier on this training data
    # learn bpnn model for transitions. hmm how would you predict transitions. you can predict the head number or word for a particular word but how do you predict transitions
    # learn what is the head of one word and if it is the first word in the buffer then do a right transition
class Weights(dict):
    # default all unknown feature values to zero
    def __getitem__(self, idx):
        if self.has_key(idx):
            return dict.__getitem__(self, idx)
        else:
            return 0.

    # given a feature vector, compute a dot product
    def dotProduct(self, x):
        dot = 0.
        for feat,val in x.iteritems():
            dot += val * self[feat]
        return dot

    # given an example _and_ a true label (y is +1 or -1), update the
    # weights according to the perceptron update rule (we assume
    # you've already checked that the classification is incorrect
    def update(self, x, y):
        for feat,val in x.iteritems():
            if val != 0.:
                self[feat] += y * val

def predictWeightedGraph(graph):
    for i,j in graph.edges_iter():
        # apply averaged perceptron algorithm for multiclass classification of the transition 'r' 'l' 's'. remove the feature 'transition' from the edge
        features = graph[i][j]
        #print graph[i][j]
        trueTransition = features['transition']
        #print trueTransition
        del features['transition']

        # to remove the transition state from the features
        yR = weightsR.dotProduct(features)
        #print yR
        yL = weightsL.dotProduct(features)
        #print yL
        yS = weightsS.dotProduct(features)
        #print yS
        #print '/n'
        predTransition = ''
        predTransitionVal = max(yR,yL,yS)
        if predTransitionVal == yR:
            predTransition = 'r'
        elif predTransitionVal == yL:
            predTransition = 'l'
        else: predTransition = 's'
        #if predTransition!='s': print predTransition
        graph[i][j]['predTransition'] = ''
        graph[i][j]['predTransition'] = predTransition
        graph[i][j]['transition'] = trueTransition
        #print graph[i][j]
    return graph

def numMistakes(trueGraph, predGraph):
    # remove trueGraph as it is not required
    err = 0.
    for i,j in predGraph.edges_iter():
        #print predGraph[i][j]
        #print trueGraph[i][j]
        if predGraph[i][j]['predTransition'] != predGraph[i][j]['transition']:
            err += 1
            print err
    return err

def perceptronUpdate(weightsR, weightsL, weightsS, predGraph):
    for i,j in predGraph.edges_iter():
        #print predGraph[i][j]
        predTransition = predGraph[i][j]['predTransition']
        transition = predGraph[i][j]['transition']
        
        del predGraph[i][j]['predTransition']
        del predGraph[i][j]['transition']

        if  predTransition == 'r':
            weightsR.update(predGraph[i][j], -1)
        if  predTransition == 'l':
            weightsL.update(predGraph[i][j], -1)
        if  predTransition == 's':
            weightsS.update(predGraph[i][j], -1)
        if  transition == 'r':
            weightsR.update(predGraph[i][j], 1)
        if  transition == 'l':
            weightsL.update(predGraph[i][j], 1)
        if  transition == 's':
            weightsS.update(predGraph[i][j], 1)
        #print weightsS
        #print weightsL
        #print weightsR

        predGraph[i][j]['predTransition'] = predTransition
        predGraph[i][j]['transition'] = transition





def runOneExample(weightsR, weightsL, weightsS, trueGraph):
    G = convertToTransitions(trueGraph)

    predGraph = predictWeightedGraph(G)

    err = numMistakes(G,predGraph)

    if err > 0:
        perceptronUpdate(weightsR, weightsL, weightsS, predGraph)

    return err


def convertToTransitions(inputGraph):
    # create a new graph to return
    out = nx.Graph()
    bufferDepG = []
    stacksDepG = [0] # 0 corresponds to 'root'
    for i in inputGraph.nodes():
        bufferDepG.append(i+1) # because i starts from 0 which is 'root'

    stackTop = stacksDepG[0]
    bufferTop = bufferDepG[0]
    # TODO implement buffer and stack

    while len(bufferDepG):
        # if head of the top of the stack is the top of the buffer
        if inputGraph.node[stacksDepG[0]]['head'] == bufferTop:
            transition = 'l'
            # l is for left transition
            stacksDepG.pop()

        # if the top of the stack is the head of the first element of the buffer
        elif inputGraph.node[bufferDepG[0]]['head'] == stacksDepG[0]:
            transition = 'r'
            # r is for right transition
            bufferDepG[0] = stacksDepG[0]
            stacksDepG.pop()

        else:
            transition = 's'
            # s is for shift transition
            stacksDepG.insert(0,bufferDepG[0])
            bufferDepG.pop()

        f = inputGraph.node[stackTop] # get node information for i (eg {word: blah, pos: blah})
        g = inputGraph.node[bufferTop] # get node information for j

        feats = {   'stack_top=' + f['word']: 1.,
                    'buffer_top' + g['word'] : 1.,
                    'cpos_stack_top=' + f['cpos']: 1.,
                    'cpos_buffer_head=' + g['cpos']: 1.,
                    'w_pair=' + f['word'] + '_' + g['word']: 1.,
                    'cp_pair=' + f['cpos'] + '_' + g['cpos']: 1.,
                    'transition': transition}
            # TODO is graph output the correct form of output

        out.add_edge(stackTop, bufferTop, feats)
        #print out             
    return out


def iterCoNLL(filename):
    h = open(filename, 'r')
    G = None
    nn = 0
    for l in h:
        #print l
        l = l.strip()
        if l == "":
            if G != None:
                yield G
            G = None
        else:
            if G == None:
                nn = nn + 1
                G = nx.Graph()
                G.add_node(0, {'word': '*root*', 'lemma': '*root*', 'cpos': '*root*', 'pos': '*root*', 'feats': '*root*', 'head' : '*root*'})
                newGraph = False
            [id, word, lemma, cpos, pos, feats, head, drel, phead, pdrel] = l.split('\t')
            G.add_node(int(id), {'word' : word,
                                 'lemma': lemma,
                                 'cpos' : cpos,
                                 'pos'  : pos,
                                 'feats': feats,
                                 'head' : head})
            
            G.add_edge(int(head), int(id), {}) # 'true_rel': drel, 'true_par': int(id)})

    if G != None:
        yield G
    h.close()


weightsR = Weights()
weightsL = Weights()
weightsS = Weights()

for iteration in range(5):
         totalErr = 0.
         for G in iterCoNLL(inputTrainSet): 
            totalErr += runOneExample(weightsR, weightsL, weightsS, G)
         print totalErr