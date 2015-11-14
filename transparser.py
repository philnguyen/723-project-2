import networkx as nx
import sys

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
                
def predTransitionGraph(graph):
    for i,j in graph.edges_iter():
        # apply averaged perceptron algorithm for multiclass classification of the transition 'r' 'l' 's'. remove the feature 'transition' from the edge
        features = graph[i][j]
        #print graph[i][j]
        trueTransition = features['transition']
        #print trueTransition
        del features['transition']

        # to remove the transition state from the features
        yR = weightsR.dotProduct(features) + bias[0]
        #print yR
        yL = weightsL.dotProduct(features) + bias[1]
        #print yL
        yS = weightsS.dotProduct(features) + bias[2]
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
        #print predTransition
        #print trueTransition
        #print '/n'
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
            #print err
    return err

#def avgPerceptronUpdate(weightsR, weightsL, weightsS, predGraph):
def avgPerceptronUpdate(predGraph):
    for i,j in predGraph.edges_iter():
        #TODO make this averaged perceptron
        #TODO update after every sentence vs update after every word
        #print predGraph[i][j]
        predTransition = predGraph[i][j]['predTransition']
        transition = predGraph[i][j]['transition']
        
        del predGraph[i][j]['predTransition']
        del predGraph[i][j]['transition']

        if  predTransition == 'r':
            weightsR.update(predGraph[i][j], -1)
            cachedWeightsR.update(predGraph[i][j], -1*avgCounter)
            bias[0] += -1
            cachedBias[0] += -1*avgCounter
        if  predTransition == 'l':
            weightsL.update(predGraph[i][j], -1)
            cachedWeightsL.update(predGraph[i][j], -1*avgCounter)
            bias[1] += -1
            cachedBias[1] += -1*avgCounter
        if  predTransition == 's':
            weightsS.update(predGraph[i][j], -1)
            cachedWeightsS.update(predGraph[i][j], -1*avgCounter)
            bias[2] += -1
            cachedBias[2] += -1*avgCounter
        if  transition == 'r':
            weightsR.update(predGraph[i][j], 1)
            cachedWeightsR.update(predGraph[i][j], 1*avgCounter)
            bias[0] += 1
            cachedBias[0] += 1*avgCounter
        if  transition == 'l':
            weightsL.update(predGraph[i][j], 1)
            cachedWeightsL.update(predGraph[i][j], 1*avgCounter)
            bias[1] += 1
            cachedBias[1] += 1*avgCounter
        if  transition == 's':
            weightsS.update(predGraph[i][j], 1)
            cachedWeightsS.update(predGraph[i][j], 1*avgCounter)
            bias[2] += 1
            cachedBias[2] += 1*avgCounter
        #print weightsS
        #print weightsL
        #print weightsR

        predGraph[i][j]['predTransition'] = predTransition
        predGraph[i][j]['transition'] = transition

def trainOracle(inputGraph):
    # create a new graph to return
    out = nx.Graph()
    bufferDepG = []
    stacksDepG = [0] # 0 corresponds to 'root'
    #print inputGraph
    for i in inputGraph.nodes():
        bufferDepG.append(i) 
    bufferDepG.pop(0) # because i starts from 0 which is 'root'

    while len(bufferDepG)!=0:
        # if head of the top of the stack is the top of the buffer
        #print inputGraph.node[stacksDepG[0]]['head']

        f = inputGraph.node[stacksDepG[0]] # get node information for i (eg {word: blah, pos: blah})
        #print bufferDepG
        #print stacksDepG
        if len(bufferDepG)!=0: g = inputGraph.node[bufferDepG[0]] # get node information for j
        stackTop = stacksDepG[0]
        bufferTop = bufferDepG[0]

        #print inputGraph.node[stacksDepG[0]]['head']
        #print bufferDepG[0]

        if inputGraph.node[stacksDepG[0]]['head'] == str(bufferDepG[0]):
            flag = True
            bufferDepG.pop(0) # this is added again afterwards
            # left arc precondition
            for i in bufferDepG:
                #print i
                #print inputGraph.node[i]
                if inputGraph.node[i]['head'] == stacksDepG[0]:
                    flag = False
                    continue
            bufferDepG.insert(0,bufferTop)
            if inputGraph.node[stacksDepG[0]]!= str(0) and flag:
                transition = 'l'
                # l is for left transition
                stacksDepG.pop(0)
                #print inputGraph.node[stacksDepG[0]]['head']
                #print bufferDepG[0]

                #print 'left'
            #print 1

        # if the top of the stack is the head of the first element of the buffer
        elif inputGraph.node[bufferDepG[0]]['head'] == str(stacksDepG[0]) and stacksDepG[0] != 0:
            transition = 'r'
            # r is for right transition
            bufferDepG.pop(0)
            bufferDepG.insert(0,stacksDepG[0])
            stacksDepG.pop(0)
            #print inputGraph.node[bufferDepG[0]]['head']
            #print stacksDepG[0]
            #print 'right'
            #print 2

        else:
            transition = 's'
            # s is for shift transition
            stacksDepG.insert(0,bufferDepG[0])
            bufferDepG.pop(0)
            #if len(bufferDepG)!=0: print bufferDepG[0]
        #print transition
        feats = {   'stack_top=' + f['word']: 1.,
                    'buffer_top=' + g['word'] : 1.,
                    'cpos_stack_top=' + f['cpos']: 1.,
                    'cpos_buffer_head=' + g['cpos']: 1.,
                    'w_pair=' + f['word'] + '_' + g['word']: 1.,
                    'cp_pair=' + f['cpos'] + '_' + g['cpos']: 1.,
                    'transition': transition}
            # TODO is graph output the correct form of output

        out.add_edge(stackTop, bufferTop , feats)
        #print out             
    return out

def runOneExample(weightsR, weightsL, weightsS, trueGraph):
    G = trainOracle(trueGraph)

    predGraph = predTransitionGraph(G)

    err = numMistakes(G,predGraph)

    if err > 0:
        #avgPerceptronUpdate(weightsR, weightsL, weightsS, predGraph)
        avgPerceptronUpdate(predGraph)

    global avgCounter
    avgCounter+=1

    return err

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
                G.add_node(0, {'word': '*root*', 'lemma': '*root*', 'cpos': '*root*', 'pos': '*root*', 'feats': '*root*', 'head' : ''})
                newGraph = False
            [id, word, lemma, cpos, pos, feats, head, drel, phead, pdrel] = l.split('\t')
            G.add_node(int(id), {'word' : word,
                                 'lemma': lemma,
                                 'cpos' : cpos,
                                 'pos'  : pos,
                                 'feats': feats,
                                 'head' : head})
            
            #G.add_edge(int(head), int(id), {}) # 'true_rel': drel, 'true_par': int(id)})

    if G != None:
        yield G
    h.close()


weightsR = Weights()
weightsL = Weights()
weightsS = Weights()
bias = [0.,0.,0.] #[biasR, biasL, biasS]
cachedWeightsR = Weights()
cachedWeightsL = Weights()
cachedWeightsS = Weights()
cachedBias = [0.,0.,0.]
avgCounter = 1.0

for arg in sys.argv:
    inputTrainSet = sys.argv[1]
    inputTestSet = sys.argv[2]
    outputTestSet = sys.argv[3]

for iteration in range(5):
    totalErr = 0.
    for G in iterCoNLL(inputTrainSet): 
        totalErr += runOneExample(weightsR, weightsL, weightsS, G)

    print totalErr

weightsR.update(cachedWeightsR, -1.0/avgCounter)
weightsL.update(cachedWeightsL, -1.0/avgCounter)
weightsS.update(cachedWeightsS, -1.0/avgCounter)
bias[0] -= cachedBias[0]/avgCounter
bias[1] -= cachedBias[1]/avgCounter
bias[2] -= cachedBias[2]/avgCounter


# read in test filename
for G in iterCoNLL(inputTestSet):
    #output = predicttheHeads(weightsR, weightsL, weightsS, G)
    writetheheads(output, inputTestSet)
    outputTestSet

# for end of file:
    # for each sentence
        # fill the buffer and stack
        # extract features
        # multiply with weights
        # find the maximum weight and determine the transition
        # assign the head
        # make the transition and update the stack and the buffer
    # udpate the output file with the correct heads - copy the input file to another and update that file with the heads

def predictTheHeads(G):
    out = nx.Graph()
    bufferDepG = []
    stacksDepG = [0] # 0 corresponds to 'root'
    #print inputGraph
    for i in inputGraph.nodes():
        bufferDepG.append(i) 
    bufferDepG.pop(0) # because i starts from 0 which is 'root'
    
    while len(bufferDepG)!=0:
        # if head of the top of the stack is the top of the buffer

        f = inputGraph.node[stacksDepG[0]] # get node information for i (eg {word: blah, pos: blah})
        #print bufferDepG
        #print stacksDepG
        if len(bufferDepG)!=0: 
            g = inputGraph.node[bufferDepG[0]] # get node information for j
        
        stackTop = stacksDepG[0]
        bufferTop = bufferDepG[0]

        feats = {   'stack_top=' + f['word']: 1.,
                    'buffer_top=' + g['word'] : 1.,
                    'cpos_stack_top=' + f['cpos']: 1.,
                    'cpos_buffer_head=' + g['cpos']: 1.,
                    'w_pair=' + f['word'] + '_' + g['word']: 1.,
                    'cp_pair=' + f['cpos'] + '_' + g['cpos']: 1.}
                    #'transition': transition}
            # TODO is graph output the correct form of output


        out.add_edge(stackTop, bufferTop , feats)

        #print inputGraph.node[stacksDepG[0]]['head']
        #print bufferDepG[0]


        # left transition
        if inputGraph.node[stacksDepG[0]]['head'] == str(bufferDepG[0]):
            flag = True
            bufferDepG.pop(0) # this is added again afterwards
            # left arc precondition
            for i in bufferDepG:
                #print i
                #print inputGraph.node[i]
                if inputGraph.node[i]['head'] == stacksDepG[0]:
                    flag = False
                    continue
            bufferDepG.insert(0,bufferTop)
            if inputGraph.node[stacksDepG[0]]!= str(0) and flag:
                transition = 'l'
                # l is for left transition
                stacksDepG.pop(0)
                #print inputGraph.node[stacksDepG[0]]['head']
                #print bufferDepG[0]

                #print 'left'
            #print 1
        # if the top of the stack is the head of the first element of the buffer

        # right transition
        elif inputGraph.node[bufferDepG[0]]['head'] == str(stacksDepG[0]) and stacksDepG[0] != 0:
            transition = 'r'
            # r is for right transition
            bufferDepG.pop(0)
            bufferDepG.insert(0,stacksDepG[0])
            stacksDepG.pop(0)
            #print inputGraph.node[bufferDepG[0]]['head']
            #print stacksDepG[0]
            #print 'right'
            #print 2

        # shift transition
        else:
            transition = 's'
            # s is for shift transition
            stacksDepG.insert(0,bufferDepG[0])
            bufferDepG.pop(0)
            #if len(bufferDepG)!=0: print bufferDepG[0]
        #print transition


    
        #print out             
    return out
