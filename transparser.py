import networkx as nx

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


def predictWeightedGraph(graph):
    for i,j in graph.edges_iter():
        # apply averaged perceptron algorithm for multiclass classification of the transition 'r' 'l' 's'. remove the feature 'transition' from the edge
        weights.dotProduct(graph[i][j])


def runOneExample(weights, TrueGraph):
    G = convertToTransitions(TrueGraph)

    predGraph = predictWeightedGraph(G)



def convertToTransitions(InputGraph):
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
            stacksDepG.pop[0]

        # if the top of the stack is the head of the first element of the buffer
        elif inputGraph.node[bufferDepG[0]]['head'] == stacksDepG[0]:
            transition = 'r'
            # r is for right transition
            bufferDepG[0] = stacksDepG[0]
            stacksDepG.pop[0]

        else:
            transition = 's'
            # s is for shift transition
            stacksDepG.insert[0,bufferDepG[0]]
            bufferDepG.pop[0]

        f = inputGraph.node[stackTop] # get node information for i (eg {word: blah, pos: blah})
        g = inputGraph.node[bufferTop] # get node information for j

        feats = {   'stack_top=' + stackTop: 1.,
                    'buffer_top' + bufferTop: 1.,
                    'cpos_stack_top=' + f['cpos']: 1.,
                    'cpos_buffer_head=' + g['cpos']: 1.,
                    'w_pair=' + f['word'] + '_' + g['word']: 1.,
                    'cp_pair=' + f['cpos'] + '_' + g['cpos']: 1.,
                    'transition='+transition: 1.}
            # TODO is graph output the correct form of output

        out.add_edge(stackTop, bufferTop, feats)
                      
    return out


def iterCoNLL(filename):
    h = open(filename, 'r')
    G = None
    nn = 0
    for l in h:
        l = l.strip()
        if l == "":
            if G != None:
                yield G
            G = None
        else:
            if G == None:
                nn = nn + 1
                G = nx.Graph()
                G.add_node(0, {'word': '*root*', 'lemma': '*root*', 'cpos': '*root*', 'pos': '*root*', 'feats': '*root*'})
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




weights = Weights()
print weights
for iteration in range(5):
         totalErr = 0.
         for G in iterCoNLL(inputTestSet): totalErr += runOneExample(weights, G, quiet=True)
         print totalErr