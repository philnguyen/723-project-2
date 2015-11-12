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

x = iterCoNLL(inputTrainSet)
convertToTransitions(x) 

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

def convertToTransitions(InputGraph):
    # create a new graph to return
    out = nx.Graph()
    bufferDep = []
    stacksDep = ['*root*']
    for i in inputGraph.nodes():
        bufferDep.append(inputGraph.node[i]['id'])
    stackTop = stacksDep[0]
    bufferTop = bufferDep[0]
    # TODO implement buffer and stack

    while bufferDep!=empty:
        if inputGraph.node[stacksDep[0]]['head'] == bufferDep[0]:
            transition = 'r'
            # r is for right transition
            bufferDep[0] = stacksDep[0]
            stacksDep.pop

        elif inputGraph.node[bufferDep[0]]['head'] == stacksDep[0]:
            transition = 'l'
            # l is for left transition
            stacksDep.pop
        else:
            transition = 's'
            # s is for shift transition
            stacksDep.push[bufferDep[0]]
            bufferDep.pop

    # for each pair of words (nodes) in the input graph, create an
    # edge in the output graph, on which we write some features
    #for i in inputGraph.nodes():
    #    for j in inputGraph.nodes():
    #        if j <= i: continue    # we're undirected, so skip half the edges

            f = inputGraph.node[stackTop] # get node information for i (eg {word: blah, pos: blah})
            g = inputGraph.node[bufferTop] # get node information for j

            feats = {   'stack_top=' + stackTop: 1.,
                        'buffer_top' + bufferTop: 1.,
                        'cpos_stack_top=' + f['cpos']: 1.,
                        'cpos_buffer_head=' + g['cpos']: 1.,
                        'w_pair=' + f['word'] + '_' + g['word']: 1.,
                        'cp_pair=' + f['cpos' ] + '_' + g['cpos' ]: 1.}
            # TODO is graph output the correct form of output

            out.add_edge(f['id'], g['id'], feats)
                      
    return out