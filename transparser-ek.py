import networkx as nx
import sys
import operator


#############################################################################    

class Weights(dict):
    def __getitem__(self, idx):
       if self.has_key(idx):
           return dict.__getitem__(self, idx)
       else:
           return 0.
    
    def dotProduct(self, x, t):
        dot = 0.
        for feat,val in x.iteritems():
            dot += val * self[feat,t]
        return dot
    
    def update(self, x, y, t, counter=1):
        for feat,val in x.iteritems():
            if val != 0.:
                self[feat, t] += y * val * counter
    
    def average(self, cache, counter): 
       avg = {}
       for (wFeat, wVal), (cFeat, cVal) in self, cache:
           avg[wFeat] = (wVal - ((1/counter) * cVal))
       return avg


class Config():  
    sent = None  # Seq of words -- a list of nodes. arcs are kept as attributes.
    
    stack = []  
    buff = []
    
    def __init__(self, sent):
        self.sent = sent
        for w in sent.nodes():
            if w['word'] == "*root*":   #add root to stack
                self.stack << w
            else: 
                self.buff.append(w)     #all other words to buffer
        
    def pp(self, out): 
        file = openfile(out, "a")
        for w in sent.nodes(): 
            if w['word'] == "root": continue    
            instance = [w['id'], w['word'], w['lemma'], w['cpos'], w['pos'], w['feats'], w['head'], w['drel'], w['phead'], w['pdrel']]
            file.write("\t".join(itemlist)) 
        file.write("\n")                 
           
    def features(self):
        return { 'w_stack=' + self.stack[0]['word']: 1., 
                 'p_stack=' + self.stack[0]['cpos']: 1., 
                 'w_buff='  + self.buff[0]['word']: 1., 
                 'p_buff='  + self.buff[0]['cpos']: 1., 
                 'w_pair='  + self.stack[0]['word'] + '_' +  self.buff[0]['word']: 1., 
                 'p_pair='  + self.stack[0]['cpos'] + '_' + self.buff[0]['cpos']: 1. }
 
 
#############################################################################      

def shift(config): 
    # Move top of buffer to top of stack.
    config.stack.insert(0, config.buff.pop(0))
    return config

def arc_left(config): 
    # Add arc from top of buffer to top of stack. Remove top of stack.
    topOfStack = config.stack.get(0)
    topOfBuffer = config.buff.get(0)
    config.sent.node[topOfStack['id']]['head'] = topOfBuffer['id']
    config.stack.pop(0)
    return config

def arc_right(config):  
    # Add arc from top of stack to top of buffer. Remove top of buffer and move top of stack to top of buffer.
    topOfStack = config.stack.get(0)
    topOfBuffer = config.buff.get(0)
    config.sent.node[topOfBuffer['id']]['head'] = topOfStack['id']
    config.buff.pop(0)
    config.buff.insert(0, config.stack.pop(0))
    return config


def true_next_move(config): 
    def right_precondition(config): 
        # The top of the stack is the head of the top of the buffer. 
        topOfStack = config.stack.get(0)
        topOfBuffer = config.buff.get(0)
        if (topOfBuffer['head'] != topOfStack['id']): 
            return False 
        for i from range(1, length(config.buff)): 
            if topOfBuffer['id'] == i['head']: 
                   return False     
        return True 
    def left_precondition(config): 
         # The top of buffer is the top of stacks's head and the top of the stack is not the head
         # of any word currently in the buffer or stack.
         topOfStack = config.stack.get(0)
         topOfBuffer = config.buff.get(0)
         if (topOfStack['head'] != topOfBuffer['id']): 
             return False 
         for i from range(1, length(config.buff)): 
             if topOfStack['id'] == i['head']: 
                    return False    
         return True
         
    nextMove = shift
    if(right_precondition(config)): 
        nextMove = arc_right
    elif(left_precondition(config)): 
        nextMove = arc_left   
    return nextMove

    
def predict_next_move(config, weights, trans): 
    feat = config.features()        
    argmax = (t for t in trans if max([weights.dotProduct(feat, t) for t in trans]))
    next_move = shift
    if (argmax.equals("ARCRIGHT")): 
        next_move = arc_right
    elif (argmax.equals("ARCLEFT")): 
        next_move = arc_left
    return next_move
  
  
def arc_standard(config, weights): 
    while config.buff: 
        next_move = predict_next_move(config, weights) 
        config = next_move(config)   
    return config

#############################################################################  

# For Training: Runs average perceptron algorithm for one instance. 
def train_instance(weights, cache, counter, instance): 
   config = Config(instance)
   
   while config.buff:  
     pred_move = predict_next_move(weights, config, ['ARCRIGHT', 'ARCLEFT', 'SHIFT'])
     true_move = true_next_move(config)
    
     if pred_move != true_move:
        feat = config.features()
        weights = weights.update(feat, -1, predMove)
        weights = weights.update(feat, 1, trueMove)
        cache = cache.update(feat, -1, predMove, counter=counter)
        cache = cache.update(feat, 1, trueMove, counter=counter)
     
     config = true_move(config)     # Execute oracle move on config.
     counter += 1 
    
   return (weights, cache, counter)
       
            
# For Test Data: Runs arc standard algorithm using weights for move predictions.   
def test_instance(weights, instance): 
    config = Config(instance)
    parse = arc_standard(config, weights)
    return parse


############################################################################# 


# Utility function & Main

# Reads each phrase from file. 
def iterCoNLL(filename, train=True):
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
            [id, word, lemma, cpos, pos, feats, head, drel, phead, pdrel] = l.split('\t')
            G.add_node(int(id), {'id': id,
                                 'word' : word,
                                 'lemma': lemma,
                                 'cpos' : cpos,
                                 'pos'  : pos,
                                 'feats': feats, 
                                 'head': head,
                                 'drel': drel, 
                                 'phead': phead,
                                 'pdrel': pdrel })             
    if G != None:
        yield G
    h.close()


    
def main(argv):
   # NOTE: change to "devFile, testFile, outFile = argv" for submission.
   dev, test, out = ['en.tr100', 'en.tst', 'en.tst.out']
   
   counter = 1
   weights = Weights()
   cache = Weights()
   
   # Iterates dev file instances and runs average perceptron algorithm on each.
   for iteration in range(100):
       for S in iterCoNLL(devFile): 
           (weights, cache, counter) = train_instance(weights, cache, counter, S)

   avg = weights.average(cache, counter) 
    
   # Iterates test file instances. Uses the weights trained above to predict arc standard moves.
   # Then prints the predicted parse to output file. 
   for iteration in range(100):
       for S in iterCoNLL(testFile):
           parse = test_instance(avg, S)
           parse.pp(out)


if __name__ == "__main__":
   main(sys.argv[1:])


#############################################################################   
