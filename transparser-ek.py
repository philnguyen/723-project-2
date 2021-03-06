import networkx as nx
import sys, os
import operator
import numpy as np
import random


#############################################################################   

# Converts transition function to string 
def transToStr(t): 
   move = "False Move"
   if (t == shift): 
        move = "SHIFT"
   elif (t == arc_left): 
        move = "ARCLEFT"
   elif (t == arc_right): 
        move = "ARCRIGHT"
   return move
    
class Weights(dict):
    def __getitem__(self, idx):
       if self.has_key(idx):
           return dict.__getitem__(self, idx)
       else:
           return 0.
    
    def dotProduct(self, x, t):
        dot = 0.
        for feat, val in x.iteritems():
            dot += val * self[feat,t]
        return dot
    
    def update(self, x, y, t, counter=1):
        t = transToStr(t)
        for feat,val in x.iteritems():
            if val != 0.:
                self[feat, t] += y * val * counter
    
    # Computes the average for the perceptron 
    def average(self, cache, counter):
       avg = {}
       for feat, wVal in self.iteritems():
           cVal = cache[feat]
           avg[feat] = (wVal - ((1/counter) * cVal))
       return Weights(avg)


class Config():  
    #Initializes config based on passed phrase. Phrase is a list of nodes.
    def __init__(self, phrase):
        self.stack = []
        self.buff = []
        self.sent = phrase
        for i in self.sent.nodes():
            if self.sent.node[i]['word'] == "*root*":   #add root node to stack
                self.stack.append(self.sent.node[i])
            else:
                self.buff.append(self.sent.node[i])     #all other word nodes to buffer


    # For Debugging: Prints config's state
    def debugger(self, predMove=None, trueMove=None): 
        print "--------------------------------------------------------------"
        instance = None
        temp = []
        for i in self.sent:
            tup = "(" + self.sent.node[i]['word'] + ", " + self.sent.node[i]['id'] + ")"
            temp.append(tup)    
        print "Current Sentence: \n" + " ".join(temp)
        
        print "Current Stack: " + str(len(self.stack))
        for i in self.stack:
            instance = [i['id'], i['word'], i['lemma'], i['cpos'], i['pos'], i['feats'], i['head'], i['drel'], i['phead'], i['pdrel'], i['predhead']]
            print "\t".join(instance)

        print "Current Buffer: " + str(len(self.buff))
        for i in self.buff:
            instance = [i['id'], i['word'], i['lemma'], i['cpos'], i['pos'], i['feats'], i['head'], i['drel'], i['pdrel'], i['phead'], i['predhead']]
            print "\t".join(instance)
  
        print "Current Heads: " 
        for i in self.sent:
            instance = [self.sent.node[i]['id'], self.sent.node[i]['word'], self.sent.node[i]['head'], self.sent.node[i]['predhead']]
            print "\t".join(instance)
        
        if predMove != None:
            move = transToStr(predMove)
            print "Predicted Move: " + move
        else: 
            print "No move to predict."
        
        if trueMove != None:
            move = transToStr(trueMove)
            print "True Move: " + move
        else: 
            print "No true move."
            
    def getTrueHead(self, idx): 
        return self.sent.node[int(idx)]['head']
    
    def getPredHead(self, idx): 
        return self.sent.node[int(idx)]['predhead']

    def setPredHead(self, head, tail): 
         self.sent.node[int(tail)]['predhead'] = str(head)
        
    #Prints config to file.   
    def pp(self, out):             
        # Open file and append. Note: predicted head is printed in head position.
        file = open(out, "a")
        for i in self.sent.nodes(): 
            if self.sent.node[i]['word'] == "*root*": continue    
            instance = [self.sent.node[i]['id'], self.sent.node[i]['word'], self.sent.node[i]['lemma'], self.sent.node[i]['cpos'], 
            self.sent.node[i]['pos'], self.sent.node[i]['feats'], self.sent.node[i]['predhead'], self.sent.node[i]['drel'],  self.sent.node[i]['phead'], self.sent.node[i]['pdrel']]
            file.write("\t".join(instance)) 
            file.write("\n") 
        file.write("\n")                
         
    # Returns current config's features.    
    def features(self):
        return { 'w_stack=' + self.stack[0]['word']: 1.,     
                 'p_stack=' + self.stack[0]['cpos']: 1., 
                 'w_buff='  + self.buff[0]['word']: 1., 
                 'p_buff='  + self.buff[0]['cpos']: 1., 
                 'w_pair='  + self.stack[0]['word'] + '_' +  self.buff[0]['word']: 1., 
                 'p_pair='  + self.stack[0]['cpos'] + '_' + self.buff[0]['cpos']: 1. }

 
#############################################################################      

# Arc-standard trainsitions. 

def shift(config): 
    # Move top of buffer to top of stack.
    config.stack.insert(0, config.buff.pop(0))
    return config

def arc_left(config): 
    # Add arc from top of buffer to top of stack. Remove top of stack.
    topOfStack = config.stack[0]['id']
    topOfBuffer = config.buff[0]['id']
    config.setPredHead(topOfBuffer, topOfStack)  
    config.stack.pop(0)
    return config

def arc_right(config):  
    # Add arc from top of stack to top of buffer. Remove top of buffer and move top of stack to top of buffer.
    topOfStack = config.stack[0]['id']
    topOfBuffer = config.buff[0]['id']
    config.setPredHead(topOfStack, topOfBuffer)
    config.buff.pop(0)
    config.buff.insert(0, config.stack[0])
    config.stack.pop(0)
    return config


# Returns the true next move based on the passed configuration. 
def true_next_move(config): 
    def right_precondition(config): 
        # Determines an arc-right move is possible...
        topOfStack = config.stack[0]['id']
        topOfBuffer = config.buff[0]['id']
        if (config.getTrueHead(topOfBuffer) != topOfStack):   # if topOfStack is not the head of topOfBuffer
            return False
        for i in config.buff:
            if (topOfBuffer == config.sent.node[int(i['id'])]['head']):   # if topOfBuffer is head of any other word in stack or buffer
                   return False
        for i in config.stack:
           if (topOfBuffer == config.sent.node[int(i['id'])]['head']):      
                  return False
        return True 
    def left_precondition(config): 
         # Determines whether an arc-left move is possible...
         topOfStack = config.stack[0]['id']
         topOfBuffer = config.buff[0]['id']
         if int(topOfStack) == 0:       # if topOfStack is the root, false
             return False
         if (config.getTrueHead(topOfStack) != topOfBuffer):   # if topOfBuffer is not the head of topOfStack
             return False
         for i in config.buff:
             if (topOfStack == config.sent.node[int(i['id'])]['head']):   # if topOfStack is head of any other word in buffer or stack
                    return False
         for i in config.stack:
            if (topOfStack == config.sent.node[int(i['id'])]['head']):
                   return False
         return True
         
    trueMove = shift    
    if(left_precondition(config)): 
        trueMove = arc_left
    elif(right_precondition(config)): 
        trueMove = arc_right
  
    return trueMove  #Note: true move is a function.

# Returns the predicted next move.  Next move is the argmax transition.
def predict_next_move(config, weights, bias, trans):
    feat = config.features()
    
    # find argmax transition
    temp = []
    for t in trans: 
        temp.append(weights.dotProduct(feat, t) + bias[t])  
    idx = np.argmax(temp)  
    argmax = trans[idx]
    
    next_move = shift
    if (argmax == "ARCRIGHT"):
        next_move = arc_right
    elif (argmax == "ARCLEFT"):
        next_move = arc_left
    return next_move    # note: next_move is a function


# For test data: Run arc-standard algorithm on the passed config using weights and bias. 
def arc_standard(config, weights, bias): 
    while (len(config.buff) > 0) and (len(config.stack) > 0):
        next_move = predict_next_move(config, weights, bias, ['ARCRIGHT', 'ARCLEFT', 'SHIFT'])
        config = next_move(config)   
    return config


#############################################################################  

# def numMistakes(config):
#     err = 0.
#     for i in config.sent.nodes():
#         if config.getTrueHead(i) != config.getPredHead(i):
#             err += 1
#     return err

# For training data: Runs avg perceptron algorithm for one instance. Predicts and adjustsweights for predicted moves.
def train_instance(weights, cache, counter, bias, cacheBias, totalErr, instance):
   config = Config(instance)
   
   while (len(config.buff) > 0) and (len(config.stack) > 0):
     pred_move = predict_next_move(config, weights, bias, ['ARCRIGHT', 'ARCLEFT', 'SHIFT'])
     true_move = true_next_move(config)

     if pred_move != true_move:
        feat = config.features()
        weights.update(feat, -1, pred_move)
        weights.update(feat, 1, true_move)
        cache.update(feat, -1, pred_move, counter=counter)
        cache.update(feat, 1, true_move, counter=counter)
        bias[transToStr(pred_move)] -= 1
        bias[transToStr(true_move)] += 1
        cacheBias[transToStr(pred_move)] -= 1 * counter
        cacheBias[transToStr(true_move)] += 1 * counter
        
     counter += 1 
     config = true_move(config)

   # config.debugger()
   # totalErr += numMistakes(config)
   return (weights, cache, counter, bias, cacheBias, totalErr)


# For Test Data: Runs arc standard algorithm using weights for move predictions.   
def test_instance(weights, bias, instance): 
    config = Config(instance)
    parse = arc_standard(config, weights, bias)
    return parse

############################################################################# 

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
                G.add_node(0, {'id': '0','word': '*root*', 'lemma': '*root*', 'cpos': '*root*', 'pos': '*root*', 'feats': '*root*', 'head': '*root*', 'drel': '*root*', 'phead':'*root*', 'pdrel':'*root*', 'predhead': '*root*'})
            [id, word, lemma, cpos, pos, feats, head, drel, phead, pdrel] = l.split('\t')
            G.add_node(int(id), {'id': str(id),
                                 'word' : word,
                                 'lemma': lemma,
                                 'cpos' : cpos,
                                 'pos'  : pos,
                                 'feats': feats, 
                                 'head': head,
                                 'drel': drel, 
                                 'phead': phead,
                                 'pdrel': pdrel,
                                 'predhead': '_' })          
    if G != None:
        yield G
    h.close()


    
def main(argv):
   # train, test, out = ['en.tr', 'en.tst', 'en.tst.out']
   
   _, train, test, out = argv
   
   # Delete prior output files if they exist
   if os.path.exists(out):
       os.remove(out)
   else:
       print("Can't remove %s file." % out)
   
   counter = 1.
   weights = Weights()
   bias = Weights()
   cache = Weights()
   cacheBias = Weights()
   
   # Iterates dev file instances and runs average perceptron algorithm on each.
   for iteration in range(5):
       totalErr = 0.
       for S in iterCoNLL(train):
          (weights, cache, counter, bias, cacheBias, totalErr) = train_instance(weights, cache, counter, bias, cacheBias, totalErr, S)
       # print totalErr

   avgWeights = weights.average(cache, counter)
   avgBias = bias.average(cacheBias, counter)
    
   # Iterates test file instances. Uses the avg of the weights trained above to predict arc standard moves.
   # Then prints the predicted parse to output file. 
   for S in iterCoNLL(test):
       parse = test_instance(avgWeights, avgBias, S)
       parse.pp(out)


if __name__ == "__main__":
   main(sys.argv[1:])


#############################################################################   