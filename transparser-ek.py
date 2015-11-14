import networkx as nx
import sys, os
import operator
import numpy as np


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
        if (t == shift): 
            t = "SHIFT"
        elif (t == arc_left): 
            t = "ARCLEFT"
        elif (t == arc_right): 
            t = "ARCRIGHT"
        for feat,val in x.iteritems():
            if val != 0.:
                self[feat, t] += y * val * counter
    
    def average(self, x, cache, counter):
       avg = {}
       for (wFeat, wVal), (cFeat, cVal) in self.iteritems(), cache,.iteritems():
           avg[wFeat] = (wVal - ((1/counter) * cVal))
       return avg


class Config():  
    def __init__(self, phrase):
        self.stack = []
        self.buff = []
        self.sent = phrase
        for i in self.sent.nodes():
            if self.sent.node[i]['word'] == "*root*":   #add root to stack
                self.stack.append(self.sent.node[i])
            else:
                self.buff.append(self.sent.node[i])     #all other words to buffer


    # Prints config's state
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
            if (predMove == shift):
                move = "SHIFT"
            if (predMove == arc_right):
                move = "ARCRIGHT"
            if (predMove == arc_left):
                move = "ARCLEFT"
            print "Predicted Move: " + move
        else: 
            print "No move to predict."
        
        if trueMove != None:
            if (trueMove == shift):
                move = "SHIFT"
            if (trueMove == arc_right):
                move = "ARCRIGHT"
            if (trueMove == arc_left):
                move = "ARCLEFT"
            print "True Move: " + move
        else: 
            print "No true move."

    def getTrueHead(self, idx): 
        return self.sent.node[int(idx)]['head']

    def setHead(self, head, tail): 
         self.sent.node[int(tail)]['predhead'] = str(head)
        
    def pp(self, out):             
        #open file and append. Note: predicted head is printed in head position.
        file = open(out, "a")
        for i in self.sent.nodes(): 
            if self.sent.node[i]['word'] == "*root*": 
                continue    
            instance = [self.sent.node[i]['id'], self.sent.node[i]['word'], self.sent.node[i]['lemma'], self.sent.node[i]['cpos'], 
            self.sent.node[i]['pos'], self.sent.node[i]['feats'], self.sent.node[i]['predhead'], self.sent.node[i]['drel'],  self.sent.node[i]['phead'], self.sent.node[i]['pdrel']]
            file.write("\t".join(instance)) 
            file.write("\n") 
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
    topOfStack = config.stack[0]['id']
    topOfBuffer = config.buff[0]['id']
    config.setHead(topOfStack, topOfBuffer)  
    config.stack.pop(0)
    return config

def arc_right(config):  
    # Add arc from top of stack to top of buffer. Remove top of buffer and move top of stack to top of buffer.
    topOfStack = config.stack[0]['id']
    topOfBuffer = config.buff[0]['id']
    config.setHead(topOfBuffer, topOfStack)
    config.buff.pop(0)
    config.buff.insert(0, config.stack[0])
    config.stack.pop(0)
    return config


def true_next_move(config): 
    def right_precondition(config): 
        # The top of the stack is the head of the top of the buffer. 
        topOfStack = config.stack[0]['id']
        topOfBuffer = config.buff[0]['id']
        print config.getTrueHead(topOfBuffer)
        print topOfStack
        if (config.getTrueHead(topOfBuffer) != topOfStack):
            return False
        for i in config.buff:
            if (topOfBuffer == config.sent.node[int(i['id'])]['head']):
                   return False
        # for i in config.stack:
        #    if topOfBuffer == i: continue
        #    if (topOfBuffer == config.sent.node[int(i)]['head']):
        #           return False   
        return True 
    def left_precondition(config): 
         # The top of buffer is the top of stacks's head and the top of the stack is not the head
         # of any word currently in the buffer or stack.
         topOfStack = config.stack[0]['id']
         topOfBuffer = config.buff[0]['id']
         if int(topOfStack) == 0: 
             return False
         if (config.getTrueHead(topOfStack) != topOfBuffer): 
             return False 
         for i in config.buff:
             if (topOfStack == config.sent.node[int(i['id'])]['head']):
                    return False
         # for i in config.stack:
         #    if topOfStack == i: continue
         #    if (topOfStack == config.sent.node[int(i)]['head']):
         #           return False
         return True
         
    nextMove = shift
    if(right_precondition(config)): 
        nextMove = arc_right
    elif(left_precondition(config)): 
        nextMove = arc_left   
    return nextMove


def predict_next_move(config, weights, trans):
    feat = config.features()
    
    temp = []
    for t in trans: 
        temp.append(weights.dotProduct(feat, t))
    idx = np.argmax(temp)
    argmax = trans[idx]
    
    next_move = shift
    if (argmax == "ARCRIGHT"):
        next_move = arc_right
    elif (argmax == "ARCLEFT"):
        next_move = arc_left
    return next_move


#############################################################################  

# For Training: Runs average perceptron algorithm for one instance. 
def train_instance(weights, cache, counter, instance): 
   config = Config(instance)
   config.debugger()

   while (config.buff != []) and (config.buff[0]['id'] != '0'):
     pred_move = predict_next_move(config, weights, ['ARCRIGHT', 'ARCLEFT', 'SHIFT'])
     true_move = true_next_move(config)
          
     print "debugger"
     config.debugger(pred_move, true_move)

     if pred_move != true_move:
        feat = config.features()
        weights.update(feat, -1, pred_move)
        weights.update(feat, 1, true_move)
        cache.update(feat, -1, pred_move, counter=counter)
        cache.update(feat, 1, true_move, counter=counter)

     config = true_move(config)     # Execute oracle move on config.
     counter += 1

   return (weights, cache, counter)


############################################################################# 


# Utility function & Main

# Reads each phrase from file. 
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
   # NOTE: change to "devFile, testFile, outFile = argv" for submission.
   # NOTE 2: currently overwrites the output currently in en.tst.out, so save somewhere else if you want that output saved.
   dev, test, out = ['en.tr100', 'en.tst', 'en.tst.out']
   
   #delete prior output files if they exist
   if os.path.exists(out):
       os.remove(out)
   else:
       print("Can't remove %s file." % out)
   
   counter = 1
   weights = Weights()
   cache = Weights()
   
   # Iterates dev file instances and runs average perceptron algorithm on each.
   for iteration in range(5):
       for S in iterCoNLL(dev): 
          (weights, cache, counter) = train_instance(weights, cache, counter, S)

   avg = weights.average(cache, counter)
   print avg
    
   # Iterates test file instances. Uses the weights trained above to predict arc standard moves.
   # Then prints the predicted parse to output file. 
   # for iteration in range(100):
#        for S in iterCoNLL(testFile):
#            parse = test_instance(avg, S)
#            parse.pp(out)


if __name__ == "__main__":
   main(sys.argv[1:])


#############################################################################   