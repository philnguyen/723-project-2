import networkx as nx
import sys, os
import operator
import numpy as np
import random
from collections import defaultdict


#############################################################################   
# at 68% right now on dev with entr100
#############################################################################   

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
           return .02
    
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
    
    def average(self, cache, counter):
       avg = {}
       for feat, wVal in self.iteritems():
           cVal = cache[feat]
           avg[feat] = (wVal - ((1/counter) * cVal))
       return Weights(avg)


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
        
    def pp(self, out):             
        #open file and append. Note: predicted head is printed in head position.
        file = open(out, "a")
        for i in self.sent.nodes(): 
            if self.sent.node[i]['word'] == "*root*": continue    
            instance = [self.sent.node[i]['id'], self.sent.node[i]['word'], self.sent.node[i]['lemma'], self.sent.node[i]['cpos'], 
            self.sent.node[i]['pos'], self.sent.node[i]['feats'], self.sent.node[i]['predhead'], self.sent.node[i]['drel'],  self.sent.node[i]['phead'], self.sent.node[i]['pdrel']]
            file.write("\t".join(instance)) 
            file.write("\n") 
        file.write("\n")                 
           
    def defFeatures(self):
        return { 'w_stack=' + self.stack[0]['word']: 1., 
                 'p_stack=' + self.stack[0]['cpos']: 1., 
                 'w_buff='  + self.buff[0]['word']: 1., 
                 'p_buff='  + self.buff[0]['cpos']: 1., 
                 'w_pair='  + self.stack[0]['word'] + '_' +  self.buff[0]['word']: 1., 
                 'p_pair='  + self.stack[0]['cpos'] + '_' + self.buff[0]['cpos']: 1. }
 
    def features(self):  
        stemp = []
        btemp = []

        if (len(self.stack) == 1): 
            stemp.insert(0, self.stack[0])
            stemp.insert(1, {'cpos': "<s>", 'word': "<s>"})
            stemp.insert(2, {'cpos': "<s>", 'word': "<s>"})
        elif (len(self.stack) == 2): 
            stemp.insert(0, self.stack[0])
            stemp.insert(1, self.stack[1])
            stemp.insert(2, {'cpos': "<s>", 'word': "<s>"})
        elif (len(self.stack) >= 3): 
            stemp.insert(0, self.stack[0])
            stemp.insert(1, self.stack[1])
            stemp.insert(2, self.stack[2])
        
        if (len(self.buff) == 1): 
            btemp.insert(0, self.buff[0])
            btemp.insert(1, {'cpos': "<s>", 'word': "<s>"})
            btemp.insert(2, {'cpos': "<s>", 'word': "<s>"})
        elif (len(self.buff) == 2):
            btemp.insert(0, self.buff[0])
            btemp.insert(1, self.buff[1])
            btemp.insert(2, {'cpos': "<s>", 'word': "<s>"}) 
        elif (len(self.buff) >= 3):
            btemp.insert(0, self.buff[0])
            btemp.insert(1, self.buff[1])
            btemp.insert(2, self.buff[2])
            
        feat = { 
                'u_stack0_pair='  + stemp[0]['word'] + '_' + stemp[0]['cpos'] : 1.,
                'u_stack0_word=' + stemp[0]['word']: 1., 
                'u_stack0_pos=' + stemp[0]['cpos']: 1.,
                
                'u_buff0_pair='  + btemp[0]['word'] + '_' + btemp[0]['cpos'] : 1., 
                'u_buff0_pos=' + btemp[0]['cpos']: 1., 
                'u_buff0_word=' + btemp[0]['word']: 1.,
                
                'u_buff1_pair='  + btemp[1]['word'] + '_' + btemp[1]['cpos'] : 1., 
                'u_buff1_pos=' + btemp[1]['cpos']: 1., 
                'u_buff1_word=' + btemp[1]['word']: 1.,
                
                'u_buff2_pair='  + btemp[2]['word'] + '_' + btemp[2]['cpos'] : 1., 
                'u_buff2_pos=' + btemp[2]['cpos']: 1., 
                'u_buff2_word=' + btemp[2]['word']: 1.,
                
                'w_pair00_1='  + stemp[0]['word'] + '_' + stemp[0]['cpos'] + '_'+  btemp[0]['word'] + '_'+  btemp[0]['cpos']: 1.,
                'w_pair00_2='  + stemp[0]['word'] + '_' + stemp[0]['cpos'] + '_'+  btemp[0]['word'] : 1.,
                'w_pair00_3='  + stemp[0]['word'] + '_' +  btemp[0]['word'] + '_'+  btemp[0]['cpos']: 1.,
                'w_pair00_4='  + stemp[0]['word'] + '_' + stemp[0]['cpos'] + '_'+  btemp[0]['cpos'] : 1.,
                'w_pair00_5='  + stemp[0]['cpos'] + '_' +  btemp[0]['word'] + '_'+  btemp[0]['cpos']: 1.,
                'w_pair00_6='  + stemp[0]['word'] + '_' +  btemp[0]['word']: 1.,
                'w_pair00_7='  + stemp[0]['cpos'] + '_' +  btemp[0]['cpos']: 1.,
                'w_pair01_8='  + btemp[0]['cpos'] + '_' +  btemp[1]['cpos']: 1.,
            
                'w_triple012_1=' + btemp[0]['cpos'] + '_' +  btemp[1]['cpos'] + '_' +  btemp[2]['cpos']: 1.,
                'w_triple012_2=' + stemp[0]['cpos'] + '_' +  btemp[1]['cpos'] + '_' +  btemp[2]['cpos']: 1. 
                
                #TODO: add other features described in paper
            }
     
        return feat
 
    def oldfeatures(self):         
        unigram_feat = { 
                'u_stack_id=' + self.stack[0]['id']: 1., 
                'u_stack_word=' + self.stack[0]['word']: 1., 
                'u_stack_pos=' + self.stack[0]['cpos']: 1.,
                'u_buff_id=' + self.buff[0]['id']: 1., 
                'u_buff_word=' + self.buff[0]['word']: 1., 
                'u_buff_pos=' + self.buff[0]['cpos']: 1.,
                'u_pair_id='  + self.stack[0]['id'] + '_' +  self.buff[0]['id']: 1., 
                'u_pair_word='  + self.stack[0]['word'] + '_' +  self.buff[0]['word']: 1.,
                'u_pair_pos='  + self.stack[0]['pos'] + '_' +  self.buff[0]['pos']: 1.}
                
        if len(self.stack) > 1: 
            bigram_feat = {
                'b_stack_id=' + self.stack[0]['id'] + '_' +  self.stack[1]['id']: 1., 
                'b_stack_word=' + self.stack[0]['word'] + '_' +  self.stack[1]['word']: 1., 
                'b_stack_pos=' + self.stack[0]['pos'] + '_' +  self.stack[1]['pos']: 1.
            }
        else: 
            bigram_feat = {
                'b_stack_id=' + self.stack[0]['id'] + '_' +  "<s>": 1., 
                'b_stack_word=' + self.stack[0]['word'] + '_' +  "<s>": 1., 
                'b_stack_pos=' + self.stack[0]['pos'] + '_' +  "<s>": 1.
            }
            
        if len(self.buff) > 1: 
            bigram_feat = { 
                'b_buff_id=' + self.buff[0]['id'] + '_' +  self.buff[1]['id']: 1., 
                'b_buff_word=' + self.buff[0]['word'] + '_' +  self.buff[1]['word']: 1., 
                'b_buff_pos=' + self.buff[0]['pos'] + '_' +  self.buff[1]['pos']: 1. }
        else: 
            bigram_feat = {
                'b_buff_id=' + self.buff[0]['id'] + '_' +  "<s>": 1., 
                'b_buff_word=' + self.buff[0]['word'] + '_' +  "<s>": 1., 
                'b_buff_pos=' + self.buff[0]['pos'] + '_' +  "<s>": 1. }
            
        # if len(self.buff > 1) and len(self.stack > 1):
        if len(self.stack) > 2: 
            trigram_feat = { 
                't_stack_id=' + self.stack[0]['id'] + '_' +  self.stack[1]['id'] + '_' +  self.stack[2]['id']: 1., 
                't_stack_word=' + self.stack[0]['word'] + '_' +  self.stack[1]['word'] + '_' +  self.stack[2]['word']: 1., 
                't_stack_pos=' + self.stack[0]['pos'] + '_' +  self.stack[1]['pos'] + '_' +  self.stack[2]['pos']: 1. }
        elif (len(self.stack) == 2): 
            trigram_feat = { 
                't_stack_id=' + self.stack[0]['id'] + '_' +  self.stack[1]['id'] + '_' +  "<s>": 1., 
                't_stack_word=' + self.stack[0]['word'] + '_' +  self.stack[1]['word'] + '_' +  "<s>": 1., 
                't_stack_pos=' + self.stack[0]['pos'] + '_' +  self.stack[1]['pos'] + '_' +  "<s>": 1. }
        else: 
            trigram_feat = { 
                't_stack_id=' + self.stack[0]['id'] + '_' +  "<s>" + '_' +  "<s>": 1., 
                't_stack_word=' + self.stack[0]['word'] + '_' +  "<s>" + '_' +  "<s>": 1., 
                't_stack_pos=' + self.stack[0]['pos'] + '_' +  "<s>" + '_' + "<s>": 1. }
            
        if len(self.buff) > 2: 
            trigram_feat = { 
                't_buff_id=' + self.buff[0]['id'] + '_' +  self.buff[1]['id'] + '_' +  self.buff[2]['id']: 1., 
                't_buff_word=' + self.buff[0]['word'] + '_' +  self.buff[1]['word'] + '_' +  self.buff[2]['word']: 1., 
                't_buff_pos=' + self.buff[0]['pos'] + '_' +  self.buff[1]['pos'] + '_' +  self.buff[2]['pos']: 1. }
        elif (len(self.buff) == 2): 
            trigram_feat = { 
                't_buff_id=' + self.buff[0]['id'] + '_' +  self.buff[1]['id'] + '_' +  "<s>": 1., 
                't_buff_word=' + self.buff[0]['word'] + '_' +  self.buff[1]['word'] + '_' +  "<s>": 1., 
                't_buff_pos=' + self.buff[0]['pos'] + '_' +  self.buff[1]['pos'] + '_' +  "<s>": 1. }
        else: 
            trigram_feat = { 
                't_buff_id=' + self.buff[0]['id'] + '_' +  "<s>" + '_' +  "<s>": 1., 
                't_buff_word=' + self.buff[0]['word'] + '_' +  "<s>" + '_' +  "<s>": 1., 
                't_buff_pos=' + self.buff[0]['pos'] + '_' +  "<s>" + '_' +  "<s>": 1. }
                
        # if len(self.buff > 2) and len(self.stack > 2):
        #     trigram_feat = {}
            
        return dict(unigram_feat.items() + bigram_feat.items() + trigram_feat.items())
 
#############################################################################      


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


def true_next_move(config): 
    def right_precondition(config): 
        # The top of the stack is the head of the top of the buffer. 
        topOfStack = config.stack[0]['id']
        topOfBuffer = config.buff[0]['id']
        if (config.getTrueHead(topOfBuffer) != topOfStack):
            return False
        if (config.getPredHead(topOfBuffer) != "_"):
            return False
        for i in config.buff:
            if (topOfBuffer == config.sent.node[int(i['id'])]['head']):
                   return False
        for i in config.stack:
           if (topOfBuffer == config.sent.node[int(i['id'])]['head']):
                  return False
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
         if (config.getPredHead(topOfStack) != "_"):
             return False
         for i in config.buff:
             if (topOfStack == config.sent.node[int(i['id'])]['head']):
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
  
    return trueMove


def predict_next_move(config, weights, bias, trans):
    feat = config.features()
    
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
    return next_move


def arc_standard(config, weights, bias): 
    while (len(config.buff) > 0) and (len(config.stack) > 0):
        next_move = predict_next_move(config, weights, bias, ['ARCRIGHT', 'ARCLEFT', 'SHIFT'])
        config = next_move(config)   
    return config


#############################################################################  

# For Training: Runs avg perceptron algorithm for one instance. Predicts and adjustsweights for predicted moves.
def numMistakes(config):
    err = 0.
    for i in config.sent.nodes():
        if config.getTrueHead(i) != config.getPredHead(i):
            err += 1
    return err

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

   config.debugger()
   totalErr += numMistakes(config)
   return (weights, cache, counter, bias, cacheBias, totalErr)



# For Test Data: Runs arc standard algorithm using weights for move predictions.   
def test_instance(weights, bias, instance): 
    config = Config(instance)
    parse = arc_standard(config, weights, bias)
    return parse

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
   train, test, out = ['en.tr100', 'en.dev', 'en.dev.out.2']
   
   #delete prior output files if they exist
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
   for iteration in range(500):
       totalErr = 0.
       instances = []
       for S in iterCoNLL(train): 
           instances.append(S)
       print len(instances)
       for i in range(len(instances)): 
           S = random.choice(instances)   #randomize training instances for each iteration
           instances.remove(S)
           (weights, cache, counter, bias, cacheBias, totalErr) = train_instance(weights, cache, counter, bias, cacheBias, totalErr, S)
       print totalErr

   avgWeights = weights.average(cache, counter)
   avgBias = bias.average(cacheBias, counter)
    
   # Iterates test file instances. Uses the weights trained above to predict arc standard moves.
   # Then prints the predicted parse to output file. 
   for S in iterCoNLL(test):
       parse = test_instance(avgWeights, avgBias, S)
       parse.pp(out)


if __name__ == "__main__":
   main(sys.argv[1:])


#############################################################################   