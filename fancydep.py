"""
some ideas for part 3... 

- add features, including trigram feature []

- other paper showed dynamic explore oracle worked better
- could try randomizing the order the training instances are chosen 
- could try the larger training set
- any combination of these


"""

# call this before instead of using true next move at end of loop
def chooseNextExplore(i, t): 
    if (i > k and rand() > p): 
        return predMove
    else 
        return chooseNextAmb(i, t)


        
    
        

