
# defining the similarity function 
def similarity(x,y): 
    '''
    Function which calculates the similarity from 

    https://link.springer.com/content/pdf/10.1023/A:1022689900470.pdf

    What it does it takes the negative square root of the summation from 
    f(xi, yi) 
    
    where f(x, y) --> (xi - yi) ** 2 ONLY FOR NUMERICAL VALUES 
    for boolean values or symbolic values: 
    f(xi, yi) --> (xi =! yi) 

    if the value is NaN then we return 1 
    
    we would need to implement a check to see what type the value is 
    and whether we would use OHE or not in order to resolve it. 
    '''
    

    return -np.sqrt(np.sum(np.square(x-y)))




'''

Instance Based Learning Section: 

    - There are 3 different algorithms for this. 
    - We can use numba to use JIT to facilitate the compilation 
        increase the efficiency of the model 
        as well as vectorization whenever it is possible. 
   - Multiprocessing / threading to increase usage --> will consume more RAM.      
'''

def IBL1(x, y): 
    '''
    Concept Description initializes as 0 
    for each x in training set do: 
        for each y in CD do: 
            similarity(x, y) = similarity(y)
            y_max = np.max(similarity(x, y)) 
            if class(x) == class(y_max): 
                class += 1
            else: 
                class -+ 1
    '''
    # given a matrix of features X and labels Y we can just: 
    #vectorize the entire operation 
    return 1 if map(lambda x: similarity(x, y), X)==Y else -1



