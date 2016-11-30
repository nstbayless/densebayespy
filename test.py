# monty hall problem

from btree import BayesNet

def p_prize(varset):
    return 1/3.0
    
def p_choice(varset):
    return 1/3.0
    
def p_monty(varset):
    prize = varset[0]
    choice = varset[1]
    monty = varset[2]
    if (monty == prize):
        return 0.0
    if (monty == choice):
        return 0.0
    if (choice == prize):
        return 0.5
    return 1;
    
bn = BayesNet([
    [3,  p_prize],
    [3,  p_choice],
    #[3,  p_monty],
])

bn._make_wmat([-1,-1])
