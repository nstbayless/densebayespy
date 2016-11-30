# dense bayesian network sampler

import numpy as np

TOLERANCE = 0.000001

class BayesNet:
    size   = []
    indep  = []
    probfn = []
    node_n  = 0
    leaf_n  = 0
    space_n = 0
    # relmat: absmat of p(a|a_1,a_2,...)
    # wmat: working absmat
    def __init__(self,nodes):
        """
        nodes: array of [
                range: possible values this var can take
                p: probability function of the form
                (varset) -> double, output summing to one over this variable.
                indep: bool, is this var actually list of independent vars? (terminal node only)
            ]
        """
        meta_dim = []
        for node in nodes:
            self.size.append(node[0])
            self.probfn.append(node[1])
            if (len(node)>=3):
                self.indep.append(node[2])
            else:
                self.indep.append(False)
            meta_dim.append(node[0]+1)
            self.node_n+=1
        
        self.relmat = np.zeros(meta_dim,dtype=np.double)
        self.wmat = np.zeros(meta_dim,dtype=np.double)
        
        self.leaf_n = np.prod(self.size)
        self.space_n = np.prod(meta_dim)
        
        self.relmat.itemset(tuple([-1]*self.node_n),1)
        self._fill(0,[-1]*self.node_n)

        print(self.relmat);
        print("node count:", self.node_n)
        print("leaf count:", self.leaf_n)
        print("space used:", self.space_n)
    
    def _fill(self,dim,varset):
        if (dim>=self.node_n):
            return
        sum_lp = 0
        for i in range(self.size[dim]):
            varset[dim]=i
            loc_p = (self.probfn[dim])(varset)
            self.relmat.itemset(tuple(varset),loc_p)
            sum_lp+=loc_p
            self._fill(dim+1,varset)
            varset[dim]=-1
        if (not self.indep[dim]):
            assert(abs(sum_lp-1)<TOLERANCE)
            
    def _make_wmat(self,obs):
        self._make_wmat_helper(0,obs,[-1]*self.node_n,1.0)
        
        print(self.wmat)
    
    def _make_wmat_helper(self,dim,obs,varset,loss):
        print(varset)
        print(loss)
        self.wmat.itemset(tuple(varset),loss)
        if (dim>=self.node_n):
            return
        rel_prior = self.relmat.item(tuple(varset))
        print(rel_prior);
        if obs[dim]==-1: #no observation
            for i in range(self.size[dim]):
                varset[dim]=i
                self._make_wmat_helper(dim+1,obs,varset,loss*rel_prior)
                varset[dim]=-1
        else:
            assert(obs[dim]>=0 and obs[dim] < self.size[dim])
            varset[dim]=obs[dim]
            self._make_wmat_helper(dim+1,obs,varset,loss)
            varset[dim]=-1
