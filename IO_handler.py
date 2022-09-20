'''
Created on Apr 11, 2017

@author: Daniel Duque
'''
import numpy as np

# CONSTANTS
demand_log_mean = 1.0
demand_log_std = 1.0
demand_rho = 0.0


class Datahandler:
    '''
    This class manage the input data of the Capacity Allocation model 
    '''
    def __init__(self, n, m, scen, b, rho, distribution='lognormal'):
        '''
        Define attributes used throughout the models
        '''
        #Sets
        self.Omega = None  # Set of scenarios
        self.I = None  # Set of generators
        self.J = None  # Set of Customers
        
        #Parameters
        self.prob = None  # Prob. of scenario w
        self.c = None  # Unit operation cost of energy sent form generator i to customer j
        self.k = None  # Unit cost of installing capacity
        self.rho = None  # Unit subcontracting cost for demand site j
        self.b = None  # Maximum capacity instalation (scalar)
        self.d_ub = None
        '''Random parameters'''
        self.d = None  # Demand for customer j in scenario w
        self.f = None  # Available fraction of allocated capacity at generator i in scenario w
        self.distribution = distribution
        self.buildData(n, m, scen, b, rho)
    
    def buildData(self, n, m, scen, b, rho):
        '''
        Builds an instance of the problem
        ==================
        input parameters:
        n:  Number of customers
        m:  Number of facilities
        scen: Number of scenarios
        b:  butget
        rho:  Shortfall penalty
        #Original code in AMPL capacity.run
        #Author: Prof. David Morton
        '''
        #Set a seed
        np.random.seed(0)
        self.Omega = list(range(0, scen))
        self.I = list(range(0, m))
        self.J = list(range(0, n))
        self.b = b
        self.rho = rho
        
        # pmf
        self.prob = np.ones(scen) * (1.0 / scen)
        self.n_sce = scen
        
        # Facility locations (x,y) coordinates in unit square
        xi = np.zeros(m)
        yi = np.zeros(m)
        for i in self.I:
            xi[i] = np.random.uniform()
            yi[i] = np.random.uniform()
        # Customer locations (x,y) coordinates in unit square
        xj = np.zeros(n)
        yj = np.zeros(n)
        for i in self.J:
            xj[i] = np.random.uniform()
            yj[i] = np.random.uniform()
        # unit cost of satisfying demand j from i is proportional to distance
        self.c = np.zeros((m, n))
        for i in self.I:
            for j in self.J:
                self.c[i, j] = np.round(np.sqrt((xi[i] - xj[j])**2 + (yi[i] - yj[j])**2), 3)
        # demands are independent lognormals
        np.random.seed(1000)  # 1000 for big experiments
        self.d = np.zeros((n, scen))
        
        mean_vec = demand_log_mean * np.ones(n)
        cov_mat = (demand_log_std**2) * (np.eye(n) + demand_rho * (np.ones((n, n)) - np.eye(n)))
        for w in self.Omega:
            if self.distribution == 'lognormal':
                self.d[:, w] = np.round(np.exp(np.random.multivariate_normal(mean_vec, cov_mat, size=1)), 1)
            else:
                self.d[:, w] = np.round(np.random.uniform(0, 50, size=n), 1)
        '''Additional parameter for potential extensions'''
        # Unit cost of installing capacity
        self.k = np.zeros(m)
        # Available fraction of allocated capacity at generator i   in scenario w
        self.f = np.ones((m, scen))
        # Demand upper bound
        self.d_ub = np.max(self.d)
        # Demand worst-case point
        self.d_wc = self.d_ub * np.ones(n)
        
        # out of sample demand
        np.random.seed(6000)
        self.d_out_of_sample = np.zeros((n, 10_000))
        for w in range(10_000):
            self.d_out_of_sample[:, w] = np.round(np.exp(np.random.multivariate_normal(mean_vec, cov_mat, size=1)), 1)
        # for w in range(10_000):
        #     for j in self.J:
        #         self.d_out_of_sample[j, w] = np.round(np.exp(np.random.normal(1, 1**2)), 1)
    
    def l1_norm_dist(self, w1, w2):
        return np.linalg.norm(self.d[:, w1] - self.d[:, w2], ord=1)
    
    def l2_norm_dist(self, w1, w2):
        return np.linalg.norm(self.d[:, w1] - self.d[:, w2], ord=2)
    
    def l2_norm_squared(self, w1, w2):
        return np.linalg.norm(self.d[:, w1] - self.d[:, w2], ord=2)**2
    
    def expand_scenario_list(self):
        #raise 'Deprecated'
        n = len(self.J)
        N = len(self.Omega)
        self.Omega.append(N)
        self.d = np.hstack((self.d, self.d_wc.reshape(n, 1)))


def L_norm(v1, v2, ord=1):
    return np.linalg.norm(v1 - v2, ord=ord)


def L_norm_squared(v1, v2, ord=1):
    return 0.5 * np.linalg.norm(v1 - v2, ord=ord)**2
