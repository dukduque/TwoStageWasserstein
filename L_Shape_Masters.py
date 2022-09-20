'''
Created on Apr 11, 2017

@author: dduque
'''
from gurobipy import GRB, Model, quicksum
from IO_handler import L_norm, L_norm_squared
from collections import defaultdict
import numpy as np
# from gurobipy.gurobipy import quicksum
'''
=========================================================================================
IMPLEMENTATION OF MASTER PROBLEM FOR L_SHAPE METHOD
Problem a:  Generic Restricted Master Problem
=========================================================================================
'''


class GenericMasterProblem:
    '''
    This class implements a general version of the relaxed master problem
    for the capacity allocation problem. This master problem determines
    the allocation decision for each generator that is later used by the 
    subproblems. 
    '''
    def __init__(self, data):
        self.data = data
        self.model = Model('GenericMasterProblem')  #Create a model
        self.model.params.outputflag = 0
        self.x = None
        self.theta = None
        self.cx = None
        self.optCuts = []  #Initialize an empty list of cuts
    
    def buildModel(self):
        '''Set the problem orientation'''
        self.model.modelSense = GRB.MINIMIZE
        '''Create decision variables'''
        self.x = self.model.addVars(self.data.I, vtype=GRB.CONTINUOUS, lb=0, ub=self.data.b, obj=self.data.k, name='x')
        self.theta = self.model.addVar(lb=0, obj=1, vtype=GRB.CONTINUOUS, name='theta')
        self.cx = quicksum(self.data.k[i] * self.x[i] for i in self.data.I)
        self.model.update()
        '''
        Add maximum installation capacity constraint
        '''
        self.model.addConstr(sum(self.x[i] for i in self.data.I), GRB.LESS_EQUAL, self.data.b, 'globalCap')
        self.model.update()
    
    def solve(self):
        '''
        Solves the master problem to get a solution
        of the first stage problem
        '''
        self.model.optimize()
        status = self.model.status
        if status == GRB.OPTIMAL:
            #Save x
            x_hat = np.zeros(len(self.data.I))
            for i in self.data.I:
                x_hat[i] = self.x[i].x
            
            #\lambda^T*A
            lagr_dual_vect = (self.model.getConstrByName('globalCap').Pi) * np.ones(len(self.x))
            objval = self.model.ObjVal
            theta_hat = self.theta.x
            return status, objval, x_hat, theta_hat, lagr_dual_vect
        else:
            raise 'Infeasible or unbounded Master'
    
    def getValue_cx(self):
        '''
        Gets the numerical value of the first stage
        objective function, i.e., cx (where cx is 
        a linear expression).
        '''
        return self.cx.getValue()
    
    def getValue_theta(self):
        return self.theta.x
    
    def resetModel(self):
        '''
        Remove all optimality cuts
        '''
        for ctr in self.optCuts:
            self.model.remove(ctr)
        self.model.update()
        self.optCuts = []
    
    def addOptCut(self, dualCapEP, dualRecEP):
        '''
        Adds an optimality cut to the master problem.
        
        ==================
        Input parameters
        ==================
        dualCapEP:
            Dual extreme point from the subproblem related to the capacity
            constraints. The structure of this parameter is a LIST of dual
            vectors, where each dual vector corresponds to a subproblem. 
        
        dualRecEP:
            Dual extreme point from the subproblem related to the recourse
            (subcontracting) constraints. The structure of this parameter 
            is a LIST of dual vectors, where each dual vector corresponds 
            to a subproblem. 
        '''
        
        #Retrieve the number of cuts
        cutNumber = len(self.optCuts)
        
        #Create the cut and add it to the model
        
        G_i = quicksum(self.data.prob[w] * dualCapEP[w][i] * 1 * self.x[i] for i in self.data.I
                       for w in self.data.Omega)
        g_i = quicksum(self.data.prob[w] * dualRecEP[w][j] * self.data.d[j, w] for j in self.data.J
                       for w in self.data.Omega)
        mynewCut = self.model.addConstr(G_i + g_i, GRB.LESS_EQUAL, self.theta, ('cut' + str(cutNumber)))
        self.model.update()
        # Store the cut
        self.optCuts.append(mynewCut)
        return [sum(self.data.prob[w] * dualCapEP[w][i] * 1 for w in self.data.Omega)
                for i in self.data.I], sum(self.data.prob[w] * dualRecEP[w][j] * self.data.d[j, w] for j in self.data.J
                                           for w in self.data.Omega)
    
    def setRegularizedMasterObjFun(self, sigma, x_bar, decType):
        '''
        Sets the RD objective function, i.e., adds the proximal
        term in addition to the original objective function. 
        
        This method is a building block of the original master 
        problem. This means that the master problem feasible 
        region and cutting plane scheme remain unchanged, and the
        objective is set such that regularized term is used.
    
        
        ================
        Input parameters
        ================
        sigma:
            The adaptive parameter of the RD. The objective function
            of the master problem is: min cx+ theta + sigma/2 (||x-x_bar||^2).
            Note than when sigma = 0, this objective function yields to the
            original master problem.
        
        x_bar:
            Is the current value of the first stage variable where the proximal
            term is centered (see explanation of sigma). 
        decType:
            Type of decomposition.
            0: L_Shape decomposition
            1: Regularized decomposition
            2: Level-set decomposition
        '''
        
        #Build objective function according to the require decomposition
        RD_obj = None
        if decType == 0:
            RD_obj = quicksum((self.data.k[i] * self.x[i]) for i in self.data.I) + self.theta
        elif decType == 1:
            RD_obj = quicksum(
                (self.data.k[i] * self.x[i] + (sigma / 2) * (self.x[i] - x_bar[i]) * (self.x[i] - x_bar[i]))
                for i in self.data.I) + self.theta
        elif decType == 2:
            RD_obj = quicksum(((self.x[i] - x_bar[i]) * (self.x[i] - x_bar[i])) for i in self.data.I)
        else:
            raise 'Unknown type of decomposition at @GenericMasterProblem.setRegularizedMasterObjFun(.)'
        #Set the obj function
        self.model.setObjective(RD_obj, GRB.MINIMIZE)
        #Update the model
        self.model.update()
    
    def modifyUpperBoundTheta(self, ub):
        '''
        Modifies the upper bound of theta for the
        level-set implementation.
        '''
        self.theta.UB = ub
        self.model.update()


class DROMasterProblem:
    '''
    This class implements a the relaxed master problem for the DRO-Wasserstein
    capacity allocation problem. This master problem determines
    the allocation decision for each generator that is later used by the
    subproblems.
    '''
    def __init__(self, data, dro_r, discrete_wasserstein, add_scenario, dist_type):
        self.data = data
        self.dro_r = dro_r
        self.discrete_wasserstein = discrete_wasserstein
        self.distance_type = dist_type
        self.add_scenario = add_scenario
        self.model = Model('DROMasterProblem')  # Create a model
        self.model.params.outputflag = 0
        
        self.x = None
        self.theta = None
        self.lambda_dro = None
        self.cx = None
        self.optCuts = defaultdict(int)
        self.Scenarios = []
        
        self.lambda_underbar = 0
        if dist_type == 'optimal_transport' and dro_r > 0:
            #self.lambda_underbar = np.sqrt(len(data.J)) / np.sqrt(4 * dro_r * data.n_sce)
            self.lambda_underbar = 1 / np.sqrt(dro_r)
    
    def buildModel(self):
        '''Set the problem orientation'''
        self.model.modelSense = GRB.MINIMIZE
        '''Create decision variables'''
        self.x = self.model.addVars(self.data.I, vtype=GRB.CONTINUOUS, lb=0, ub=self.data.b, obj=0, name='x')
        self.model.update()
        '''
        Add maximum installation capacity constraint
        '''
        self.model.addConstr(sum(self.x[i] for i in self.data.I), GRB.LESS_EQUAL, self.data.b, 'globalCap')
        self.model.update()
        '''Create DRO variables'''
        self.Scenarios = {w: self.data.d[:, w] for w in self.data.Omega}
        self.theta = self.model.addVars(self.data.Omega, lb=-1E8, obj=0, vtype=GRB.CONTINUOUS, name='theta')
        self.lambda_dro = self.model.addVar(lb=self.lambda_underbar, obj=0, vtype=GRB.CONTINUOUS, name='lambda')
        self.cx = quicksum(self.data.k[i] * self.x[i] for i in self.data.I)
        self.model.update()
        '''Create DRO constraints according to model'''
        
        c = None
        if self.distance_type == 'wasserstein':
            c = self.data.l1_norm_dist
        elif self.distance_type == 'optimal_transport':
            c = self.data.l2_norm_squared
        
        self.nu = self.model.addVars(self.data.n_sce, lb=-1E0, obj=0, vtype=GRB.CONTINUOUS, name='nu')
        self.dro_ctrs = self.model.addConstrs(
            (c(w, o) * self.lambda_dro + self.nu[w] >= self.theta[o] for w in self.nu for o in self.theta), 'dro_ctrs')
        obj_fun = self.cx + self.dro_r * self.lambda_dro + (1 / len(self.nu)) * quicksum(self.nu)
        self.model.setObjective(obj_fun, GRB.MINIMIZE)
        
        # Om = self.data.Omega
        # N_scenarios = len(Om)
        
        # if self.discrete_wasserstein:
        #     '''
        #         Formulation with discreate support
        #     '''
        #     l1 = self.data.l1_norm_dist
        #     self.nu = self.model.addVars(self.data.n_sce, lb=0, obj=0, vtype=GRB.CONTINUOUS, name='nu')
        #     self.model.addConstrs(
        #         (l1(w, o) * self.lambda_dro + self.nu[w] >= self.theta[o] for w in self.nu for o in Om), 'dro_ctrs')
        #     obj_fun = self.cx + self.dro_r * self.lambda_dro + (1 / self.data.n_sce) * quicksum(self.nu)
        #     self.model.setObjective(obj_fun, GRB.MINIMIZE)
        # else:  # Continuous Wasserstein
        #     assert not self.add_scenario, "Cant extend scenario list for cont' Wasserstein"
        #     obj_fun = self.cx + self.dro_r * self.lambda_dro + (1 / self.data.n_sce) * quicksum(self.theta)
        #     self.model.setObjective(obj_fun, GRB.MINIMIZE)
    
    def update_lambda_underbar(self):
        if self.lambda_underbar == 0:
            return False
        
        lambda_hat = self.lambda_dro.X
        if np.abs(lambda_hat - self.lambda_underbar) < 1E-8:
            self.lambda_underbar = 0.5 * self.lambda_underbar
            self.lambda_dro.lb = self.lambda_underbar
            return True
        return False
    
    def solve(self):
        '''
        Solves the master problem to get a solution
        of the first stage problem
        '''
        self.model.optimize()
        status = self.model.status
        if status == GRB.OPTIMAL:
            x_hat = np.zeros(len(self.data.I))
            for i in self.data.I:
                x_hat[i] = np.maximum(0, self.x[i].x)
            
            objval = self.model.ObjVal
            theta_hat = [self.theta[w].x for w in self.theta]
            lambda_hat = self.lambda_dro.x
            return status, objval, x_hat, lambda_hat, theta_hat
        else:
            raise 'Infeasible or unbounded Master'
    
    def getValue_cx(self):
        '''
        Gets the numerical value of the first stage
        objective function, i.e., cx (where cx is
        a linear expression).
        '''
        return self.cx.getValue()
    
    def getValue_DRO(self):
        return self.dro_r * self.lambda_dro.X
    
    def getValue_theta(self, w):
        return self.theta[w].x
    
    def add_support_point(self, new_xi):
        '''
        Args:
            new_xi (ndarray): vector with the new realization
        '''
        new_scenario = True
        existing_ix = -1
        for (w, d_w) in self.Scenarios.items():
            if L_norm(d_w, new_xi, 2) < 1E-5:
                new_scenario = False
                existing_ix = w
                break
        
        if new_scenario:
            # print(np.round(new_xi[:], 4))
            # Add scenario, create variable, and add corresponding DRO constraints.
            j = len(self.Scenarios)
            self.Scenarios[j] = new_xi
            self.theta[j] = self.model.addVar(lb=-1E8, obj=0, vtype=GRB.CONTINUOUS, name=f'theta[{j}]')
            for w in self.nu:  # For loop over original support points
                if self.distance_type == 'wasserstein':
                    norm_w_j = L_norm(self.Scenarios[w], new_xi)
                elif self.distance_type == 'optimal_transport':
                    norm_w_j = L_norm_squared(self.Scenarios[w], new_xi, 2)
                self.dro_ctrs[w, j] = self.model.addConstr((norm_w_j * self.lambda_dro + self.nu[w] >= self.theta[j]),
                                                           f'dro_ctrs[{w},{j}]]')
            return True, j
        else:
            return False, existing_ix
    
    def addOptCut(self, w, grad, intercept, force_cut=False):
        '''
        Adds an optimality cut to the master problem.
        
        ==================
        Input parameters
        ==================
        grads:
            Dual extreme point from the subproblem related to the capacity
            constraints. The structure of this parameter is a LIST of dual
            vectors, where each dual vector corresponds to a subproblem.
        
        intercepts:
            Value of the intercepts for each scenario, directly computed from
            the dro subproblem.
        
        l1_norm_coeffs:
            Values for the l1-norm distance between demand vector for scenario
            w and the corresponding worst-case probability distribution.
        '''
        cut_exp = self.theta[w] - quicksum(grad[i] * 1 * self.x[i] for i in self.data.I) - intercept
        try:
            if cut_exp.getValue() < -1E-6:
                self.model.addConstr(cut_exp, GRB.GREATER_EQUAL, 0, f'cut_{w}_{self.optCuts[w]}')
                self.optCuts[w] += 1
                return True
        except AttributeError as grbAttErr:
            self.model.addConstr(cut_exp, GRB.GREATER_EQUAL, 0, f'cut_{w}_{self.optCuts[w]}')
            self.optCuts[w] += 1
            return True
        
        return False
        
        # if self.discrete_wasserstein:
        #     # Create the cut without the term for lambda as it was already
        #     # included in the constraints
        #     for w in self.theta:
        #         cut_exp = quicksum(dualCapEP[w][i] * 1 * self.x[i] for i in self.data.I)
        #         # cut_exp = cut_exp - self.lambda_dro * l1_norm_coeffs[w]
        #         cut_exp = cut_exp + intercepts[w]
        #         self.model.addConstr(cut_exp, GRB.LESS_EQUAL, self.theta[w], f'cut_{w}_{self.optCuts}')
        #     self.model.update()
        
        # else:
        #     #Create the cut and add it to the model
        #     for w in self.theta:
        #         cut_exp = quicksum(dualCapEP[w][i] * 1 * self.x[i] for i in self.data.I)
        #         cut_exp = cut_exp - self.lambda_dro * l1_norm_coeffs[w]
        #         cut_exp = cut_exp + intercepts[w]
        #         self.model.addConstr(cut_exp, GRB.LESS_EQUAL, self.theta[w], f'cut_{w}_{self.optCuts}')
        #     self.model.update()
        
        # self.optCuts += 1
        
        # return None, None
