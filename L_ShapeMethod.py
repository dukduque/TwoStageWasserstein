'''
Created on Apr 11, 2017

@author: Daniel Duque
'''
import os
print(os.getcwd())
print(os.environ['PYTHONPATH'].split(os.pathsep))
import numpy as np
from L_Shape_Masters import GenericMasterProblem, DROMasterProblem
from L_Shape_Subproblem import CA_Subproblem, CA_DualSub, CA_DualSub_DRO, CA_DualSub_DRO_L2norm, CA_DualSub_DRO_MIP, CA_DualSub_DRO_McCormick
import time
from sqlalchemy.sql.expression import false

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pkginfo.commandline import INI

from gurobipy import Model, quicksum, GRB

from Utils.file_savers import write_object_results
from OutputAnalysis.SimulationAnalysis import SimResult


class L_Shape:
    '''
    This class implements multiple decomposition algorithm for two stage stochastic programs.
    In particular:
    L Shape:
        
    Regularized Decomposition:
    
    Level-Set Decomposition:
    
    Gradient descent:
    '''
    def init_options(self):
        '''
        Initialize algorithm options for L_shape with default values
        
        ==============
        Output values:
        ==============
    
        options:
        This is a structure with field that correspond to algorithmic
        options of the method.  In particular:

        max_iter:   
            Maximum number of iterations
        opt_tol:        
            Numeric tolerance for optimality
        float_tol:
            Numeric tolerance for zero
        output_level:
            Amount of output printed
            0: No output
            1: Only summary information
            2: Summary of each iteration
            3: Detailed output per iteration
        output_lines:
            When the output level is set to 2 or more, the option
            establishes how often (iterations) is the output going 
            to be printed.
        sigma:
            Initial value of the proximal term in RD
        sigma_max:
            Max value for sigma in RD
        sigma_min:
            Min value for sigma in RD

        lambda:
            Parameter for the level set decomposition
        to:
            Initial step size for gradient descent
        '''
        
        options = {}
        options['max_iter'] = 50000
        options['opt_tol'] = 1E-4
        options['float_tol'] = 1E-8
        options['output_level'] = 1
        options['output_lines'] = 1
        options['sigma'] = 1
        options['sigma_max'] = 1E4
        options['sigma_min'] = 1E-4
        options['lambda'] = 0.5
        options['t0'] = 1.0
        
        return options
    
    def __init__(self, data):
        '''
        Initialize the algorithm
        
        ==================
        Input parameters
        ==================
        data: an instance of Datahandler 
        '''
        self.data = data
        self.UB = None
        self.LB = None
        self.options = self.init_options()
    
    def printFinalSolution(self, x_star):
        'Print detailed solution'
        str_sol = ''
        for i in self.data.I:
            if x_star[i] > self.options['float_tol']:
                str_sol = str_sol + ' -> ' + str(i)
        print('Open facilities: ', str_sol)
    
    def solveProblem_a(self, probName):
        '''
        Runs a generic L_shape method
        '''
        '''
        ======================================================
        Initialization for all decomposition approaches
        '''
        TNOW = time.time()
        self.UB = np.inf  #Upper bound
        self.LB = np.inf  #Lower bound
        gap = 1  #Relative Gap
        x_star = None  #Data structure to store optimal solution
        #Instance of the master problem
        master = GenericMasterProblem(self.data)
        master.buildModel()
        #Instance of a single subproblem
        subProblem = CA_Subproblem(self.data, 0, np.zeros(len(self.data.I)))
        subProblem.buildModel()
        iteration = 1  #Iteration counter
        if self.options['output_level'] >= 2:
            # Prepare header for output
            output_header = '%5s %10s %10s %10s' % ('Iter', 'LB', 'UB', 'Gap')
        '''
        ======================================================
        Beginning of the main loop
        '''
        while True:
            '''Solve master problem to get x_hat'''
            [status, objval, x_hat, theta_hat, _] = master.solve()
            self.LB = objval  #Update Lower Bound
            '''Update x_hat in the subproblem, for scenario 0'''
            subProblem.updateX(x_hat, 0)
            '''Data structure to store a dual extreme point'''
            dualCapEP = []  #List of duals vectors (EP) of the  capacity constraint
            dualRecEP = []  #List of duals vectors (EP) of the  recourse constraint
            
            z_hat = master.getValue_cx()  #Current obj. value
            '''Solve subproblems'''
            for w in self.data.Omega:
                subProblem.updateForScenario(w)
                [type, capDual, recDual, status, objval] = subProblem.solve()
                assert type == 'EP'
                dualCapEP.append(np.copy(capDual))
                dualRecEP.append(np.copy(recDual))
                z_hat = z_hat + self.data.prob[w] * objval
            '''Update upper bound'''
            if z_hat < self.UB:
                self.UB = z_hat
                x_star = np.copy(x_hat)
            '''Update gap and check for termination '''
            gap = (self.UB - self.LB + 0.0) / np.minimum(np.abs(self.UB), np.abs(self.LB) + self.options['float_tol'])
            #print(gap)
            if gap <= self.options['opt_tol']:
                #TODO: Report solution
                if self.options['output_level'] >= 2:
                    print('%5i %10.2f %10.2f %10.5f%%' % (iteration, self.LB, self.UB, np.minimum(1, gap) * 100))
                    if self.options['output_level'] >= 3:
                        self.printFinalSolution(x_star)
                break
            '''Output options'''
            if self.options['output_level'] == 2:
                lines = self.options['output_lines']
                if (iteration - 1) % (10 * lines) == 0:
                    # Prints the output header every 10 iterations
                    print(output_header)
                if (iteration - 1) % lines == 0:
                    print('%5i %10.2f %10.2f %10.5f%%' % (iteration, self.LB, self.UB, np.minimum(1, gap) * 100))
            '''Add cut if the tolerance criterion wasn't met'''
            master.addOptCut(dualCapEP, dualRecEP)
            
            iteration += 1  #Update iterator
            if iteration >= self.options['max_iter']:
                break
        
        if self.options['output_level'] >= 1:
            print(
                probName, ' %5i %10.2f %10.2f %10.5f%% %10.2f' %
                (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, (time.time() - TNOW)))
    
    def solveProblem_b(self, probName):
        '''
        Runs the Regularized Decomposition method
        by Ruszczyuski 86
        '''
        '''
        ======================================================
        Initialization for all decomposition approaches
        '''
        TNOW = time.time()
        self.UB = np.inf  #Upper bound
        self.LB = np.inf  #Lower bound
        gap = 1  #Relative Gap
        x_star = None  #Data structure to store optimal solution
        #Instance of the master problem
        master = GenericMasterProblem(self.data)
        master.buildModel()
        #Instance of a single subproblem
        subProblem = CA_Subproblem(self.data, 0, np.zeros(len(self.data.I)))
        subProblem.buildModel()
        iteration = 1  #Iteration counter
        if self.options['output_level'] >= 2:
            # Prepare header for output
            output_header = '%5s %10s %10s %10s %10s' % ('Iter', 'LB', 'UB', 'Gap', 'sigma')
        '''
        ======================================================
        Initialization for RD
        '''
        # Regularized Decomposition parameters
        sigma = self.options['sigma']
        cx_hat = 0
        hx_hat = 0
        #Step 0: get x_bar and compute h(x_bar)
        [status, objval, x_hat, theta_hat] = master.solve()
        x_bar = x_hat
        new_x_bar = False
        cx_bar = master.getValue_cx()
        #Solve h(x_bar):
        subProblem.updateX(x_bar, 0)  #update subproblem
        hx_bar = 0
        for w in self.data.Omega:
            subProblem.updateForScenario(w)
            [type, capDual, recDual, status, subobjval] = subProblem.solve()
            hx_bar = hx_bar + self.data.prob[w] * subobjval
        self.UB = cx_bar + hx_bar
        '''
        ======================================================
        Beginning of the main loop
        '''
        while True:
            '''Setup and solve original master to get a LB'''
            master.setRegularizedMasterObjFun(sigma, x_bar, 0)  #0=L-Shape
            master.modifyUpperBoundTheta(np.inf)
            [status_ls, objval_ls, x_ls, theta_ls] = master.solve()
            self.LB = objval_ls
            '''Update gap and check for termination'''
            gap = (self.UB - self.LB + 0.0) / np.minimum(np.abs(self.UB), np.abs(self.LB) + self.options['float_tol'])
            if gap <= self.options['opt_tol']:
                if self.options['output_level'] >= 2:
                    print('%5i %10.2f %10.2f %10.5f%% %10.2e' %
                          (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, sigma))
                    if self.options['output_level'] >= 3:
                        self.printFinalSolution(x_star)
                break
            '''Setup and solve Regularized Master problem'''
            master.setRegularizedMasterObjFun(sigma, x_bar, 1)  #1=RD
            [status, objval, x_hat, theta_hat] = master.solve()
            cx_hat = master.getValue_cx()
            '''Update x_hat in the subproblem, for scenario 0'''
            subProblem.updateX(x_hat, 0)
            '''Data structure to store a dual extreme point'''
            dualCapEP = []  #List of duals vectors (EP) of the  capacity constraint
            dualRecEP = []  #List of duals vectors (EP) of the  recourse constraint
            '''Solve subproblems'''
            cx_hat = master.getValue_cx()
            hx_hat = 0
            for w in self.data.Omega:
                subProblem.updateForScenario(w)
                [type, capDual, recDual, status, subobjval] = subProblem.solve()
                assert type == 'EP'
                dualCapEP.append(np.copy(capDual))
                dualRecEP.append(np.copy(recDual))
                hx_hat = hx_hat + self.data.prob[w] * subobjval
            '''Add cut if theta_hat<=h(x_hat)'''
            if master.getValue_theta() - hx_bar < 0:
                master.addOptCut(dualCapEP, dualRecEP)
            '''Update incumbent'''
            if np.abs(theta_hat - hx_bar) <= self.options['float_tol']:
                x_bar = np.copy(x_hat)
                new_x_bar = True
            elif (cx_hat + hx_hat) - (cx_bar + hx_bar) <= 0:
                x_bar = np.copy(x_hat)
                new_x_bar = True
            else:
                new_x_bar = False
            '''Update RD parameter sigma'''
            if hx_hat <= (1 + self.options['opt_tol']) * (theta_hat):
                sigma = np.maximum(0.5 * sigma, self.options['sigma_min'])
            elif hx_hat >= (1 + self.options['opt_tol']) * (theta_hat):
                sigma = np.minimum(2 * sigma, self.options['sigma_max'])
            '''Update cx_hat and h(x_hat) if x_bar is updated'''
            if new_x_bar:
                cx_bar = cx_hat
                hx_bar = hx_hat
                self.UB = cx_bar + hx_bar
            
            if self.options['output_level'] == 2:
                lines = self.options['output_lines']
                if (iteration - 1) % (10 * lines) == 0:
                    print(output_header)
                if (iteration - 1) % lines == 0:
                    print('%5i %10.2f %10.2f %10.5f%% %10.2e' %
                          (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, sigma))
            
            iteration += 1  #Update iterator
            if iteration >= self.options['max_iter']:
                break
        if self.options['output_level'] >= 1:
            print(
                probName, ' %5i %10.2f %10.2f %10.5f%%  %10.2f' %
                (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, (time.time() - TNOW)))
    
    def solveProblem_c(self, probName):
        '''
            Runs the Level-Set DEcomposition method
            by Lemarechal, Nemirovski, and Nesterov
        '''
        '''
        ======================================================
        Initialization for all decomposition approaches
        '''
        TNOW = time.time()
        self.UB = np.inf  #Upper bound
        self.LB = np.inf  #Lower bound
        gap = 1  #Relative Gap
        x_star = None  #Data structure to store optimal solution
        #Instance of the master problem
        master = GenericMasterProblem(self.data)
        master.buildModel()
        #Instance of a single subproblem
        subProblem = CA_Subproblem(self.data, 0, np.zeros(len(self.data.I)))
        subProblem.buildModel()
        iteration = 1  #Iteration counter
        if self.options['output_level'] >= 2:
            # Prepare header for output
            output_header = '%5s %10s %10s %10s %10s' % ('Iter', 'LB', 'UB', 'Gap', 'Theta_UB')
        '''
        ======================================================
        Initialization for RD
        '''
        # Regularized Decomposition parameter
        sigma = self.options['sigma']
        #Step 0: get x_bar and compute h(x_bar)
        [status, objval, x_hat, theta_hat] = master.solve()
        x_bar = x_hat
        new_x_bar = False
        cx_bar = master.getValue_cx()
        #Solve h(x_bar):
        subProblem.updateX(x_bar, 0)  #update subproblem
        hx_bar = 0
        for w in self.data.Omega:
            subProblem.updateForScenario(w)
            [type, capDual, recDual, status, subobjval] = subProblem.solve()
            hx_bar = hx_bar + self.data.prob[w] * subobjval
        self.UB = cx_bar + hx_bar
        hx_hat = 0
        theta_UB = 0
        '''
        ======================================================
        Beginning of the main loop
        '''
        while True:
            '''Setup original master to get a LB for theta'''
            master.setRegularizedMasterObjFun(None, None, 0)  #0=L-Shape
            master.modifyUpperBoundTheta(np.inf)
            [status_ls, objval_ls, x_ls, theta_ls] = master.solve()
            self.LB = objval_ls
            '''Update gap and check for termination '''
            gap = (self.UB - self.LB + 0.0) / np.minimum(np.abs(self.UB), np.abs(self.LB) + self.options['float_tol'])
            #print(gap)
            if gap <= self.options['opt_tol']:
                #TODO: Report solution
                if self.options['output_level'] >= 2:
                    print('%5i %10.2f %10.2f %10.5f%% %10.2f' %
                          (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, theta_UB))
                    if self.options['output_level'] >= 3:
                        self.printFinalSolution(x_star)
                break
            '''Setup leve-set master problem'''
            theta_UB = self.options['lambda'] * hx_bar + (1 - self.options['lambda']) * theta_ls
            master.setRegularizedMasterObjFun(None, x_bar, 2)  #2=Level-set
            master.modifyUpperBoundTheta(theta_UB)
            '''Solve leve-set master problem to get x_hat'''
            [status, objval, x_hat, theta_hat] = master.solve()
            cx_hat = master.getValue_cx()
            '''Update x_hat in the subproblem, for scenario 0'''
            subProblem.updateX(x_hat, 0)
            '''Data structure to store a dual extreme point'''
            dualCapEP = []  #List of duals vectors (EP) of the  capacity constraint
            dualRecEP = []  #List of duals vectors (EP) of the  recourse constraint
            
            cx_hat = master.getValue_cx()
            hx_hat = 0
            '''Solve subproblems'''
            for w in self.data.Omega:
                subProblem.updateForScenario(w)
                [type, capDual, recDual, status, subobjval] = subProblem.solve()
                assert type == 'EP'
                dualCapEP.append(np.copy(capDual))
                dualRecEP.append(np.copy(recDual))
                hx_hat = hx_hat + self.data.prob[w] * subobjval
            '''Add cut if theta_hat<=h(x_hat)
            Note: To be check if FIRST STAGE cost != 0 '''
            if master.getValue_theta() - hx_bar < 0:
                master.addOptCut(dualCapEP, dualRecEP)
            '''Update incumbent'''
            if (cx_hat + hx_hat) - (cx_bar + hx_bar) < 0:
                x_bar = np.copy(x_hat)
                cx_bar = cx_hat
                hx_bar = hx_hat
                self.UB = cx_bar + hx_bar
                new_x_bar = True
            else:
                new_x_bar = False
            
            if self.options['output_level'] == 2:
                lines = self.options['output_lines']
                if (iteration - 1) % (10 * lines) == 0:
                    # Prints the output header every 10 iterations
                    print(output_header)
                if (iteration - 1) % lines == 0:
                    print('%5i %10.2f %10.2f %10.5f%% %10.2f' %
                          (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, theta_UB))
            
            iteration += 1  #Update iterator
            if iteration >= self.options['max_iter']:
                break
        if self.options['output_level'] >= 1:
            print(
                probName, ' %5i %10.2f %10.2f %10.5f%%  %10.2f' %
                (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, (time.time() - TNOW)))
    
    def solveProblem_d(self, probName):
        '''
            Implements a gradient descent method defined as follows:
            Given x_k \ in X
            Let y_k+1 = x_k + t_k*G_k 
            Let x_k+1 = argmin{||x-y_k+1||^2 : x\in X}
            t_k = t_0/k
            
            As termination criterion, lower bounds are computed
            by mean of an L-Shape master problem.
            
        '''
        '''
        ======================================================
        Initialization for all decomposition approaches
        '''
        TNOW = time.time()
        self.UB = np.inf  #Upper bound
        self.LB = np.inf  #Lower bound
        gap = 1  #Relative Gap
        x_star = None  #Data structure to store optimal solution
        #Instance of the master problem
        master = GenericMasterProblem(self.data)
        master.buildModel()
        #Best know solution
        #Instance of a single subproblem
        subProblem = CA_Subproblem(self.data, 0, np.zeros(len(self.data.I)))
        subProblem.buildModel()
        iteration = 1  #Iteration counter
        if self.options['output_level'] >= 2:
            # Prepare header for output
            output_header = '%5s %10s %10s %10s %10s' % ('Iter', 'LB_k', 'UB_k', 'Gap', '||G_k||')
        '''
        ======================================================
        Initialization for Gradient descent
        '''
        n = len(self.data.I)
        x_k = np.zeros(n)  #Projected iterate
        y_k = np.zeros(n)  #Steepest decent iterate
        G_k = np.zeros(n)  #Gradient
        t_k = self.options['t0']  #Step
        G_k_norm = None  #Gradient norm
        '''
        ======================================================
        Beginning of the main loop
        (SUB)GRADIENT DESCENT
        '''
        while True:
            '''Setup original master to get a LB for theta'''
            master.setRegularizedMasterObjFun(None, None, 0)  #0=L-Shape
            master.modifyUpperBoundTheta(np.inf)
            [status_ls, objval_ls, x_ls, theta_ls] = master.solve()
            self.LB = objval_ls
            '''Update gap and check for termination '''
            gap = (self.UB - self.LB + 0.0) / np.minimum(np.abs(self.UB), np.abs(self.LB) + self.options['float_tol'])
            #print(gap)
            if gap <= self.options['opt_tol']:
                if self.options['output_level'] >= 2:
                    print('%5i %10.2f %10.2f %10.5f%% %10.2e ' %
                          (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, G_k_norm))
                    if self.options['output_level'] >= 3:
                        self.printFinalSolution(x_star)
                break
            '''
            =============================================================
            Compute a subgradient at x_k
                1. Update x_k in the subproblem, for scenario 0
                2. Data structure to store a dual extreme point
                3. Solve subproblems
                4. Compute G_k
            '''
            #1
            subProblem.updateX(x_k, 0)
            #2
            dualCapEP = []  #List of duals vectors (EP) of the  capacity constraint
            dualRecEP = []  #List of duals vectors (EP) of the  recourse constraint
            cx_k = self.data.k.dot(x_k)
            hx_k = 0
            #3
            for w in self.data.Omega:
                subProblem.updateForScenario(w)
                [type, capDual, recDual, status, subobjval] = subProblem.solve()
                assert type == 'EP'
                dualCapEP.append(np.copy(capDual))
                dualRecEP.append(np.copy(recDual))
                hx_k = hx_k + self.data.prob[w] * subobjval
            zx_k = cx_k + hx_k
            if zx_k < self.UB:
                self.UB = zx_k
                x_star = np.copy(x_k)
            #4
            for i in self.data.I:
                G_k[i] = 0
                for w in self.data.Omega:
                    G_k[i] += self.data.prob[w] * dualCapEP[w][i] * self.data.f[i, w]
            '''Second termination check (By Subgradient Norm)
                Recall: if subgradient 0 in D_xE[h(x, xi)], we are in a local minimizer
                if E[h(x, xi)] is strongly convex, is a global minimizer.
                In particular, if the subgradient is zero, the algorithm need to stop
            '''
            G_k_norm = np.linalg.norm(G_k)
            if G_k_norm <= self.options['float_tol']:
                x_star = x_k
                if self.options['output_level'] >= 2:
                    print('%5i %10.2f %10.2f %10.5f%% %10.2e ' %
                          (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, G_k_norm))
                    if self.options['output_level'] >= 3:
                        self.printFinalSolution(x_star)
                break
            '''Add cut if the tolerance criteria wasn't met:
                Criterion 1: gap between lower and upper bound
                Criterion 2: norm of G_k, the l_shape subgradient'''
            master.addOptCut(dualCapEP, dualRecEP)
            '''Update iterates'''
            y_k = np.maximum(0, x_k - t_k * G_k)
            LHS = sum(y_k)
            RHS = self.data.b
            if LHS <= RHS:
                #Gradient step is feasible
                x_k = y_k
            else:
                #Gradient step is infeasible and minimum distance projection onto X is given by:
                x_k = y_k + (RHS - LHS) / len(y_k)
            
            if self.options['output_level'] >= 2:
                lines = self.options['output_lines']
                if (iteration - 1) % (10 * lines) == 0:
                    # Prints the output header every 10 iterations
                    print(output_header)
                if (iteration - 1) % lines == 0:
                    print('%5i %10.2f %10.2f %10.5f%% %10.2e' %
                          (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, G_k_norm))
            
            #Update iterator and t_k
            iteration += 1  #Update iterator
            
            t_k = (1000 * self.options['t0'] + 0.0) / (np.power(iteration, 0.5))
            if iteration >= self.options['max_iter']:
                break
        if self.options['output_level'] >= 1:
            print(
                probName, ' %5i %10.2f %10.2f %10.5f%% %10.2f' %
                (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, (time.time() - TNOW)))
    
    def solveProblem_d_BFGS(self, problemName):
        '''
            Implements a BFGS method as Algorithm 6.1 in Nocedal and Wright 2006, Numerical Optimization:
            
            As termination criterion, lower bounds are computed
            by mean of an L-Shape master problem.
            
        '''
        '''
        ======================================================
        Initialization for all decomposition approaches
        '''
        TNOW = time.time()
        self.UB = np.inf  #Upper bound
        self.LB = np.inf  #Lower bound
        gap = 1  #Relative Gap
        x_star = None  #Data structure to store optimal solution
        
        #Instance of the master problem
        master = GenericMasterProblem(self.data)
        master.buildModel()
        #Best know solution
        #Instance of a single subproblem
        subProblem = CA_Subproblem(self.data, 0, np.zeros(len(self.data.I)))
        subProblem.buildModel()
        '''
        ==================================
        Setup the proposed modification'''
        subProblem.lastMinuteChangeSetUp()
        temp_tol = self.options['opt_tol']
        self.options['opt_tol'] = 1E-2
        lastMinChange = False
        '''
                End the set up
        =================================='''
        
        iteration = 1  #Iteration counter
        if self.options['output_level'] >= 2:
            # Prepare header for output
            output_header = '%5s %10s %10s %10s %10s' % ('Iter', 'LB_k', 'UB_k', 'Gap', '||G_k||')
        '''
        ======================================================
        Initialization for Gradient descent
        '''
        n = len(self.data.I)
        I = np.identity(n)
        x_k = np.zeros(n)  #Projected iterate
        x_k_last = np.zeros(n)  #Last projected iterate
        y_k = np.zeros(n)  #Steepest decent iterate
        G_k_last = np.zeros(n)  #last iteration gradient
        G_k = np.zeros(n)  #Gradient
        t_k = 1  #self.options['t0']     #Step
        G_k_norm = None  #Gradient norm
        H_k = np.identity(n) * n  #Hessian approximation
        gamma_k = 10
        '''
        ======================================================
        Beginning of the main loop
        (SUB)GRADIENT DESCENT
        '''
        while True:
            '''Setup original master to get a LB for theta'''
            master.setRegularizedMasterObjFun(None, None, 0)  #0=L-Shape
            master.modifyUpperBoundTheta(np.inf)
            [status_ls, objval_ls, x_ls, theta_ls] = master.solve()
            self.LB = objval_ls
            '''Update gap and check for termination '''
            gap = (self.UB - self.LB + 0.0) / np.minimum(np.abs(self.UB), np.abs(self.LB) + self.options['float_tol'])
            #print(gap)
            if gap <= self.options['opt_tol'] or (lastMinChange == False and iteration > 100):
                
                if self.options['output_level'] >= 2:
                    print('%5i %10.2f %10.2f %10.2e %10.5f%%' %
                          (iteration, self.LB, self.UB, G_k_norm, np.minimum(1, gap) * 100))
                    if self.options['output_level'] >= 3:
                        self.printFinalSolution(x_star)
                if lastMinChange:
                    break
                else:
                    lastMinChange = True
                    iteration = 2
                    self.options['opt_tol'] = temp_tol
                    subProblem.lastMinuteChange()
                    master.resetModel()
            '''
            =============================================================
            Compute a subgradient at x_k
                1. Update x_k in the subproblem, for scenario 0
                2. Data structure to store a dual extreme point
                3. Solve subproblems
                4. Compute G_k
            '''
            #1
            subProblem.updateX(x_k, 0)
            #2
            dualCapEP = []  #List of duals vectors (EP) of the  capacity constraint
            dualRecEP = []  #List of duals vectors (EP) of the  recourse constraint
            #3
            cx_k = self.data.k.dot(x_k)
            hx_k = 0
            for w in self.data.Omega:
                subProblem.updateForScenario(w)
                [type, capDual, recDual, status, subobjval] = subProblem.solve()
                assert type == 'EP'
                dualCapEP.append(np.copy(capDual))
                dualRecEP.append(np.copy(recDual))
                hx_k = hx_k + self.data.prob[w] * subobjval
            zx_k = cx_k + hx_k
            if zx_k < self.UB:
                self.UB = zx_k
                x_star = np.copy(x_k)
            #4
            G_k_last = np.copy(G_k)
            for i in self.data.I:
                G_k[i] = 0
                for w in self.data.Omega:
                    G_k[i] += self.data.prob[w] * dualCapEP[w][i] * self.data.f[i, w]
            '''Second termination check (By Subgradient Norm)
                Recall: if subgradient 0 in D_xE[h(x, xi)], we are in a local minimizer
                if E[h(x, xi)] is strongly convex, is a global minimizer
            '''
            G_k_norm = np.linalg.norm(G_k)
            if G_k_norm <= self.options['float_tol']:
                x_star = x_k
                if self.options['output_level'] >= 2:
                    print('%5i %10.2f %10.2f %10.5f%% %10.2e ' %
                          (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, G_k_norm))
                    if self.options['output_level'] >= 3:
                        self.printFinalSolution(x_star)
                break
            '''Add cut if the tolerance criteria wasn't met:
                Criterion 1: gap between lower and upper bound
                Criterion 2: norm of G_k, the l_shape subgradient'''
            master.addOptCut(dualCapEP, dualRecEP)
            '''BFGS Update from k>1
            '''
            if iteration > 1:
                s_k = (x_k - x_k_last).reshape((n, 1))
                yy_k = (G_k - G_k_last).reshape((n, 1))
                if np.linalg.norm(s_k) > self.options['float_tol'] and np.linalg.norm(yy_k) > self.options['float_tol']:
                    rho_k = 1.0 / ((yy_k.transpose()).dot(s_k))
                    if rho_k < 0:
                        print('Curvature condition didnt hold')
                    H_k = ((I - rho_k * s_k.dot(yy_k.transpose())).dot(H_k)).dot(I - rho_k * yy_k.dot(s_k.transpose()))
                    H_k = H_k + rho_k * s_k.dot(s_k.transpose())
            '''Update iterate y_k+1 and compute x_k+1
                compute p_k with the inverse Hessian approx'''
            p_k = -H_k.dot(G_k)
            y_k = x_k + t_k * p_k
            y_k = np.maximum(y_k, 0)
            LHS = sum(y_k)
            RHS = self.data.b
            x_k_last = np.copy(x_k)
            if LHS <= RHS:
                #Gradient step is feasible
                x_k = y_k
            else:
                #Gradient step is infeasible and minimum distance projection onto X is given by:
                x_k = y_k + (RHS - LHS) / len(y_k)
            
            if self.options['output_level'] >= 2:
                lines = self.options['output_lines']
                if (iteration - 1) % (10 * lines) == 0:
                    # Prints the output header every 10 iterations
                    print(output_header)
                if (iteration - 1) % lines == 0:
                    print('%5i %10.2f %10.2f %10.5f%% %10.2e' %
                          (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, G_k_norm))
            
            #Update iterator and t_k
            iteration += 1  #Update iterator
            t_k = (self.options['t0'])  #/(iteration**0.5)
            if iteration >= self.options['max_iter'] - 100:
                break
        if self.options['output_level'] >= 1:
            print(
                problemName, ' %5i %10.2f %10.2f %10.5f%% %10.2f' %
                (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, (time.time() - TNOW)))
    
    def solveProblem_shalow_cuts(self, probName):
        '''
        Runs a L_shape method with a shallow cut search.
        '''
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X_plot = []
        Y_plot = []
        Z_plot = []
        cut_grads = []
        cut_inters = []
        '''
        ======================================================
        Initialization for all decomposition approaches
        '''
        TNOW = time.time()
        self.UB = np.inf  #Upper bound
        self.LB = np.inf  #Lower bound
        gap = 1  #Relative Gap
        x_star = None  #Data structure to store optimal solution
        #Instance of the master problem
        master = GenericMasterProblem(self.data)
        master.buildModel()
        #Instance of a single subproblem
        subProblem = CA_Subproblem(self.data, 0, np.zeros(len(self.data.I)))
        subProblem.buildModel()
        
        #Instance of the QP
        subProblemQP = CA_DualSub(self.data, 0, np.zeros(len(self.data.I)), QP=True)
        subProblemQP.buildModel()
        
        iteration = 1  #Iteration counter
        if self.options['output_level'] >= 2:
            # Prepare header for output
            output_header = '%5s %10s %10s %10s' % ('Iter', 'LB', 'UB', 'Gap')
        '''
        ======================================================
        Beginning of the main loop
        '''
        while True:
            '''Solve master problem to get x_hat'''
            [status, objval, x_hat, theta_hat, lagr_dual] = master.solve()
            self.LB = objval  #Update Lower Bound
            
            X_plot.append(x_hat[0])
            Y_plot.append(x_hat[1])
            '''Update x_hat in the subproblem, for scenario 0'''
            subProblem.updateX(x_hat, 0)
            subProblemQP.updateX(x_hat, 0)
            '''Data structure to store a dual extreme point'''
            dualCapEP = []  #List of duals vectors (EP) of the  capacity constraint
            dualRecEP = []  #List of duals vectors (EP) of the  recourse constraint
            
            z_hat = master.getValue_cx()  #Current obj. value
            '''Solve subproblems'''
            for w in self.data.Omega:
                subProblem.updateForScenario(w)
                [type, capDual, recDual, status, objval] = subProblem.solve()
                
                subProblemQP.updateForScenario(w, subProblem, self.LB, lagr_dual)
                [typeqp, capDualqp, recDualqp, statusqp, objvalqp] = subProblemQP.solve()
                assert type == 'EP'
                if w == -1:
                    print(capDual, recDual)
                    #print(capDualqp, recDualqp)
                dualCapEP.append(np.copy(capDualqp))
                dualRecEP.append(np.copy(recDualqp))
                z_hat = z_hat + self.data.prob[w] * objval
            Z_plot.append(z_hat)
            '''Update upper bound'''
            if z_hat < self.UB:
                self.UB = z_hat
                x_star = np.copy(x_hat)
            '''Update gap and check for termination '''
            gap = (self.UB - self.LB + 0.0) / np.minimum(np.abs(self.UB), np.abs(self.LB) + self.options['float_tol'])
            #print(gap)
            if gap <= self.options['opt_tol']:
                #TODO: Report solution
                if self.options['output_level'] >= 2:
                    print('%5i %10.2f %10.2f %10.5f%%' % (iteration, self.LB, self.UB, np.minimum(1, gap) * 100))
                    if self.options['output_level'] >= 3:
                        self.printFinalSolution(x_star)
                break
            '''Output options'''
            if self.options['output_level'] == 2:
                lines = self.options['output_lines']
                if (iteration - 1) % (10 * lines) == 0:
                    # Prints the output header every 10 iterations
                    print(output_header)
                if (iteration - 1) % lines == 0:
                    print('%5i %10.2f %10.2f %10.5f%%' % (iteration, self.LB, self.UB, np.minimum(1, gap) * 100))
            '''Add cut if the tolerance criterion wasn't met'''
            #print(x_hat)
            grad, inter = master.addOptCut(dualCapEP, dualRecEP)
            #print(np.linalg.norm(grad))
            cut_grads.append(grad)
            cut_inters.append(inter)
            #plt.draw()
            iteration += 1  #Update iterator
            if iteration >= self.options['max_iter']:
                break
        
        #=======================================================================
        # cut_grads.pop(0)
        # cut_grads.pop(0)
        # cut_grads.pop(0)
        # cut_inters.pop(0)
        # cut_inters.pop(0)
        # cut_inters.pop(0)
        #=======================================================================
        def theta_x0x1(x0, x1):
            z_max = -np.inf
            for (c, g) in enumerate(cut_grads):
                z = g[0] * x0 + g[1] * x1 + cut_inters[c]
                if z > z_max:
                    z_max = z
            return z_max
        
        #=======================================================================
        # X_plot.pop(0)
        # Y_plot.pop(0)
        # Z_plot.pop(0)
        # ax.plot_trisurf(X_plot,Y_plot,Z_plot,shade=False)
        # ax.set_zlim3d(120,500)
        #=======================================================================
        #=======================================================================
        # x1 = []
        # x2 = []
        # o = []
        # for xx in range(0,7500,50):
        #     ystep = np.maximum(1,int((1-xx/7500)*50))
        #     for yy in range(0,7500-xx,ystep):
        #         zz = theta_x0x1(xx, yy)
        #         x1.append(xx)
        #         x2.append(yy)
        #         o.append(zz)
        #
        # #X, Y = np.meshgrid(x, y)
        # #Z = x+y #np.sin(X, Y)
        # #print(Z)
        # ax.plot_trisurf(x1, x2, o);
        # ax.set_zlim3d(0,500)
        #=======================================================================
        #ax.set_xlim(0,7500)
        #ax.set_xlim(0,7500)
        ini_val = 240
        fin_val = 450
        step_val = int((fin_val - ini_val) / 3)
        
        x = np.linspace(ini_val - 100, fin_val, step_val)
        y = np.linspace(ini_val, fin_val + 100, step_val)
        X, Y = np.meshgrid(x, y)
        Z = X + Y
        for i in range(len(Z)):
            for j in range(len(Z[i])):
                Z[i][j] = theta_x0x1(X[i][j], Y[i][j])
        print(x_hat)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
        #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.6)
        #ax.contour(X, Y, Z,zdir='z',offset=-10)
        #ax.contour(X, Y, Z,zdir='x',offset=-10)
        #ax.contour(X, Y, Z,zdir='y',offset=-10)
        ax.set_xlim3d(ini_val - 100, fin_val)
        ax.set_ylim3d(ini_val, fin_val + 100)
        #ax.set_zlim3d(155,300)
        ax.view_init(0, 120)
        plt.show()
        if self.options['output_level'] >= 1:
            print(
                probName, ' %5i %10.2f %10.2f %10.5f%% %10.2f' %
                (iteration, self.LB, self.UB, np.minimum(1, gap) * 100, (time.time() - TNOW)))
    
    def solveProblem_DRO_CW(self, probName, **kwargs):
        '''
        Runs a L_shape method for DRO problem using a Wasserstein
        ambiguity set. The function solves both discrite and continuous
        support versions of Wasserstein ambiguity sets.
        '''
        dro_r = kwargs['dro_r']
        discrete_wasserstein = kwargs['discrete_wasserstein']
        add_scenario = kwargs['add_scenario']
        simulate = kwargs['simulate']
        alg_output = kwargs['output']
        distance_type = kwargs['distance_type']
        cut_grads = []
        cut_inters = []
        '''
        ======================================================
        Initialization for all decomposition approaches
        '''
        TNOW = time.time()
        self.UB = np.inf  #Upper bound
        self.LB = np.inf  #Lower bound
        gap = 1  #Relative Gap
        x_star = None  #Data structure to store optimal solution
        # Modify data for given paramters
        if add_scenario:
            self.data.expand_scenario_list()
        #Instance of the master problem
        master = DROMasterProblem(self.data, dro_r, discrete_wasserstein, add_scenario, distance_type)
        master.buildModel()
        #Instance of a single subproblem
        subProblem1 = CA_DualSub_DRO(self.data,
                                     0,
                                     x_hat=np.zeros(len(self.data.I)),
                                     lambda_hat=0,
                                     discrete_wasserstein=True,
                                     add_scenario=False)
        subProblem1.buildModel()
        
        subProblem2 = CA_DualSub_DRO(self.data,
                                     0,
                                     x_hat=np.zeros(len(self.data.I)),
                                     lambda_hat=0,
                                     discrete_wasserstein=discrete_wasserstein,
                                     add_scenario=add_scenario)
        if 'transport' in probName:
            subProblem2 = CA_DualSub_DRO_L2norm(self.data,
                                                0,
                                                x_hat=np.zeros(len(self.data.I)),
                                                lambda_hat=0,
                                                discrete_wasserstein=False,
                                                add_scenario=False)
        
        # subProblem2 = CA_DualSub_DRO_McCormick(self.data,
        #                                        0,
        #                                        x_hat=np.zeros(len(self.data.I)),
        #                                        lambda_hat=0,
        #                                        discrete_wasserstein=False,
        #                                        add_scenario=False)
        if 'box_mip' in probName:
            subProblem2 = CA_DualSub_DRO_MIP(self.data, 0, x_hat=np.zeros(len(self.data.I)), lambda_hat=0)
        subProblem2.buildModel()
        
        iteration = 1  #Iteration counter
        lp_sup_calls = 0
        mip_sup_calls = 0
        if self.options['output_level'] >= 2:
            # Prepare header for output
            output_header = f"{'Iter':5s} {'LB':10s} {'UB':10s} {'Gap':10s} {'Time':10s}"
        
        worst_case_support = {}
        '''
        ======================================================
        Beginning of the main loop
        '''
        terminate = False
        while not terminate:
            '''Solve master problem to get x_hat'''
            [status, objval, x_hat, lambda_hat, theta_hat] = master.solve()
            self.LB = objval  #Update Lower Bound
            '''Update x_hat in the subproblem, for scenario 0'''
            subProblem1.updateX(x_hat, lambda_hat, 0)
            '''Data structure to store a dual extreme point'''
            dualCapEP = []  # List of duals vectors (EP) of the  capacity constraint
            dualRecEP = []  # List of duals vectors (EP) of the  recourse constraint
            l1_norm_coeffs = []  # List of values represeting the l1-norm distance from \xi^w to the worst-case support
            intercepts = []  # List of intercepts for each scenario
            z_hat = master.getValue_cx()  #Current obj. value
            second_stage_ub = np.inf  # Initialize second stage ub
            '''Solve subproblems'''
            added_cuts = 0
            Scenarios = master.Scenarios
            fast = False
            if fast:
                for w in Scenarios:
                    xi_w = Scenarios[w]
                    subProblem1.updateForScenario(w, xi_w)
                    [cut_type, capDual, recDual, l1_norm_val, intercept_val, status, objval] = subProblem1.solve()
                    lp_sup_calls += 1
                    assert cut_type == 'EP'
                    cut_status = master.addOptCut(w, capDual, intercept_val)
                    added_cuts += cut_status
            
            if added_cuts == 0 and not discrete_wasserstein:
                #print(f'lambda* = {lambda_hat}')
                added_cuts = 0
                second_stage_ub = 0
                subProblem2.updateX(x_hat, lambda_hat, 0)
                worst_case_support = {}
                for w in range(self.data.n_sce):  #self.data.n_sce
                    xi_w = Scenarios[w]
                    subProblem2.updateForScenario(w, xi_w)
                    [cut_type, capDual, recDual, l1_norm_val, intercept_val, status, objval] = subProblem2.solve()
                    second_stage_ub += objval
                    mip_sup_calls += 1
                    assert cut_type == 'EP'
                    new_xi = subProblem2.get_support_point()
                    is_new, new_w = master.add_support_point(new_xi)
                    if is_new or not fast:  # if not new, the cut is already added.
                        # print(intercept_val, capDual, np.min(new_xi), subProblem2.model.ObjVal)
                        # print(new_xi)
                        cut_status = master.addOptCut(new_w, capDual, intercept_val, force_cut=False)
                        added_cuts += cut_status
                second_stage_ub = second_stage_ub / self.data.n_sce
                if added_cuts == 0:
                    terminate = True
                if master.update_lambda_underbar():
                    terminate = False
            elif added_cuts == 0 and discrete_wasserstein:
                terminate = True
            '''Output options'''
            self.UB = master.getValue_cx() + master.getValue_DRO() + second_stage_ub
            gap = np.abs(self.UB - self.LB) / self.LB if self.LB > 0 else 1
            if self.options['output_level'] >= 2:
                lines = self.options['output_lines']
                if (iteration - 1) % (10 * lines) == 0:
                    # Prints the output header every 10 iterations
                    print(output_header)
                if (iteration - 1) % lines == 0:
                    w_time = time.time() - TNOW
                    print(
                        f'{iteration:5} {self.LB:10.2f} {lambda_hat:10.2f} {np.minimum(1,gap)*100:10.5f} {w_time:10.2f}'
                    )
            '''Add cut if the tolerance criterion wasn't met'''
            
            iteration += 1  #Update iterator
            if iteration >= self.options['max_iter']:
                terminate = True
        
        if self.options['output_level'] >= 1:
            print(
                probName, ' %5i %10.2f %10.2f %10.5f%% %10.2f %5i %5i' %
                (iteration, self.LB, self.UB, np.minimum(1, gap) * 100,
                 (time.time() - TNOW), lp_sup_calls, mip_sup_calls))
            # print(sum(x_hat))
        
        master.model.reset()
        master.model.optimize()
        worst_case_dist = {}
        for i, j in master.dro_ctrs:
            zij = np.abs(master.dro_ctrs[i, j].pi)
            if zij > 1E-6 and j > -1:
                worst_case_dist[(i, j)] = {
                    'prob': zij,
                    'new_sup': np.round(master.Scenarios[j], 3),
                    'old_sup': np.round(master.Scenarios[i], 3)
                }
                # print(f'{i:4} {j:4} {zij:8.6f} ', np.round(master.Scenarios[j][:], 3),
                #       np.round(master.Scenarios[i][:2], 3))
        alg_output['worst_case_dist'] = worst_case_dist
        # for i in master.Scenarios:
        #         print(f'{i:4}  {sum(master.Scenarios[i]):6.3f}', np.round(master.Scenarios[i], 3))
        if simulate:
            self.simulate(probName, master, subProblem1)
    
    def solveProblem_DRO_CW_UNBOUNDED(self, probname, **kwargs):
        '''
        Solve DRO problem assuming and unbounded support
        '''
        TNOW = time.time()
        dro_r = kwargs['dro_r']
        simulate = kwargs['simulate']
        I = self.data.I
        J = self.data.J
        W = self.data.Omega
        
        # Model as in Hanasusanto and Kuhn 2018
        model = Model('CW_Unbounded_support')
        model.params.OutputFlag = 0
        model.params.Method = 0
        '''Set the problem orientation'''
        model.modelSense = GRB.MINIMIZE
        '''Create first stage decision variables'''
        x = model.addVars(I, vtype=GRB.CONTINUOUS, lb=0, ub=self.data.b, obj=0, name='x')
        '''
        Add maximum installation capacity constraint
        '''
        model.addConstr(sum(x[i] for i in self.data.I), GRB.LESS_EQUAL, self.data.b, 'globalCap')
        '''Create DRO variables'''
        lambda_dro = model.addVar(lb=0, obj=0, vtype=GRB.CONTINUOUS, name='lambda')
        '''Create second stage variables'''
        y = model.addVars(I, J, W, vtype=GRB.CONTINUOUS, lb=0, obj=0, name='y')
        s = model.addVars(J, W, vtype=GRB.CONTINUOUS, lb=0, obj=0, name='s')
        model.update()
        for w in W:
            '''Capacity constraints'''
            model.addConstrs((-sum(y[i, j, w] for j in self.data.J) >= -x[i] for i in self.data.I), f'CapCtr_{w}')
            '''Recourse constraint, subcontracting '''
            model.addConstrs((sum(y[i, j, w] for i in self.data.I) + s[j, w] >= self.data.d[j, w] for j in self.data.J),
                             f'RecCtr_{w}')
        '''Create DRO variables and constraints according to model'''
        K = J  # Dimension of the random vector
        y_r1 = model.addVars(I, J, K, vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f'y_ray1')
        s_r1 = model.addVars(J, K, vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f's_ray1')
        y_r2 = model.addVars(I, J, K, vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f'y_ray2')
        s_r2 = model.addVars(J, K, vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f's_ray2')
        model.update()
        '''Capacity constraints ray 1'''
        model.addConstrs((-sum(y_r1[i, j, k] for j in self.data.J) >= 0 for i in self.data.I for k in K), f'CapCtr_r1')
        '''Recourse constraint, subcontracting '''
        model.addConstrs((sum(y_r1[i, j, k] for i in self.data.I) + s_r1[j, k] >= (1 if k == j else 0) for j in J
                          for k in K), f'RecCtr_r1')
        '''Capacity constraints ray 2'''
        model.addConstrs((-sum(y_r2[i, j, k] for j in self.data.J) >= 0 for i in self.data.I for k in K), f'CapCtr_r2')
        '''Recourse constraint, subcontracting '''
        model.addConstrs((sum(y_r2[i, j, k] for i in self.data.I) + s_r2[j, k] >= -(1 if k == j else 0)
                          for j in self.data.J for k in K), f'RecCtr_r2')
        
        model.addConstrs((lambda_dro >= quicksum(self.data.c[i, j] * y_r1[i, j, k] for i in I
                                                 for j in J) + self.data.rho * quicksum(s_r1[j, k] for j in J)
                          for k in K), f'inf_norm_1')
        model.addConstrs((lambda_dro >= quicksum(self.data.c[i, j] * y_r2[i, j, k] for i in I
                                                 for j in J) + self.data.rho * quicksum(s_r2[j, k] for j in J)
                          for k in K), f'inf_norm_2')
        
        n_scenarios = len(W)
        q = 1 / n_scenarios
        mod_obj_exp = dro_r * lambda_dro + q * quicksum(self.data.c[i, j] * y[i, j, w] for i in I for j in J
                                                        for w in W) + q * self.data.rho * quicksum(s[j, w] for j in J
                                                                                                   for w in W)
        model.setObjective(mod_obj_exp, GRB.MINIMIZE)
        read_time = time.time() - TNOW
        model.optimize()
        cpu_time = time.time() - TNOW
        print([x[xv].X for xv in x])
        print(probname, model.objval, read_time, cpu_time)
    
    def solveProblem_DRO_CW_BOUNDED(self, probname, **kwargs):
        '''
        Solve DRO problem assuming an bounded support
        '''
        TNOW = time.time()
        dro_r = kwargs['dro_r']
        simulate = kwargs['simulate']
        I = self.data.I
        J = self.data.J
        W = self.data.Omega
        
        # Model as in Hanasusanto and Kuhn 2018
        model = Model('CW_Unbounded_support')
        model.params.OutputFlag = 0
        '''Set the problem orientation'''
        model.modelSense = GRB.MINIMIZE
        '''Create first stage decision variables'''
        x = model.addVars(I, vtype=GRB.CONTINUOUS, lb=0, ub=self.data.b, obj=0, name='x')
        '''
        Add maximum installation capacity constraint
        '''
        model.addConstr(sum(x[i] for i in self.data.I), GRB.LESS_EQUAL, self.data.b, 'globalCap')
        '''Create DRO variables'''
        lambda_dro = model.addVar(lb=0, obj=0, vtype=GRB.CONTINUOUS, name='lambda')
        '''Create second stage variables'''
        y = model.addVars(I, J, W, vtype=GRB.CONTINUOUS, lb=0, obj=0, name='y')
        s = model.addVars(J, W, vtype=GRB.CONTINUOUS, lb=0, obj=0, name='s')
        model.update()
        for w in W:
            '''Capacity constraints'''
            model.addConstrs((-sum(y[i, j, w] for j in self.data.J) >= -x[i] for i in self.data.I), f'CapCtr_{w}')
            '''Recourse constraint, subcontracting '''
            model.addConstrs((sum(y[i, j, w] for i in self.data.I) + s[j, w] >= self.data.d[j, w] for j in self.data.J),
                             f'RecCtr_{w}')
        '''Create DRO variables and constraints according to model'''
        K = J  # Dimension of the random vector
        beta_u = model.addVars(J, W, vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f'beta_u')
        beta_l = model.addVars(J, W, vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f'beta_l')
        
        y_r1 = model.addVars(I, J, K, vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f'y_ray1')
        s_r1 = model.addVars(J, K, vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f's_ray1')
        y_r2 = model.addVars(I, J, K, vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f'y_ray2')
        s_r2 = model.addVars(J, K, vtype=GRB.CONTINUOUS, lb=0, obj=0, name=f's_ray2')
        model.update()
        '''Capacity constraints ray 1'''
        model.addConstrs((-sum(y_r1[i, j, k] for j in self.data.J) >= 0 for i in self.data.I for k in K), f'CapCtr_r1')
        '''Recourse constraint, subcontracting '''
        model.addConstrs((sum(y_r1[i, j, k] for i in self.data.I) + s_r1[j, k] >= (1 if k == j else 0) for j in J
                          for k in K), f'RecCtr_r1')
        '''Capacity constraints ray 2'''
        model.addConstrs((-sum(y_r2[i, j, k] for j in self.data.J) >= 0 for i in self.data.I for k in K), f'CapCtr_r2')
        '''Recourse constraint, subcontracting '''
        model.addConstrs((sum(y_r2[i, j, k] for i in self.data.I) + s_r2[j, k] >= -(1 if k == j else 0)
                          for j in self.data.J for k in K), f'RecCtr_r2')
        
        model.addConstrs(
            (lambda_dro >= -(beta_u[k, w] - beta_l[k, w]) + quicksum(self.data.c[i, j] * y_r1[i, j, k] for i in I
                                                                     for j in J) + self.data.rho * quicksum(s_r1[j, k]
                                                                                                            for j in J)
             for k in K for w in W), f'inf_norm_1')
        model.addConstrs((lambda_dro >=
                          (beta_u[k, w] - beta_l[k, w]) + quicksum(self.data.c[i, j] * y_r2[i, j, k] for i in I
                                                                   for j in J) + self.data.rho * quicksum(s_r2[j, k]
                                                                                                          for j in J)
                          for k in K for w in W), f'inf_norm_2')
        
        n_scenarios = len(W)
        q = 1 / n_scenarios
        mod_obj_exp = dro_r * lambda_dro + q * quicksum(self.data.c[i, j] * y[i, j, w] for i in I for j in J
                                                        for w in W) + q * self.data.rho * quicksum(s[j, w] for j in J
                                                                                                   for w in W)
        mod_obj_exp = mod_obj_exp + q * quicksum(
            (self.data.d_ub - self.data.d[j, w]) * beta_u[j, w] for j in J
            for w in W) + q * quicksum(self.data.d[j, w] * beta_l[j, w] for j in J for w in W)
        model.setObjective(mod_obj_exp, GRB.MINIMIZE)
        read_time = time.time() - TNOW
        model.optimize()
        cpu_time = time.time() - TNOW
        
        print(probname, model.objval, read_time, cpu_time)
    
    def simulate(self, ins_name, master, subProblem, N_sim=1_000):
        
        [status, objval, x_hat, lambda_hat, theta_hat] = master.solve()
        # print(f'First stage x: {str(x_hat):20s}      cx={master.cx.getValue()}')
        z_hat = master.getValue_cx()  #Current obj. value
        '''Update x_hat in the subproblem, for scenario 0'''
        subProblem.updateX(x_hat, lambda_hat=0, scenario=0)
        
        z_vals = []
        '''Solve subproblems'''
        for w in range(N_sim):
            subProblem.updateForScenario(w, self.data.d_out_of_sample[:, w])
            [type, capDual, recDual, l1_norm_val, intercept_val, status, objval] = subProblem.solve()
            z_vals.append(z_hat + objval)
        sim_result = SimResult({'risk_measure_params': {'radius': master.dro_r}}, z_vals)
        print(f'Median {np.median(z_vals):10.2f}')
        print(f'Mean  {np.mean(z_vals):10.2f}')
        print(f'SD    {np.std(z_vals):10.2f}')
        print(f'90p   {np.percentile(z_vals, 90):10.2f}')
        print(f'95p   {np.percentile(z_vals, 95):10.2f}')
        print(f'99p   {np.percentile(z_vals, 99):10.2f}')
        filename = ins_name.replace(' ', '_')
        write_object_results(f'./output/{ins_name}_OOS.pickle', sim_result)
    
    def run(self, type, probName, **kwargs):
        '''
        Run the required algorithm
        '''
        if type == 'a':
            self.solveProblem_a(probName)
        elif type == 'b':
            self.solveProblem_b(probName)
        elif type == 'c':
            self.solveProblem_c(probName)
        elif type == 'd':
            self.solveProblem_d(probName)
        elif type == 'dd':
            self.solveProblem_d_BFGS(probName)
        elif type == 'shalow_cuts':
            self.solveProblem_shalow_cuts(probName)
        elif type == 'dro_cw':
            self.solveProblem_DRO_CW(probName, **kwargs)
        elif type == 'dro_cw_us':
            self.solveProblem_DRO_CW_UNBOUNDED(probName, **kwargs)
        elif type == 'dro_cw_bs':
            self.solveProblem_DRO_CW_BOUNDED(probName, **kwargs)
        else:
            raise 'No such problem!! '


# master.model.optimize()
# for w in range(30):
#  for ix in range(master.optCuts):
#   c = master.model.getConstrByName(f'cut_{w}_{ix}')
#   if c.pi < 0 :
#     print(c.constrname, np.round(c.pi,6), '| \t' , np.round(worst_case_support[w,ix],3), '| \t', np.round(self.data.d[:,w],3))
