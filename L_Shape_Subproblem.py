'''
Created on Apr 11, 2017

@author: dduque
'''
from gurobipy import Model, quicksum, GRB
import numpy as np


class CA_Subproblem:
    '''
    This class implements the subproblem for the 
    Capacity Allocation problem. Given an install capacity
    and a particular scenario, this model determines the
    energy allocation for each customer under the given scenario.
    Additionally, it determines the amount of subcontracted demand.
    '''
    def __init__(self, data, scen, x_hat):
        '''
        Initialize an empty model and safe references of
        the gurboi mode and decision variables. 
        '''
        self.data = data
        self.sceNumber = scen
        self.model = Model('CA_Subproblem')  #Create a model
        self.model.params.outputflag = 0
        self.ready = False  #Flag to check if the model is ready to solve
        self.x = x_hat  #First stage decisions (installed capacity)
        self.y = None  #Reference for decision variables (allocation)
        self.s = None  #Reference for decision variables (subcontracting)
    
    def buildModel(self):
        '''
        Builds the optimization model for the given scenario
        and first stage decisions.
        '''
        '''Set the problem orientation'''
        self.model.modelSense = GRB.MINIMIZE
        '''Create decision variables'''
        self.y = self.model.addVars(self.data.I, self.data.J, vtype=GRB.CONTINUOUS, lb=0, obj=self.data.c, name='y')
        self.s = self.model.addVars(self.data.J, vtype=GRB.CONTINUOUS, lb=0, obj=self.data.rho, name='s')
        self.model.update()
        '''Capacity constraints'''
        self.model.addConstrs((sum(self.y[i, j] for j in self.data.J) <= self.data.f[i, self.sceNumber] * self.x[i]
                               for i in self.data.I), 'CapCtr')
        '''Recourse constraint, subcontracting '''
        self.model.addConstrs((sum(self.y[i, j] for i in self.data.I) + self.s[j] >= self.data.d[:, self.sceNumber][j]
                               for j in self.data.J), 'RecCtr')
        self.model.update()
    
    def lastMinuteChange(self):
        '''
        Changes back the sense of the capacity constraint in the
        BFGS implementation
        '''
        ctrs = self.model.getConstrs()
        for i in self.data.I:
            ctrs[i].Sense = '<'
        self.model.update()
    
    def updateX(self, x_hat, scenario):
        '''
        Update the subproblem for a new iterate x_hat
        '''
        self.sceNumber = scenario
        self.x = x_hat
        ctrs = self.model.getConstrs()
        for i in self.data.I:
            ctrs[i].RHS = self.data.f[i, self.sceNumber] * self.x[i]
        self.model.update()
        self.ready = True
    
    def solve(self):
        '''
        Solve the subproblem defined for the scenario
        
        =======================
        Output variables:
        =======================
        type:
            Type of the output, either is an extreme point (EP) or extreme ray (ER) of the dual subproblem
        capDual:
            Vector with the dual information (point or ray) of the capacity constraints
        recDual:
            Vector with the dual information (point or ray) of the recourse constraints (subcontracting)
        status:
            Status of the problem
        objval:
            Objective function value of the subproblem
        '''
        if self.ready == True:
            #Solve the model
            self.model.optimize()
            status = self.model.status
            capDual = None
            recDual = None
            cut_type = None
            objval = None
            if status == GRB.OPTIMAL:
                cut_type = 'EP'
                capDual = np.zeros(len(self.data.I))
                recDual = np.zeros(len(self.data.J))
                ctrs = self.model.getConstrs()
                for i in self.data.I:
                    capDual[i] = ctrs[i].Pi
                m = len(self.data.I)
                for j in self.data.J:
                    recDual[j] = ctrs[j + m].Pi
                objval = self.model.ObjVal
            elif status == GRB.INFEASIBLE:
                raise "Model can't be infeasible as it has complete recourse"
            else:
                raise 'Unexpected result for the subproblem'
            return cut_type, capDual, recDual, status, objval
        
        else:
            raise 'Subproblem for scenario %3i not ready to optimize' % (self.sceNumber)
        self.ready = False
    
    def lastMinuteChangeSetUp(self):
        '''
        Changes the sense of the capacity constraint in the
        BFGS implementation
        '''
        ctrs = self.model.getConstrs()
        for i in self.data.I:
            ctrs[i].Sense = '=='
        self.model.update()
    
    def updateForScenario(self, scenario):
        '''
        Update the subproblem for a new scenario
        '''
        self.sceNumber = scenario
        ctrs = self.model.getConstrs()
        m = len(self.data.I)
        for j in self.data.J:
            ctrs[j + m].RHS = self.data.d[:, self.sceNumber][j]
        self.model.update()
        self.ready = True


class CA_DualSub:
    '''
    This class implements the dual subproblem for the 
    Capacity Allocation problem. Given an install capacity
    and a particular scenario, this model determines the
    energy allocation for each customer under the given scenario.
    Additionally, it determines the amount of subcontracted demand.
    '''
    def __init__(self, data, scen, x_hat, QP=False):
        '''
        Initialize an empty model and safe references of
        the gurboi mode and decision variables. 
        '''
        self.data = data
        self.sceNumber = scen
        self.model = Model('CA_Subproblem')  #Create a model
        self.model.params.outputflag = 0
        self.model.params.InfUnbdInfo = 1  #Enable Farkas Duals
        if QP == True:
            self.QP = True
        
        self.dual_var1 = None  #Reference to the dual variables of the cap ctr for shallow cut search
        self.dual_var2 = None  #Reference to the dual variables of the demand ctr for shallow cut search
        self.strong_duality_ctr = None  #Reference to the strong duality constraint in the shallow cut QP
        self.ready = False  #Flag to check if the model is ready to solve
        self.x = x_hat  #First stage decisions (installed capacity)
        self.seqval = 0.9
    
    def buildModel(self):
        '''
        Builds the optimization model for the given scenario
        and first stage decisions.
        '''
        '''Set the problem orientation'''
        self.model.modelSense = GRB.MINIMIZE
        '''Create decision variables'''
        self.dual_var1 = self.model.addVars(self.data.I,
                                            vtype=GRB.CONTINUOUS,
                                            lb=-GRB.INFINITY,
                                            ub=GRB.INFINITY,
                                            obj=0,
                                            name='pi1')
        self.dual_var2 = self.model.addVars(self.data.J,
                                            vtype=GRB.CONTINUOUS,
                                            lb=0,
                                            ub=self.data.rho,
                                            obj=0,
                                            name='pi2')
        
        self.comp_slak = self.model.addVar(lb=-GRB.INFINITY,
                                           ub=GRB.INFINITY,
                                           vtype=GRB.CONTINUOUS,
                                           name='CompSlacknessDiff')
        self.model.update()
        '''Capacity constraints'''
        self.model.addConstrs((self.dual_var1[i] + self.dual_var2[j] <= self.data.c[i, j] for i in self.data.I
                               for j in self.data.J), 'DualCtr')
        '''Strong duality ctr'''
        self.strong_duality_ctr = self.model.addConstr(lhs=self.dual_var1.sum() + self.dual_var2.sum() - self.comp_slak,
                                                       rhs=0,
                                                       sense='==')
        
        qp_obj = QuadExpr()
        for i in self.data.I:
            qp_obj.add(self.dual_var1[i] * self.dual_var1[i])
        qp_obj.add(self.comp_slak * self.comp_slak)
        self.model.setObjective(qp_obj, GRB.MINIMIZE)
        
        #self.model.setObjective(-self.dual_var1.sum(), GRB.MINIMIZE)
        self.model.update()
    
    def updateForScenario(self, scenario, linear_sub, lb, lagr_dual):
        '''
        Update the subproblem for a new scenario in the shallow cut setting
        Update the strong duality constraint
        Args:
            scenario (int): ID of the snario being solved.
            linear_sub (CA_Subproblem): The corresponding subproblem to obtain primal and dual variables.
            lagr_dual (ndarray): vector representing the gradient of the lagrangian for the first stage
        '''
        self.sceNumber = scenario
        
        qp_obj = QuadExpr()
        for i in self.data.I:
            qp_obj.add((self.dual_var1[i] + lagr_dual[i]) * (self.dual_var1[i] + lagr_dual[i]))
        qp_obj.add(0.1 * self.comp_slak * self.comp_slak)
        self.model.setObjective(qp_obj, GRB.MINIMIZE)
        
        ctrs_sublin = linear_sub.model.getConstrs()
        m = len(self.data.I)
        #print(linear_sub.model.ObjVal*self.seqval + (1-self.seqval)*lb)
        self.strong_duality_ctr.RHS = linear_sub.model.ObjVal  #*self.seqval + (1-self.seqval)*lb
        for i in self.data.I:
            self.model.chgCoeff(self.strong_duality_ctr, self.dual_var1[i], self.x[i])
        for j in self.data.J:
            self.model.chgCoeff(self.strong_duality_ctr, self.dual_var2[j], self.data.d[:, self.sceNumber][j])
        
        self.model.update()
        #print(self.model.getRow(self.strong_duality_ctr))
        sum1 = sum(linear_sub.model.getConstrs()[i].Pi * self.x[i] for i in self.data.I)
        sum2 = sum(linear_sub.model.getConstrs()[m + j].Pi * self.data.d[:, self.sceNumber][j] for j in self.data.J)
        
        if self.sceNumber == -1:
            print(sum1 + sum2, linear_sub.model.ObjVal)
            print(self.model.getRow(self.strong_duality_ctr))
        self.ready = True
        #if self.sceNumber == 0:
        #self.seqval = np.minimum(1, self.seqval*1.2)
        #print(self.seqval)
    
    def updateX(self, x_hat, scenario):
        '''
        Update the subproblem for a new iterate x_hat or x_k
        '''
        self.sceNumber = scenario
        self.x = x_hat
        #=======================================================================
        # ctrs = self.model.getConstrs()
        # for i in self.data.I:
        #     ctrs[i].RHS = self.data.f[i,self.sceNumber]*self.x[i]
        # self.model.update()
        #=======================================================================
        self.ready = True
    
    def solve(self):
        '''
        Solve the subproblem defined for the scenario
        
        =======================
        Output variables:
        =======================
        type:
            Type of the output, either is an extreme point (EP) or extreme ray (ER) of the dual subproblem
        capDual:
            Vector with the dual information (point or ray) of the capacity constraints
        recDual:
            Vector with the dual information (point or ray) of the recourse constraints (subcontracting)
        status:
            Status of the problem
        objval:
            Objective function value of the subproblem
        '''
        if self.ready == True:
            #Solve the model
            self.model.optimize()
            status = self.model.status
            capDual = None
            recDual = None
            type = None
            objval = None
            if status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                type = 'EP'
                capDual = np.zeros(len(self.data.I))
                recDual = np.zeros(len(self.data.J))
                ctrs = self.model.getConstrs()
                for i in self.data.I:
                    capDual[i] = self.dual_var1[i].X
                m = len(self.data.I)
                for j in self.data.J:
                    recDual[j] = self.dual_var2[j].X
                objval = self.model.ObjVal
            elif status == GRB.INFEASIBLE:
                self.model.write("full_sum%i.lp" % (self.sceNumber))
                self.model.computeIIS()
                self.model.write("inf_sum%i.ilp" % (self.sceNumber))
                type = 'ER'
                capDual = np.zeros(len(self.data.I))
                recDual = np.zeros(len(self.data.J))
                for i in self.data.I:
                    capDual[i] = self.model.getConstrByName('CapCtr' + str(i)).FarkasDual
                for j in self.data.J:
                    recDual[j] = self.model.getConstrByName('CapCtr' + str(i)).FarkasDual
                objval = np.inf
            else:
                print(status)
                raise 'Unexpected result for the subproblem %i' % (status)
            return type, capDual, recDual, status, objval
        
        else:
            raise 'Subproblem for scenario %3i not ready to optimize' % (self.sceNumber)
        self.ready = False


class CA_DualSub_DRO:
    '''
    This class implements the dual subproblem for the
    Capacity Allocation problem allowing for the DRO-related
    penalty via duality. Given an install capacity, a dro dual variable
    and a particular scenario, this model finds a potentially new
    scenario as well as the cut gradient and intercept.
    '''
    def __init__(self, data, scen, x_hat, lambda_hat, discrete_wasserstein, add_scenario):
        '''
        Initialize an empty model and safe references of
        the gurboi mode and decision variables.
        '''
        self.data = data
        self.sceNumber = scen
        self.discrete_wasserstein = discrete_wasserstein
        self.add_scenario = add_scenario
        self.model = Model('CA_Subproblem')  # Create a model
        self.model.params.outputflag = 0
        self.model.params.NonConvex = 2  # Activates non-convex solver
        #self.model.params.TimeLimit = 1
        #self.model.params.InfUnbdInfo = 1   # Enable Farkas Duals
        
        self.dual_var1 = None  # Reference to the dual variables of the cap ctr for shallow cut search
        self.dual_var2 = None  # Reference to the dual variables of the demand ctr for shallow cut search
        self.demand_var = None  # Reference to the variable modeling demand (as new support point)
        self.l1_norm = None  # Reference to the expressions of th l_1 norm
        self.inter_exp = None  # Reference to the expressions of \sum_j \alpha_j * d_j (the intercept of the cut)
        self.ready = 0  # Flag to check if the model is ready to solve
        self.x = x_hat  # First stage decisions (installed capacity)
        self.lambda_dro = lambda_hat  # DRO dual variable
        
        self.d_star = {w: data.d[:, w] for w in data.Omega}  # Best solution of d (the randomness) in preview iteration
        self.norm_pos = None  # L1-norm constraints for +
        self.norm_neg = None  # L1-norm constraints for -
        self.norm_ctr = None  # L1-norm constraints
    
    def buildModel(self):
        '''
        Builds the optimization model for the given scenario
        and first stage decisions.
        '''
        
        m = self.model
        gens = self.data.I
        usrs = self.data.J
        '''Set the problem orientation'''
        m.modelSense = GRB.MAXIMIZE
        '''Create decision variables'''
        #pi = m.addVars(gens, lb=0, vtype=GRB.CONTINUOUS, name='pi')
        pi = m.addVars(gens, lb=-1, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='pi')
        al = m.addVars(usrs, lb=0, ub=self.data.rho, vtype=GRB.CONTINUOUS, name='al')
        d = m.addVars(usrs, lb=0, ub=self.data.d_ub, vtype=GRB.CONTINUOUS, name='d')
        # z = m.addVars(usrs, lb=0, vtype=GRB.CONTINUOUS, name='norm')
        z_plus = m.addVars(usrs, lb=0, vtype=GRB.CONTINUOUS, name='norm_plus')
        z_minus = m.addVars(usrs, lb=0, vtype=GRB.CONTINUOUS, name='norm_minus')
        m.update()
        '''Capacity constraints'''
        m.addConstrs((-pi[i] + al[j] <= self.data.c[i, j] for i in gens for j in usrs), 'dual_ctr')
        '''Demand polyhedron constraints'''
        # total_max_demand = self.data.d_ub * len(usrs) * 0.5
        # m.addConstr(lhs=quicksum(d), sense=GRB.LESS_EQUAL, rhs=total_max_demand)
        '''Norm constraints'''
        if not self.discrete_wasserstein:  # i.e., the continuous case
            self.norm = m.addConstrs((z_plus[j] - z_minus[j] + d[j] == 0 for j in usrs), 'norm_pos')
            b_plus = m.addVars(usrs, lb=0, vtype=GRB.BINARY, name='norm_plus')
            b_minus = m.addVars(usrs, lb=0, vtype=GRB.BINARY, name='norm_minus')
            m.addConstrs((self.data.d_ub * b_plus[j] >= z_plus[j] for j in usrs), 'b_plus')
            m.addConstrs((self.data.d_ub * b_minus[j] >= z_minus[j] for j in usrs), 'b_minus')
            m.addConstrs((b_minus[j] + b_plus[j] <= 1 for j in usrs), 'b_1')
            # self.norm_pos = m.addConstrs((-z[j] + d[j] == 0 for j in usrs), 'norm_pos')
            # self.norm_neg = m.addConstrs((-z[j] - d[j] <= 0 for j in usrs), 'norm_neg')
        '''l1_norm expression'''
        l1_norm = quicksum(z_plus) + quicksum(z_minus)
        intercept_exp = quicksum(d[j] * al[j] for j in usrs)
        '''Discrete wasserstein case'''
        if self.discrete_wasserstein:
            for j in usrs:
                z_plus[j].ub = 0
                z_minus[j].ub = 0
        
        exp_obj = self.build_dro_objective(pi, al, d, l1_norm, self.x, self.lambda_dro)
        m.setObjective(exp_obj, GRB.MAXIMIZE)
        m.update()
        '''Save variables'''
        self.dual_var1 = pi
        self.dual_var2 = al
        self.demand_var = d
        self.l1_norm = l1_norm
        self.inter_exp = intercept_exp
    
    def build_dro_objective(self, pi, al, d, l1_norm, x_hat, lambda_hat):
        gens = self.data.I
        usrs = self.data.J
        self.inter_exp = quicksum(d[j] * al[j] for j in usrs)
        return quicksum(-x_hat[i] * pi[i] for i in gens) + self.inter_exp - lambda_hat * l1_norm
    
    def updateForScenario(self, w, xi_w):
        '''
        Update the subproblem for a new scenario in the DRO setting, i.e.,
        need to updated the l1-norm constraints.
        Update the strong duality constraint
        Args:
            w (int): ID of the scenario being solved.
            xi_w (ndarray): realization of the random vector
        '''
        self.sceNumber = w
        d = xi_w
        if self.discrete_wasserstein:
            exp_obj = self.build_dro_objective(self.dual_var1, self.dual_var2, xi_w, 0, self.x, 0)
            self.model.setObjective(exp_obj, GRB.MAXIMIZE)
        else:
            for j in self.data.J:
                self.norm[j].rhs = xi_w[j]
    
    def updateX(self, x_hat, lambda_hat, scenario):
        '''
        Update the subproblem for a new iterate x_hat or x_k
        '''
        self.sceNumber = scenario
        self.x = x_hat
        self.lambda_dro = lambda_hat
        exp_obj = self.build_dro_objective(self.dual_var1, self.dual_var2, self.demand_var, self.l1_norm, self.x,
                                           self.lambda_dro)
        self.model.setObjective(exp_obj, GRB.MAXIMIZE)
    
    def get_support_point(self):
        self.d_star[self.sceNumber] = np.array([self.demand_var[j].x for j in self.demand_var])
        return self.d_star[self.sceNumber]
    
    def solve(self):
        '''
        Solve the subproblem defined for the scenario
        
        =======================
        Output variables:
        =======================
        type:
            Type of the output, either is an extreme point (EP) or extreme ray (ER) of the dual subproblem
        capDual:
            Vector with the dual information (point or ray) of the capacity constraints
        recDual:
            Vector with the dual information (point or ray) of the recourse constraints (subcontracting)
        l1_norm_val:
            Value of the penalty associated to the DRO dual variable
        status:
            Status of the problem
        objval:
            Objective function value of the subproblem
        '''
        
        #Set up opt sol for the scenario
        try:
            for j in self.data.J:
                #sol = self.d_star[self.sceNumber][j] if self.lambda_dro > 0 else self.data.d_wc[j]
                self.demand_var[j].Start = self.data.d_wc[j]  ## self.d_star[self.sceNumber][j]
        except KeyError:
            # Out-of-sample solves don't use re-start
            pass
        # Solve the model
        self.model.optimize()
        status = self.model.status
        capDual = None
        recDual = None
        l1_norm_val = None
        type = None
        objval = None
        if status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, 11]:
            type = 'EP'
            capDual = np.zeros(len(self.data.I))
            recDual = np.zeros(len(self.data.J))
            ctrs = self.model.getConstrs()
            for i in self.data.I:
                capDual[i] = -self.dual_var1[i].X  # Negative due to change of var
            m = len(self.data.I)
            for j in self.data.J:
                recDual[j] = self.dual_var2[j].X
            dstar = np.array([self.demand_var[j].x for j in self.data.J])
            l1_norm_val = self.l1_norm.getValue()
            #l1_norm_val1 = np.abs(self.data.d[:, self.sceNumber] - dstar).sum()  # self.l1_norm.getValue()
            intercept_val = self.inter_exp.getValue()
            objval = self.model.ObjVal - self.lambda_dro * l1_norm_val
        elif status == GRB.INFEASIBLE:
            raise 'Infeasible is not an option'
        else:
            print(status)
            raise 'Unexpected result for the subproblem %i' % (status)
        return type, capDual, recDual, l1_norm_val, intercept_val, status, objval
        self.ready = 0


class CA_DualSub_DRO_L2norm:
    '''
    This class implements the dual subproblem for the
    Capacity Allocation problem allowing for the DRO-related
    penalty via duality. Given an install capacity, a dro dual variable
    and a particular scenario, this model finds a potentially new
    scenario as well as the cut gradient and intercept.
    Xi = R^d 
    c(xi,xi^i) = ||xi  - xi^j|}|_2^2
    '''
    def __init__(self, data, scen, x_hat, lambda_hat, discrete_wasserstein, add_scenario):
        '''
        Initialize an empty model and safe references of
        the gurboi mode and decision variables.
        '''
        self.data = data
        self.sceNumber = scen
        self.discrete_wasserstein = discrete_wasserstein
        self.add_scenario = add_scenario
        self.model = Model('CA_Subproblem')  # Create a model
        self.model.params.outputflag = 0
        self.model.params.NonConvex = 2  # Activates non-convex solver
        self.model.params.TimeLimit = 300
        #self.model.params.InfUnbdInfo = 1   # Enable Farkas Duals
        
        self.dual_var1 = None  # Reference to the dual variables of the cap ctr for shallow cut search
        self.dual_var2 = None  # Reference to the dual variables of the demand ctr for shallow cut search
        self.demand_var = None  # Reference to the variable modeling demand (as new support point)
        self.squared_term = None  # Reference to the expressions of th squared l_2 norm term
        self.inter_exp = None  # Reference to the expressions of \sum_j \alpha_j * d_j^w (NOT THE INTERCEPT)
        self.ready = 0  # Flag to check if the model is ready to solve
        self.x = x_hat  # First stage decisions (installed capacity)
        self.lambda_dro = lambda_hat  # DRO dual variable
        
        self.d_star = {w: data.d[:, w] for w in data.Omega}  # Best solution of d (the randomness) in preview iteration
        self.norm_pos = None  # L1-norm constraints for +
        self.norm_neg = None  # L1-norm constraints for -
        self.norm_ctr = None  # L1-norm constraints
    
    def buildModel(self):
        '''
        Builds the optimization model for the given scenario
        and first stage decisions.
        '''
        
        m = self.model
        gens = self.data.I
        usrs = self.data.J
        '''Set the problem orientation'''
        m.modelSense = GRB.MAXIMIZE
        '''Create decision variables'''
        # pi = m.addVars(gens, lb=0, vtype=GRB.CONTINUOUS, name='pi')
        pi = m.addVars(gens, lb=-1, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='pi')
        al = m.addVars(usrs, lb=0, ub=self.data.rho, vtype=GRB.CONTINUOUS, name='al')
        self.squared_term = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='l2_norm_squared')
        m.update()
        '''Capacity constraints'''
        m.addConstrs((-pi[i] + al[j] <= self.data.c[i, j] for i in gens for j in usrs), 'dual_ctr')
        '''Norm constraints'''
        m.addConstr((self.squared_term == quicksum(al[j] * al[j] for j in usrs)), 'l2norm2_ctr')
        
        exp_obj = self.squared_term
        m.setObjective(exp_obj, GRB.MAXIMIZE)
        m.update()
        '''Save variables'''
        self.dual_var1 = pi
        self.dual_var2 = al
    
    def build_dro_objective(self, pi, al, d, x_hat, lambda_hat):
        gens = self.data.I
        usrs = self.data.J
        lin_obj = quicksum(-x_hat[i] * pi[i] for i in gens) + quicksum(d[j] * al[j] for j in usrs)
        quadratic_obj = (1 / (2 * lambda_hat)) * self.squared_term
        return lin_obj + quadratic_obj
    
    def updateForScenario(self, w, xi_w):
        '''
        Update the subproblem for a new scenario in the DRO setting, i.e.,
        need to updated the l1-norm constraints.
        Update the strong duality constraint
        Args:
            w (int): ID of the scenario being solved.
            xi_w (ndarray): realization of the random vector
        '''
        self.sceNumber = w
        exp_obj = self.build_dro_objective(self.dual_var1, self.dual_var2, xi_w, self.x, self.lambda_dro)
        self.model.setObjective(exp_obj, GRB.MAXIMIZE)
    
    def updateX(self, x_hat, lambda_hat, scenario):
        '''
        Update the subproblem for a new iterate x_hat or x_k
        '''
        self.sceNumber = scenario
        self.x = x_hat
        self.lambda_dro = lambda_hat
    
    def get_support_point(self):
        change_direction = (1 / (self.lambda_dro)) * np.array([self.dual_var2[j].x for j in self.data.J])
        self.d_star[self.sceNumber] = self.data.d[:, self.sceNumber] + change_direction
        return self.d_star[self.sceNumber]
    
    def solve(self):
        '''
        Solve the subproblem defined for the scenario
        
        =======================
        Output variables:
        =======================
        type:
            Type of the output, either is an extreme point (EP) or extreme ray (ER) of the dual subproblem
        capDual:
            Vector with the dual information (point or ray) of the capacity constraints
        recDual:
            Vector with the dual information (point or ray) of the recourse constraints (subcontracting)
        l1_norm_val:
            Value of the penalty associated to the DRO dual variable
        status:
            Status of the problem
        objval:
            Objective function value of the subproblem
        '''
        
        # Solve the model
        self.model.optimize()
        status = self.model.status
        capDual = None
        recDual = None
        l1_norm_val = None
        type = None
        objval = None
        if status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            type = 'EP'
            capDual = np.zeros(len(self.data.I))
            recDual = np.zeros(len(self.data.J))
            ctrs = self.model.getConstrs()
            for i in self.data.I:
                capDual[i] = -self.dual_var1[i].X  # Negative due to change of var
            m = len(self.data.I)
            for j in self.data.J:
                recDual[j] = self.dual_var2[j].X
            xi_star = self.get_support_point()
            intercept_val = sum(xi_star[j] * recDual[j] for j in self.data.J)
            objval = self.model.ObjVal
        elif status == GRB.INFEASIBLE:
            raise 'Infeasible is not an option'
        else:
            print(status)
            raise 'Unexpected result for the subproblem %i' % (status)
        return type, capDual, recDual, l1_norm_val, intercept_val, status, objval
        self.ready = 0


class CA_DualSub_DRO_McCormick:
    '''
    This class implements the dual subproblem for the
    Capacity Allocation problem allowing for the DRO-related
    penalty via duality. Given an install capacity, a dro dual variable
    and a particular scenario, this model finds a potentially new
    scenario as well as the cut gradient and intercept.
    '''
    def __init__(self, data, scen, x_hat, lambda_hat, discrete_wasserstein, add_scenario):
        '''
        Initialize an empty model and safe references of
        the gurboi mode and decision variables.
        '''
        self.data = data
        self.sceNumber = scen
        self.discrete_wasserstein = discrete_wasserstein
        self.add_scenario = add_scenario
        self.model = Model('CA_Subproblem')  # Create a model
        self.model.params.outputflag = 0
        #self.model.params.NonConvex = 2  # Activates non-convex solver
        #self.model.params.TimeLimit = 1
        #self.model.params.InfUnbdInfo = 1   # Enable Farkas Duals
        
        self.dual_var1 = None  # Reference to the dual variables of the cap ctr for shallow cut search
        self.dual_var2 = None  # Reference to the dual variables of the demand ctr for shallow cut search
        self.demand_var = None  # Reference to the variable modeling demand (as new support point)
        self.l1_norm = None  # Reference to the expressions of th l_1 norm
        self.inter_exp = None  # Reference to the expressions of \sum_j \alpha_j * d_j (the intercept of the cut)
        self.ready = 0  # Flag to check if the model is ready to solve
        self.x = x_hat  # First stage decisions (installed capacity)
        self.lambda_dro = lambda_hat  # DRO dual variable
        
        self.d_star = {w: data.d[:, w] for w in data.Omega}  # Best solution of d (the randomness) in preview iteration
        self.norm_pos = None  # L1-norm constraints for +
        self.norm_neg = None  # L1-norm constraints for -
        self.norm_ctr = None  # L1-norm constraints
    
    def buildModel(self):
        '''
        Builds the optimization model for the given scenario
        and first stage decisions.
        '''
        
        m = self.model
        gens = self.data.I
        usrs = self.data.J
        '''Set the problem orientation'''
        m.modelSense = GRB.MAXIMIZE
        '''Create decision variables'''
        pi = m.addVars(gens, lb=0, vtype=GRB.CONTINUOUS, name='pi')
        al = m.addVars(usrs, lb=0, ub=self.data.rho, vtype=GRB.CONTINUOUS, name='al')
        d = m.addVars(usrs, lb=0, ub=self.data.d_ub, vtype=GRB.CONTINUOUS, name='d')
        w = m.addVars(usrs, lb=0, ub=self.data.d_ub * self.data.rho, vtype=GRB.CONTINUOUS, name='w')
        z_plus = m.addVars(usrs, lb=0, vtype=GRB.CONTINUOUS, name='norm_plus')
        z_minus = m.addVars(usrs, lb=0, vtype=GRB.CONTINUOUS, name='norm_minus')
        m.update()
        '''Capacity constraints'''
        m.addConstrs((-pi[i] + al[j] <= self.data.c[i, j] for i in gens for j in usrs), 'dual_ctr')
        '''Demand polyhedron constraints'''
        #total_max_demand = self.data.d_ub * len(usrs) * 0.5
        #m.addConstr(lhs=quicksum(d), sense=GRB.LESS_EQUAL, rhs=total_max_demand)
        '''Norm constraints'''
        self.norm = m.addConstrs((z_plus[j] - z_minus[j] + d[j] == 0 for j in usrs), 'norm_pos')
        '''l1_norm expression'''
        l1_norm = quicksum(z_plus) + quicksum(z_minus)
        intercept_exp = quicksum(d[j] * al[j] for j in usrs)
        ''' McCormick'''
        mccormick_intercept = quicksum(w)
        m.addConstrs((w[j] <= self.data.d_ub * al[j] for j in usrs), 'mc_1')
        m.addConstrs((w[j] <= self.data.rho * d[j] for j in usrs), 'mc_2')
        
        m.setObjective(mccormick_intercept, GRB.MAXIMIZE)
        m.update()
        '''Save variables'''
        self.dual_var1 = pi
        self.dual_var2 = al
        self.demand_var = d
        self.l1_norm = l1_norm
        self.inter_exp = intercept_exp
        self.McIntercept = mccormick_intercept
    
    def build_dro_objective(self, pi, al, d, l1_norm, x_hat, lambda_hat):
        '''
        Builds dro subproblem objective for the McCormick relaxation
        '''
        gens = self.data.I
        usrs = self.data.J
        self.inter_exp = quicksum(d[j] * al[j] for j in usrs)
        return quicksum(-x_hat[i] * pi[i] for i in gens) + self.McIntercept - lambda_hat * l1_norm
    
    def updateForScenario(self, w, xi_w):
        '''
        Update the subproblem for a new scenario in the DRO setting, i.e.,
        need to updated the l1-norm constraints.
        Update the strong duality constraint
        Args:
            w (int): ID of the scenario being solved.
            xi_w (ndarray): realization of the random vector
        '''
        self.sceNumber = w
        for j in self.data.J:
            self.norm[j].rhs = xi_w[j]
    
    def updateX(self, x_hat, lambda_hat, scenario):
        '''
        Update the subproblem for a new iterate x_hat or x_k
        '''
        self.sceNumber = scenario
        self.x = x_hat
        self.lambda_dro = lambda_hat
        exp_obj = self.build_dro_objective(self.dual_var1, self.dual_var2, self.demand_var, self.l1_norm, self.x,
                                           self.lambda_dro)
        self.model.setObjective(exp_obj, GRB.MAXIMIZE)
    
    def get_support_point(self):
        self.d_star[self.sceNumber] = np.array([self.demand_var[j].x for j in self.demand_var])
        return self.d_star[self.sceNumber]
    
    def solve(self):
        '''
        Solve the subproblem defined for the scenario
        
        =======================
        Output variables:
        =======================
        type:
            Type of the output, either is an extreme point (EP) or extreme ray (ER) of the dual subproblem
        capDual:
            Vector with the dual information (point or ray) of the capacity constraints
        recDual:
            Vector with the dual information (point or ray) of the recourse constraints (subcontracting)
        l1_norm_val:
            Value of the penalty associated to the DRO dual variable
        status:
            Status of the problem
        objval:
            Objective function value of the subproblem
        '''
        
        #Set up opt sol for the scenario
        try:
            for j in self.data.J:
                #sol = self.d_star[self.sceNumber][j] if self.lambda_dro > 0 else self.data.d_wc[j]
                self.demand_var[j].Start = self.data.d_wc[j]  ## self.d_star[self.sceNumber][j]
        except KeyError:
            # Out-of-sample solves don't use re-start
            pass
        # Solve the model
        self.model.optimize()
        status = self.model.status
        capDual = None
        recDual = None
        l1_norm_val = None
        type = None
        objval = None
        if status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, 11]:
            type = 'EP'
            capDual = np.zeros(len(self.data.I))
            recDual = np.zeros(len(self.data.J))
            ctrs = self.model.getConstrs()
            for i in self.data.I:
                capDual[i] = -self.dual_var1[i].X  # Negative due to change of var
            m = len(self.data.I)
            for j in self.data.J:
                recDual[j] = self.dual_var2[j].X
            dstar = np.array([self.demand_var[j].x for j in self.data.J])
            l1_norm_val = self.l1_norm.getValue()
            #l1_norm_val = np.abs(self.data.d[:, self.sceNumber] - dstar).sum()  # self.l1_norm.getValue()
            intercept_val = self.inter_exp.getValue()
            objval = self.model.ObjVal - self.lambda_dro * l1_norm_val
        elif status == GRB.INFEASIBLE:
            raise 'Infeasible is not an option'
        else:
            print(status)
            raise 'Unexpected result for the subproblem %i' % (status)
        return type, capDual, recDual, l1_norm_val, intercept_val, status, objval
        self.ready = 0


class CA_DualSub_DRO_MIP:
    '''
    This class implements the dual subproblem for the
    Capacity Allocation problem allowing for the DRO-related
    penalty via duality. Given an install capacity, a dro dual variable
    and a particular scenario, this model finds a potentially new
    scenario as well as the cut gradient and intercept.
    '''
    def __init__(self, data, scen, x_hat, lambda_hat, discrete_wasserstein=False, add_scenario=False):
        '''
        Initialize an empty model and safe references of
        the gurboi mode and decision variables.
        '''
        self.data = data
        self.sceNumber = scen
        self.discrete_wasserstein = discrete_wasserstein
        self.add_scenario = add_scenario
        self.model = Model('CA_Subproblem_MIP')  # Create a model
        self.model.params.outputflag = 0
        #self.model.params.NonConvex = 2  # Activates non-convex solver
        #self.model.params.TimeLimit = 1
        #self.model.params.InfUnbdInfo = 1   # Enable Farkas Duals
        
        self.dual_var1 = None  # Reference to the dual variables of the cap ctr for shallow cut search
        self.dual_var2 = None  # Reference to the dual variables of the demand ctr for shallow cut search
        self.demand_var = None  # Reference to the variable modeling demand (as new support point)
        self.l1_norm = None  # Reference to the expressions of th l_1 norm
        self.inter_exp = None  # Reference to the expressions of \sum_j \alpha_j * d_j (the intercept of the cut)
        self.ready = 0  # Flag to check if the model is ready to solve
        self.x = x_hat  # First stage decisions (installed capacity)
        self.lambda_dro = lambda_hat  # DRO dual variable
        
        self.bin_ctrs_p = {}
        self.bin_ctrs_m = {}
    
    def buildModel(self):
        '''
        Builds the optimization model for the given scenario
        and first stage decisions.
        '''
        
        m = self.model
        gens = self.data.I
        usrs = self.data.J
        '''Set the problem orientation'''
        m.modelSense = GRB.MAXIMIZE
        '''Create decision variables'''
        #pi = m.addVars(gens, lb=0, vtype=GRB.CONTINUOUS, name='pi')
        pi = m.addVars(gens, lb=-1, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='pi')
        al = m.addVars(usrs, lb=0, ub=self.data.rho, vtype=GRB.CONTINUOUS, name='al')
        
        w_p = m.addVars(usrs, lb=0, ub=self.data.rho, vtype=GRB.CONTINUOUS, name='w_p')
        w_m = m.addVars(usrs, lb=0, ub=self.data.rho, vtype=GRB.CONTINUOUS, name='w_m')
        w_b = m.addVars(usrs, lb=0, ub=self.data.rho, vtype=GRB.CONTINUOUS, name='w_m')
        B_p = m.addVars(usrs, lb=0, ub=1, vtype=GRB.BINARY, name='B_p')
        B_m = m.addVars(usrs, lb=0, ub=1, vtype=GRB.BINARY, name='B_m')
        B_b = m.addVars(usrs, lb=0, ub=1, vtype=GRB.BINARY, name='B_b')
        
        m.update()
        '''Capacity constraints'''
        m.addConstrs((-pi[i] + al[j] <= self.data.c[i, j] for i in gens for j in usrs), 'dual_ctr')
        '''McCormic ctrs'''
        al_ub = self.data.rho
        al_lb = 0
        m.addConstrs((w_p[j] <= al_ub * B_p[j] for j in usrs), 'mc_p_1')
        m.addConstrs((w_p[j] <= al[j] + al_lb * (B_p[j] - 1) for j in usrs), 'mc_p_2')
        m.addConstrs((w_p[j] >= al_lb * B_p[j] for j in usrs), 'mc_p_3')
        m.addConstrs((w_p[j] >= al[j] - al_ub * (1 - B_p[j]) for j in usrs), 'mc_p_4')
        
        m.addConstrs((w_m[j] <= al_ub * B_m[j] for j in usrs), 'mc_m_1')
        m.addConstrs((w_m[j] <= al[j] + al_lb * (B_m[j] - 1) for j in usrs), 'mc_m_2')
        m.addConstrs((w_m[j] >= al_lb * B_m[j] for j in usrs), 'mc_m_3')
        m.addConstrs((w_m[j] >= al[j] - al_ub * (1 - B_m[j]) for j in usrs), 'mc_m_4')
        
        m.addConstrs((w_b[j] <= al_ub * B_b[j] for j in usrs), 'mc_b_1')
        m.addConstrs((w_b[j] <= al[j] + al_lb * (B_b[j] - 1) for j in usrs), 'mc_b_2')
        m.addConstrs((w_b[j] >= al_lb * B_b[j] for j in usrs), 'mc_b_3')
        m.addConstrs((w_b[j] >= al[j] - al_ub * (1 - B_b[j]) for j in usrs), 'mc_b_4')
        
        m.addConstrs((B_p[j] + B_m[j] + B_b[j] == 1 for j in usrs), 'cvx_ctr')
        
        self.bin_ctrs_m = m.addConstrs((w_m[j] + B_m[j] <= 0 for j in usrs), name='bin_ctrs_m')
        self.bin_ctrs_p = m.addConstrs((w_p[j] - B_p[j] >= 0 for j in usrs), name='bin_ctrs_p')
        
        d_ub = self.data.d_ub
        d_lb = 0
        self.obj_w = quicksum(d_ub * w_p[j] + d_lb * w_m[j] for j in usrs)
        
        m.setObjective(self.obj_w, GRB.MAXIMIZE)
        m.update()
        '''Save variables'''
        self.dual_var1 = pi
        self.dual_var2 = al
        self.B_p = B_p
        self.B_m = B_m
        self.B_b = B_b
        self.w_p = w_p
        self.w_m = w_m
        self.w_b = w_b
    
    def build_dro_objective(self, d_w, x_hat, lambda_hat):
        gens = self.data.I
        usrs = self.data.J
        pi = self.dual_var1
        bin_obj = quicksum(lambda_hat * (d_w[j] - self.data.d_ub) * self.B_p[j] for j in usrs)
        bin_obj += quicksum(lambda_hat * (0 - d_w[j]) * self.B_m[j] for j in usrs)
        self.bin_obj = bin_obj
        self.inter_exp = self.obj_w + quicksum(d_w[j] * self.w_b[j] for j in usrs)
        return quicksum(-x_hat[i] * pi[i] for i in gens) + self.inter_exp + bin_obj
    
    def updateForScenario(self, w, xi_w):
        '''
        Update the subproblem for a new scenario in the DRO setting, i.e.,
        need to updated the l1-norm constraints.
        Update the strong duality constraint
        Args:
            w (int): ID of the scenario being solved.
            xi_w (ndarray): realization of the random vector
        '''
        self.sceNumber = w
        exp_obj = self.build_dro_objective(xi_w, self.x, self.lambda_dro)
        self.model.setObjective(exp_obj, GRB.MAXIMIZE)
    
    def updateX(self, x_hat, lambda_hat, scenario):
        '''
        Update the subproblem for a new iterate x_hat or x_k
        '''
        self.sceNumber = scenario
        self.x = x_hat
        self.lambda_dro = lambda_hat
        for j in self.data.J:
            self.model.chgCoeff(self.bin_ctrs_m[j], self.B_m[j], lambda_hat)
            self.model.chgCoeff(self.bin_ctrs_p[j], self.B_p[j], -lambda_hat)
        self.model.update()
    
    def get_support_point(self):
        # self.d_star[self.sceNumber] = np.array([self.demand_var[j].x for j in self.demand_var])
        d_star = np.zeros(len(self.data.J))
        for j in self.data.J:
            if self.B_p[j].x > 0.9:
                d_star[j] = self.data.d_ub
            elif self.B_m[j].x > 0.9:
                d_star[j] = 0
            elif self.B_b[j].x > 0.9:
                d_star[j] = self.data.d[j, self.sceNumber]
            else:
                raise 'One of the binary variables should have taken value'
        return d_star
    
    def solve(self):
        '''
        Solve the subproblem defined for the scenario
        
        =======================
        Output variables:
        =======================
        type:
            Type of the output, either is an extreme point (EP) or extreme ray (ER) of the dual subproblem
        capDual:
            Vector with the dual information (point or ray) of the capacity constraints
        recDual:
            Vector with the dual information (point or ray) of the recourse constraints (subcontracting)
        l1_norm_val:
            Value of the penalty associated to the DRO dual variable
        status:
            Status of the problem
        objval:
            Objective function value of the subproblem
        '''
        
        # Solve the model
        self.model.optimize()
        status = self.model.status
        capDual = None
        recDual = None
        l1_norm_val = None
        type = None
        objval = None
        if status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, 11]:
            type = 'EP'
            capDual = np.zeros(len(self.data.I))
            recDual = np.zeros(len(self.data.J))
            ctrs = self.model.getConstrs()
            for i in self.data.I:
                capDual[i] = -self.dual_var1[i].X  # Negative due to change of var
            m = len(self.data.I)
            for j in self.data.J:
                recDual[j] = self.dual_var2[j].X
            
            l1_norm_val = self.bin_obj.getValue()
            intercept_val = self.inter_exp.getValue()
            objval = self.model.ObjVal - self.lambda_dro * l1_norm_val
        elif status == GRB.INFEASIBLE:
            raise 'Infeasible is not an option'
        else:
            print(status)
            raise 'Unexpected result for the subproblem %i' % (status)
        return type, capDual, recDual, l1_norm_val, intercept_val, status, objval
        self.ready = 0
