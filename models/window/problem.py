import gurobipy as gb
from gurobipy import GRB
from .params import Parameters
import numpy as np
from typing import Any

class ProblemGlobal:
    def __init__(self, parameters:Parameters, prev_problem:Any = None,
                 ac_fixed:list = None) -> None:
        self.p = parameters
        self.prev_problem = prev_problem
        self.ac_fixed = ac_fixed
        self.rng = np.random.default_rng(42)
        
        self.model = gb.Model(self.p.scen_name)
        self.createVars()
        self.model.update
        self.createObjectiveFunction()
        self.createConstraints()
        self.model.update
        
    def solve(self) -> bool:
        self.model.optimize()
        # Output some data
        # Violations
        v_tot = 0
        for ntw,y in self.v_ntwy:
            if self.v[ntw,y].X > 0:
                v_tot += 1
        print(f'Total flow constraints violated: {v_tot}')
        # More than one altitude and path
        for f in self.p.F:
            found_one = False
            for k in self.p.K_f[f]:
                for y in self.p.Y:
                    if self.z[f,k,y].X > 0:
                        if not found_one:
                            found_one = True
                        else:
                            print(self.p.idx2acid(f))
                            continue
        return True
    
    def createVars(self) -> None:
        """Decision variables:
        z_fky - Whether flight f will use path k at altitude y
        v_ntwy - Constraint violation
        pen_ntwy - Constraint violation penalty
        """
        # Create the variable
        self.z_fky = []
        for f in self.p.F:
            for k in self.p.K_f[f]:
                for y in self.p.Y:
                    tp = f,k,y
                    self.z_fky.append(tp)
        
        # Add the variables to gurobi
        self.z = self.model.addVars(self.z_fky, 
                                    vtype = GRB.BINARY, 
                                    name = 'z')
        
        # We also need a constraint violation variable
        self.v_ntwy = []
        # self.pen_ntwy = []
        for ntw in range(len(self.p.nt_list)):
            for y in self.p.Y:
                tp = ntw,y
                self.v_ntwy.append(tp)
                
        self.v = self.model.addVars(self.v_ntwy,
                                    lb=0,
                                    name = 'v')
        
        if self.prev_problem is not None:
            # We have a previous problem, set the upper bound and lower bound 
            # of the z variables to whatever was found in the previous problem.
            for f,k,y in self.prev_problem.z_fky:
                if self.p.idx2acid[f] in self.ac_fixed:
                    self.z[f,k,y].lb = self.prev_problem.z[f,k,y].X
                    self.z[f,k,y].ub = self.prev_problem.z[f,k,y].X
        
    def createObjectiveFunction1(self) -> None:
        self.model.setObjective(
            (gb.quicksum(self.z[f,k,y] * (self.p.B_fk[f][k] + self.p.Dlt_y[y])
                for f,k,y in self.z) + \
            gb.quicksum(self.pen[ntw,y] 
                for ntw,y in self.pen))/len(self.p.F), 
            gb.GRB.MINIMIZE) 
        
    def createObjectiveFunction2(self) -> None:
        self.model.setObjective((gb.quicksum(self.pen[ntw,y] 
                for ntw,y in self.pen))/len(self.p.F), 
            gb.GRB.MINIMIZE) 
        
    def createObjectiveFunction(self) -> None:
        self.model.setObjective(
            (gb.quicksum(self.z[f,k,y] * (self.p.B_fk[f][k] + self.p.Dlt_y[y])
                for f,k,y in self.z)/len(self.p.F) + \
            gb.quicksum(self.v[ntw,y] 
                for ntw,y in self.v)), 
            gb.GRB.MINIMIZE)

    def createConstraints(self) -> None:
        # First of all, one altitude and route combination can be
        # allocated per flight.
        self.model.addConstrs((gb.quicksum(self.z[f,k,y] 
                                for k in self.p.K_f[f]
                                for y in self.p.Y 
                                ) == 1 
                            # forall vars
                            for f in self.p.F
                            ),
                            name = 'palt'
        )
        
        # Flow capacity constraint
        self.model.addConstrs((gb.quicksum(self.z[f,k,y]
                                for f,k in self.p.nt_list[ntw]
                                ) - self.p.C_n <= self.v[ntw,y]
                            # forall vars
                            for ntw in range(len(self.p.nt_list))
                            for y in self.p.Y
                            ),
                            name = 'flow'
        )
    
    def printSolution(self) -> None:
        pass
