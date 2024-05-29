import gurobipy as gb
from gurobipy import GRB
from .params import Parameters
import numpy as np

class ProblemGlobal:
    def __init__(self, parameters:Parameters) -> None:
        self.p = parameters
        self.rng = np.random.default_rng(42)
        
        self.model = gb.Model(self.p.scen_name)
        self.createVars()
        self.model.update
        self.createObjectiveFunction()
        self.createConstraints()
        self.model.update
        
    def solve(self) -> bool:
        self.model.optimize()
        return self.model.Status == gb.GRB.OPTIMAL
    
    def createVars(self) -> None:
        """There is one decision variable:
        z_fky - Whether flight f will use path k at altitude y
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
        # for f in self.p.F:
        #     k = self.rng.choice(self.p.K_f[f])
        #     y = self.rng.choice(self.p.Y)
        #     self.z[f,k,y].Start = 1

    def createObjectiveFunction(self) -> None:
        self.model.setObjective(
            gb.quicksum(self.z[f,k,y] * (self.p.B_fk[f][k])
                for f,k,y in self.z), gb.GRB.MINIMIZE) 

    def createConstraints(self) -> None:
        # First of all, one and only one altitude and route combination can be
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
                                    for f,k in self.p.et_list[etw]
                                ) <= self.p.C_e
                            # forall vars
                            for etw in range(len(self.p.et_list))
                            for y in self.p.Y
                            ),
                            name = 'flow'
        )
    
    def printSolution(self) -> None:
        pass
