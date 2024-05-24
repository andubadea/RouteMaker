import gurobipy as gb
from gurobipy import GRB
from .params import Parameters

class ProblemGlobal:
    def __init__(self, parameters:Parameters) -> None:
        self.p = parameters
        
        self.model = gb.Model(self.p.scen_name)
        self.model.setParam('Threads', 8)
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

    def createObjectiveFunction(self) -> None:
        self.model.setObjective(
            gb.quicksum(self.z[f,k,y] * (self.p.B_fk[f][k] + self.p.Dlt_y[y])
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
                            name = 'one_path_alt'
        )
                
        # Flow capacity constraint
        self.model.addConstrs((gb.quicksum(
                                self.p.xp_fket.get((f,k,e,tw),0) * self.z[f,k,y]
                                for f in self.p.F
                                for k in self.p.K_f[f]
                                ) <= self.p.C_e
                            # forall vars
                            for e in self.p.E
                            for y in self.p.Y
                            for tw in range(len(self.p.W_t))
                            ),
                            name = 'flowcapacity'
        )
    
    def printSolution(self) -> None:
        pass
