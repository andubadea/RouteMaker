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
        # Output some data
        # Violations
        v_tot = 0
        for ntw,y in self.v_ntwy:
            if self.v[ntw,y].X > 0.1:
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
        # Whether flight f has path k and flight level y selected
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
        
        # Whether flight f has flight level y selected
        self.h_fy = []
        for f in self.p.F:
            for y in self.p.Y:
                tp = f,y
                self.h_fy.append(tp)
        self.h = self.model.addVars(self.h_fy, 
                                    vtype = GRB.BINARY, 
                                    name = 'h')
        
        # Whether flight f has path k selected
        self.r_fk = []
        for f in self.p.F:
            for k in self.p.K_f[f]:
                tp = f,k
                self.r_fk.append(tp)
        self.r = self.model.addVars(self.r_fk, 
                                    vtype = GRB.BINARY, 
                                    name = 'r')
        
        # We also need a constraint violation variable
        self.v_ntwy = []
        self.pen_ntwy = []
        for ntw in range(len(self.p.nt_list)):
            for y in self.p.Y:
                tp = ntw,y
                self.v_ntwy.append(tp)
                self.pen_ntwy.append(tp)
                
        self.v = self.model.addVars(self.v_ntwy,
                                    vtype = GRB.CONTINUOUS,
                                    lb=0,
                                    name = 'v')
        
        # And a penalty variable
        self.pen = self.model.addVars(self.pen_ntwy,
                                    vtype = GRB.CONTINUOUS,
                                    lb=0,
                                    name = 'pen')

    def createObjectiveFunction2(self) -> None:
        self.model.setObjective((gb.quicksum(self.pen[ntw,y] 
                for ntw,y in self.pen))/len(self.p.F), 
            gb.GRB.MINIMIZE) 
        
    def createObjectiveFunction(self) -> None:
        self.model.setObjective(
            (gb.quicksum(self.z[f,k,y] * (self.p.B_fk[f][k] + self.p.Dlt_y[y])
                for f,k,y in self.z) + \
            gb.quicksum(self.pen[ntw,y] 
                for ntw,y in self.pen))/len(self.p.F), 
            gb.GRB.MINIMIZE) 

    def createConstraints(self) -> None:
        # First of all, one altitude and route combination can be
        # allocated per flight.
        self.model.addConstrs((gb.quicksum(self.h[f,y] 
                                for y in self.p.Y 
                                ) >= 1 
                            # forall vars
                            for f in self.p.F
                            ),
                            name = 'alt'
        )
        
        self.model.addConstrs((gb.quicksum(self.r[f,k] 
                                for k in self.p.K_f[f]
                                ) >= 1 
                            # forall vars
                            for f in self.p.F
                            ),
                            name = 'path'
        )
        
        self.model.addConstrs((self.z[f,k,y] >= self.r[f,k] + self.h[f,y] - 1 
                            # forall vars
                            for f in self.p.F
                            for k in self.p.K_f[f]
                            for y in self.p.Y
                            ),
                            name = 'link'
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
        
        # First penalty constraint
        self.model.addConstrs((self.p.sw1*(self.v[ntw,y]) \
                                <= self.pen[ntw,y]
                              # forall vars
                            for ntw in range(len(self.p.nt_list))
                            for y in self.p.Y
                            ),
                            name = 'penalty1'
        )
        
        # Second penalty constraint
        self.model.addConstrs((self.p.sw2*(self.v[ntw,y]) - 
                        (self.p.sw2-self.p.sw1)*self.p.C_n <= self.pen[ntw,y]
                              # forall vars
                            for ntw in range(len(self.p.nt_list))
                            for y in self.p.Y
                            ),
                            name = 'penalty2'
        )
    
    def printSolution(self) -> None:
        pass
