import gurobipy as gb
from gurobipy import GRB
from .params import Parameters

class ProblemGlobal:
    def __init__(self, parameters:Parameters) -> None:
        self.p = parameters
        
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
        """There are two decision variables:
        xin_f_e_t_y - Whether aircraft f enters edge e at time t and altitude y
        xout_f_e_t_y - Whether aircraft f exits edge e at time t and altitude y
        """
        # Create the two variables
        xin_fety = []
        xout_fety = []
        xorg = []
        for f in self.p.F:
            for e in self.p.E:
                # Only allowed times for flight
                for t in self.p.Mt_f[f]:
                    for y in self.p.Y:
                        tp = f,e,t,y
                        xin_fety.append(tp)
                        xout_fety.append(tp)
                        
        # Create one for origin node edges and times as well
        for f in self.p.F:
            for e in self.p.Down_n[self.p.O_f[f]]:
                for t in [self.p.Mt_f[f][0]]:
                    for y in self.p.Y:
                        tp = f,e,t,y
                        xorg.append(tp)
                        
        
        self.xin_tup = gb.tuplelist(xin_fety)
        self.xout_tup = gb.tuplelist(xout_fety)
        self.xorg_tup = gb.tuplelist(xorg)
        
        # Add the variables to gurobi
        self.xp = self.model.addVars(self.xin_tup, 
                                    vtype = GRB.BINARY, 
                                    name = 'xp')
        
        self.xm = self.model.addVars(self.xout_tup, 
                                    vtype = GRB.BINARY, 
                                    name = 'xm')

    def createObjectiveFunction(self) -> None:
        self.model.setObjective(
            gb.quicksum(self.xp[f,e,t,y] * self.p.B_e[e]
                for f,e,t,y in self.xin_tup) + \
            gb.quicksum(self.xp[f,e,t,y] * self.p.Dlt_y[y] 
                for f,e,t,y in self.xorg_tup), 
            gb.GRB.MINIMIZE
        ) 

    def createConstraints(self) -> None:
        # First of all, all aircraft are supposed to leave the origin and enter
        # the destination.
        self.model.addConstrs((gb.quicksum(self.xp[f,e,t,y] 
                                for e in self.p.Down_n[self.p.O_f[f]]
                                for y in self.p.Y 
                                for t in [self.p.Mt_f[f][0]]
                                ) == 1 
                            # forall vars
                            for f in self.p.F
                            ),
                            name = 'origin'
        )
        
        self.model.addConstrs((gb.quicksum(self.xp[f,e,t,y] 
                                for e in self.p.Down_n[self.p.O_f[f]]
                                for y in self.p.Y 
                                for t in self.p.Mt_f[f]
                                ) == 1 
                            # forall vars
                            for f in self.p.F
                            ),
                            name = 'destination'
        )
        
        # If you enter an edge, you must also exit it
        self.model.addConstrs((self.xp[f][e][t][y] == 
                                self.xm[f][e][t+self.p.B_e[e]][y]
                            # forall vars
                            for f in self.p.F
                            for e in self.p.E
                            for t in self.p.Mt_f[f]
                            for y in self.p.Y
                            ),
                            name = 'edgeflow'
        )
        
        # If an edge is existed, another downstream edge needs to be entered
        self.model.addConstrs((gb.quicksum(self.xp[f,e,t,y]
                                for e in self.p.Up_n[n]) ==
                               gb.quicksum(self.xm[f,e,t,y]
                                for e in self.p.Down_n[n])
                            # forall vars
                            for f in self.p.F
                            for n in self.p.N_t
                            for t in self.p.Mt_f[f]
                            for y in self.p.Y
                            ),
                            name = 'nodeflow'
        )
        
        # Aircraft can only fly at one altitude for the whole flight
        self.model.addConstrs((gb.quicksum(self.xp[f,e,t,y] 
                                for y in self.p.Y
                                ) == 1
                            # forall vars
                            for f in self.p.F
                            for e in self.p.E
                            for t in self.p.Mt_f[f]
                            ),
                            name = 'onealt'
        )
        
        # Finally, the flow constraint
        self.model.addConstrs((gb.quicksum(self.xp[f,e,th,y]
                                for f in self.p.F
                                for th in self.p.W_t[t]
                                ) <= self.p.C_e[e]
                            # forall vars
                            for e in self.p.E
                            for t in self.p.T
                            for y in self.p.Y
                            ),
                            name = 'flowcapacity'
        )
    
    def printSolution(self) -> None:
        pass
