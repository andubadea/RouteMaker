import gurobipy as gb

class ProblemGlobal:
    def __init__(self, parameters, name = 'My Problem') -> None:
        self.param = parameters
        
        self.model = gb.Model(name)
        self.createVars()
        self.model.update
        self.createObjectiveFunction()
        self.createConstraints()
        self.model.update
        
    def solve(self) -> bool:
        self.model.optimize()
        return self.model.Status == gb.GRB.OPTIMAL
    
    def createVars(self) -> None:
        pass

    def createObjectiveFunction(self) -> None:
        self.model.setObjective(1, gb.GRB.MINIMIZE) 
        pass

    def createConstraints(self) -> None:
        pass
    
    def printSolution(self) -> None:
        pass
