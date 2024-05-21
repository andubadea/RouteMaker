import networkx as nx
import geopandas as gpd
from .params import Parameters
from .problem import ProblemGlobal

class PathModel():
    def __init__(self, scenario:dict, G:nx.MultiDiGraph, nodes:gpd.GeoDataFrame, 
                 edges:gpd.GeoDataFrame, time_horizon:int, time_step:float, 
                 fl_num:int, fl_size:float, C:int|list, time_window:int,
                 v_cruise: float, v_turn: float, v_up: float, v_down: float,
                 a_hoz: float, max_flight_time: float,
                 overlap:bool = False, name = 'pathmodel') -> None:
        
        self.name = name
        
        # Create the parameters
        self.params = Parameters(scenario, G, nodes, edges, time_horizon, 
                                 time_step, fl_num, fl_size, C, time_window, 
                                 v_cruise, v_turn, v_up, v_down, a_hoz, 
                                 max_flight_time, overlap)
        
        # Create the problem
        self.problem = ProblemGlobal(self.params, self.name)
        
    def solve(self):
        # Solve the problem, and then save
        self.problem.solve()
        self.problem.model.write(f'{self.name}.sol')