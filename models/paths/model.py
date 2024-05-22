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
                 overlap:bool = False, name = 'pathmodel',
                 num_cpus:int = 1) -> None:
        
        self.name = name
        
        # Create the parameters
        self.params = Parameters(scenario=scenario,
                                G=G,
                                nodes=nodes,
                                edges=edges,
                                time_horizon=time_horizon,
                                time_step=time_step,
                                fl_num=fl_num,
                                fl_size=fl_size,
                                C = C,
                                time_window=time_window,
                                v_cruise=v_cruise,
                                v_turn=v_cruise,
                                v_up = v_up,
                                v_down = v_down,
                                a_hoz = a_hoz,
                                max_flight_time=max_flight_time,
                                overlap=overlap,
                                num_cpus=num_cpus
        )
        
        # Create the problem
        self.problem = ProblemGlobal(self.params, self.name)
        
    def solve(self):
        # Solve the problem, and then save
        self.problem.solve()
        self.problem.model.write(f'{self.name}.sol')