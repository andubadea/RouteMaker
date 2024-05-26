from .params import Parameters
from .problem import ProblemGlobal
import pickle

class PathModel:
    def __init__(self, **kwargs) -> None:
        """Model class that handles the parameters, problem creation and 
        solving.
        
        Args:
            scenario (dict): 
                A dictionary of the scenario, with each entry representing a 
                flight. Key is ACID, data is [dep time, origin, destination],
                dep time in seconds, origin and destination using the same id 
                as the nodes GDF
            G (nx.MultiDiGraph): 
                City street graph
            nodes (gpd.GeoDataFrame): 
                GDF of nodes
            edges (gpd.GeoDataFrame): 
                GDF of edges
            time_horizon (int): 
                The time horizon [s]
            time_step (float): 
                The time increment [s]
            fl_num (int): 
                Number of flight levels
            fl_size (float): 
                The thickness of a flight layer [m]
            C (int | list): 
                Edge capacity limit, either a single scalar or a list of length
                len(G.edges), defined as vehicles within one time window.
            time_window (int): 
                The number of seconds for each flow time window.
            v_cruise (float): 
                Aircraft cruise speed [m/s]
            v_turn (float): 
                Aircraft cruise speed [m/s]
            v_up (float): 
                Aircraft ascent speed [m/s]
            v_down (float): 
                Aircraft descent speed [m/s]
            a_hoz (float): 
                Aircraft horizontal acceleration [m/s2]
            yaw_r (float):
                Yaw rate [deg/s]
            max_flight_time (float):
                Maximum admissible flight time [s]
            overlap (bool, optional): 
                Whether the flow time windows overlap. Defaults to False.
            num_cpus (int, optional):
                Number of CPUs to use to generate paths. Defaults to 1.
            force_path_gen (bool, optional):
                Whether to force the regeneration of the path cache files,
                regardless of whether a cache file exists.
            scen_name (str, optional):
                The name given to this scenario and the path cache file it will
                create or look for. Defaults to 'out'.
            seed (int, optional):
                The random seed to use when generating aircraft paths. Defaults
                to 42.
        """
        # Create the parameter instance
        print('--- Initialising parameters.')
        self.params = Parameters(kwargs)
        
        # Create the problem
        print('--- Creating problem.')
        self.problem = ProblemGlobal(self.params)
        
    def solve(self):
        # Solve the problem, and then save
        print('--- Solving problem.')
        self.problem.solve()
        self.problem.model.write(f'{self.params.scen_name}.sol')
        # Also output the aircraft information in a pickle
        with open(f'{self.params.scen_name}.pkl', 'wb') as f:
            data = [self.params.acid2idx,
                    self.params.idx2acid,
                    self.params.path_dict]
            pickle.dump(data, f)