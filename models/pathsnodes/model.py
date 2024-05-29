import pickle
import os
from datetime import datetime

from .params import Parameters
from .problem import ProblemGlobal

class PathNodeModel:
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
        print(f'************** {kwargs["scen_name"]} **************')
        # Get the current datetime
        self.now = datetime.now().strftime("%Y%m%d%H%M%S")
        # Create the parameter instance
        print('\n----------- Initialising parameters -----------\n')
        self.params = Parameters(kwargs)
        # Create the problem
        print('\n----------- Creating problem -----------\n')
        self.problem = ProblemGlobal(self.params)
        # Write the model
        self.directory = f'data/output/{self.params.scen_name}_{self.now}'
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
            
        self.notypename = f'{self.directory}/{self.params.scen_name}_{self.now}'
            
    def outputmps(self):
        """Output the MPS file."""
        print('\n----------- Saving model mps -----------\n')
        self.problem.model.write(f'{self.notypename}.mps')
        
    def solve(self):
        """Solve the problem, then save the results."""
        print('\n----------- Solving problem -----------\n')
        self.problem.solve()
        print('\n----------- Saving solution files -----------\n')
        # Save the results
        print('> Saving sol file...')
        self.problem.model.write(f'{self.notypename}.sol')
        # Also output the aircraft information in a pickle
        z_dict = {}
        for f in self.params.F:
            for k in self.params.K_f[f]:
                for y in self.params.Y:
                    z_dict[f,k,y] = self.problem.z[f,k,y].X
        print('> Saving data pickle...')
        data = [z_dict, self.params]
                    
        with open(f'{self.notypename}.pkl', 
                  'wb') as f:
            pickle.dump(data, f)