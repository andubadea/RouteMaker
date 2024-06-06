import pickle
import os
from datetime import datetime
from typing import Any

from .params import Parameters
from .problem import ProblemGlobal

class BoundModel:
    def __init__(self, kwargs:dict, hmodel:Any = None) -> None:
        """This model is very good at getting to the branch and bound part
        quickly, but it works best if it is provided with a good fesible
        solution as a warm start.
        
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
        print('\n----- Initialising parameters -----\n')
        self.params = Parameters(kwargs)
        # Create the problem
        print('\n----- Creating problem -----\n')
        self.problem = ProblemGlobal(self.params)
        if hmodel is not None:
            # We can hot start
            print('> Found warm start model, applying solution...')
            self.now = hmodel.now
            hproblem = hmodel.problem
            # First, set the values of all r_fk and h_fy as 0
            for f,k in self.problem.r_fk:
                self.problem.r[f,k].Start = 0
            for f,y in self.problem.h_fy:
                self.problem.h[f,y].Start = 0

            # Loop through the z's, and set the r_fk and h_fy as well
            for f,k,y in self.problem.z_fky:
                self.problem.z[f,k,y].Start = hproblem.z[f,k,y].X
                if hproblem.z[f,k,y].X > 0.5:
                    # Set r_fk and h_fy
                    self.problem.r[f,k].Start = 1
                    self.problem.h[f,y].Start = 1
                    
            # Loop through the penalties and the violations
            for ntw,y in self.problem.v_ntwy:
                self.problem.v[ntw,y].Start = hproblem.v[ntw,y].X
                self.problem.pen[ntw,y].Start = hproblem.pen[ntw,y].X
            print('> Warm start solution applied.')
        else:
            self.now = datetime.now().strftime("%Y%m%d%H%M%S")
            
        # Write the model
        self.directory = f'data/output/{self.params.scen_name}_B_C{kwargs["C"]
                                        }_T{kwargs["time_window"]}_{self.now}'
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
            
        self.notypename = f'{self.directory}/{self.params.scen_name}_B_C{
                            kwargs["C"]}_T{kwargs["time_window"]}_{self.now}'
            
    def outputmps(self):
        """Output the MPS file."""
        print('\n----- Saving model mps -----\n')
        self.problem.model.write(f'{self.notypename}.mps')
        
    def solve(self):
        """Solve the problem, then save the results."""
        print('\n----- Solving problem -----\n')
        self.problem.solve()
        print('\n----- Saving solution files -----\n')
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
        return True