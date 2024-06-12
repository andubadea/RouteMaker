import pickle
import os
from datetime import datetime

from .params import Parameters
from .problem import ProblemGlobal

class WindowModel:
    def __init__(self, kwargs:dict) -> None:
        """This model can be used to plan aircraft using time windows. It
        makes use of several models, and solves one planning time window at a
        time. When a time window is solved, the solution is fixed within the
        next model, and the solve is repeated.
        
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
            planning_time_step (float): 
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
        # Get the current datetime
        self.now = datetime.now().strftime("%Y%m%d%H%M%S")
        # Create the model directory
        scen_name = kwargs['scen_name']
        self.directory = f'data/output/{scen_name}_W_C'
        self.directory += f'{kwargs["C"]}_T{kwargs["time_window"]}_{self.now}'
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
            
        self.notypename = f'{self.directory}/{scen_name}_W_C'
        self.notypename += f'{kwargs["C"]}_T{kwargs["time_window"]}_{self.now}'
        # Store the kwargs
        self.p_kwargs = kwargs
        # Initialise the time counter
        self.planning_time = 0
        self.planning_step = kwargs['planning_time_step']
        self.planning_overlap = kwargs['planning_overlap']
        # Get the scenario
        self.scenario = kwargs['scenario']
        
        
    def solve(self):
        """Solve the problem, then save the results."""
        prev_problem = None
        stop_solving = False
        while not stop_solving:
            plan_time_cutoff = self.planning_time + self.planning_step
            fixed_time_cutoff = self.planning_time
            print(f'\n******* Planning window: {self.planning_time}s - '+
                   f'{plan_time_cutoff}s *******\n')
            stop_solving = True
            # We extract the aircraft in the current planning time window. The
            # aircraft that are within the planning window but outside of the
            # overlap are the ones that are fixed.
            local_scen = {}
            ac_fixed = []
            for acid in self.scenario.keys():
                ac_data = self.scenario[acid]
                if ac_data[0] <= fixed_time_cutoff:
                    local_scen[acid] = ac_data
                    # Also add it to the fixed aircraft.
                    ac_fixed.append(acid)
                elif ac_data[0] <= plan_time_cutoff:
                    local_scen[acid] = ac_data
                else:
                    # There is at least one aircraft that must depart after
                    # this time window
                    stop_solving = False
                    
            print('\n----- Initialising parameters -----\n')
            params = Parameters(self.p_kwargs, local_scen)
            # Create the problem
            print('\n----- Creating problem -----\n')
            problem = ProblemGlobal(params, prev_problem, ac_fixed)
            for prm in self.p_kwargs['model_params']:
                problem.model.setParam(prm[0], prm[1])
            print('\n----- Solving problem -----\n')
            problem.solve()
            print('> Solved!')
            prev_problem = problem
            # Increment the time window
            self.planning_time += self.planning_step - self.planning_overlap
        print('> No more aircraft to plan, moving on!')
        # Save complete problem and params
        self.problem = problem
        self.params = params
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