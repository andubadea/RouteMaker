import numpy as np
import geopandas as gpd
import networkx as nx
from .paths import PathMaker

class Parameters:
    def __init__(self, kwargs:dict) -> None:
        """Contains the model parameters for one scenario.

        Args:
            kwargs (dict): Inputs dictionary
        """
        # Store the inputs as class variables.
        self.__dict__.update(kwargs)
        
        # Compute the parameters
        self.compute_params()
        
    def compute_params(self) -> None:
        # First of all, dictionaries that link the index of the edges and nodes
        # to the actual indices of the edges and nodes
        self.idx2e = dict(zip(range(len(self.edges)), list(self.edges.index)))
        self.e2idx = dict(zip(list(self.edges.index), range(len(self.edges))))
        self.idx2n = dict(zip(range(len(self.nodes)), list(self.nodes.index)))
        self.n2idx = dict(zip(list(self.nodes.index), range(len(self.nodes))))
        # Another dictionary that links the aircraft ID to the index of the set
        self.acid2idx = dict(zip(self.scenario.keys(),
                                 range(len(self.scenario))))
        self.idx2acid = dict(zip(range(len(self.scenario)), 
                                 self.scenario.keys()))
        
        # Get the paths
        # Pass parameters to path making class
        self.path_maker = PathMaker(self)
        self.path_dict = self.path_maker.get_paths()
        
        # Set of all flights
        self.F = np.arange(len(self.scenario), dtype = int)
        # Set of all edges
        self.E = np.arange(len(self.edges), dtype = int)
        # Set of all flight levels
        self.Y = np.arange(self.fl_num, dtype = int)
        # Set of all time steps
        self.T = np.arange(np.ceil(self.time_horizon), 
                           dtype = int)
        # For each edge, the maximum flow
        self.C_e = self.compute_edge_flow_limit()
        # For each flight level, the time it takes ascend and descend
        self.Dlt_y = self.compute_time_to_alt()
        # For each time step, its time window
        self.W_t = self.compute_time_windows()
        
        # Process the paths to get the set of all paths for each flight f, the
        # travel time of each path for each flight f, and the binary variable
        # that shows whether flight f is on edge e at time t
        self.K_f, self.B_f_k, self.xp_e_f_k_t = self.process_paths(self.path_dict)
        
    def compute_edge_flow_limit(self) -> np.ndarray:
        """The flow limit for each edge.
        """
        return np.array([self.C]*len(self.edges))

    def compute_time_to_alt(self) -> np.ndarray:
        """Time it takes to get to and from each altitude level.
        """
        # Speed in function of time steps
        return np.array([int(self.fl_size * (i+1) / self.v_up) + 
                int(self.fl_size * (i+1) / self.v_down) 
                for i in range(self.fl_num)])
        
    def compute_time_windows(self) -> np.ndarray:
        """The time step indices within each time window for each time step.
        """
        # This can later be updated to include overlap logic
        max_time_window = list(range(self.time_window, self.time_horizon+1))
        # Add some time_horizon values at the end
        while len(max_time_window) < len(self.T):
            max_time_window.append(self.T[-1])
        
        return [list(range(t, t_m)) for t,t_m in zip(self.T, max_time_window)]
    
    def process_paths(self, path_dict:dict) -> list:
        """_summary_

        Args:
            path_dict (dict): The path dictionary.

        Returns:
            list: K_f, B_f_k, and xp_e_f_k_t
        """
        K_f = []
        B_f_k = []
        # Initialise xp_e_f_k_t as a big matrix
        # xp_e_f_k_t = np.zeros((len(self.E),
        #                        len(self.F),
        #                        len(self.)))
        # Path_dict is of shape {acid:{paths,times}}
        # iterate over all flight idxs
        for acidx in self.F:
            # Get the aircraft acid
            acid = self.acid2idx(acidx)
            # Get the paths and the path times
            paths = path_dict[acid]['paths']
            path_times = path_dict[acid]['times']
            # Get the edges in the path, and then convert those edges to edge
            # indices
            for path in paths:
                path_edges = [self.e2idx(edge) for edge in 
                              zip(path[:-1], path[1:], [0]*path[1:])]
                
                #
                
            