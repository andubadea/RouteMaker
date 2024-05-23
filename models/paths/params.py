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
        self.idx2e = list(self.edges.index)
        self.e2idx = dict(zip(list(self.edges.index), range(len(self.edges))))
        self.idx2n = list(self.nodes.index)
        self.n2idx = dict(zip(list(self.nodes.index), range(len(self.nodes))))
        # Another dictionary that links the aircraft ID to the index of the set
        self.acid2idx = dict(zip(self.scenario.keys(),
                                 range(len(self.scenario))))
        self.idx2acid = list(self.scenario.keys())
        
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
        # For each edge, the maximum flow
        self.C_e = self.compute_edge_flow_limit()
        # For each flight level, the time it takes ascend and descend
        self.Dlt_y = self.compute_time_to_alt()
        # Set of all time windows
        self.W_t = self.compute_time_windows()
        
        # Process the paths to get the set of all paths for each flight f, the
        # travel time of each path for each flight f, and the binary variable
        # that shows whether flight f is on edge e at time t
        self.K_f, self.B_fk, self.xp_fket = self.process_paths(self.path_dict)
        
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
        # Each time window is decribed by an index (position in list), and
        # [min time, max_time]
        if self.overlap:
            # We want overlapping windows. We overlap by half the time window.
            increment = self.time_window / 2
        else:
            increment = self.time_window
        # Create the list of time windows
        time_windows = []
        t = 0
        while True:
            time_windows.append([t, t + self.time_window])
            t += increment
            if t > self.time_horizon:
                break
        
        return time_windows
    
    def process_paths(self, path_dict:dict) -> list:
        """Processes the paths.

        Args:
            path_dict (dict): The path dictionary.

        Returns:
            list: K_f, B_f_k, and xp_fket
        """
        K_f = []
        B_fk = []
        # Initialise xp_fket as a big matrix
        xp_fket = np.zeros((len(self.F),
                            len(list(self.path_dict.values())[0]['paths']),
                            len(self.E),
                            len(self.W_t)), dtype=bool)
        
        # Path_dict is of shape {acid:{paths,times}}
        # iterate over all flight idxs
        for acidx in self.F:
            # Get the aircraft acid
            acid = self.idx2acid[acidx]
            # Get the paths and the path times
            paths = path_dict[acid]['paths']
            path_times = path_dict[acid]['times']
            # K_f simply wants path indices
            K_f.append(list(range(len(paths))))
            flight_times = []
            # Get the edges in the path, and then convert those edges to edge
            # indices
            for path_idx, path in enumerate(paths):
                for path_edge_idx, edge in enumerate(zip(path[:-1], path[1:])):
                    u,v = edge
                    # Get the global edge idx
                    edge_idx = self.e2idx[(u,v,0)]
                    # Get the equivalent time for this edge
                    edge_time = path_times[path_idx][path_edge_idx]
                    # Get the list of time window indices in which we can find
                    # this time
                    # We can calculate this in function of the time window and
                    # whether we overlap or not
                    t_idx_float = edge_time / self.time_window
                    t_idx_int = int(np.round(t_idx_float))
                    if self.overlap:
                        t_idx = [t_idx_int * 2]
                        if t_idx_int < t_idx_float and t_idx_int > 0:
                            t_idx.append(t_idx[-1]-1)
                        elif t_idx_int > t_idx_float:
                            t_idx.append(t_idx[-1]+1)
                        else:
                            pass
                    else:
                        t_idx = [t_idx_int]
                    # Set the equivalent xp values to 1
                    for t in t_idx:
                        xp_fket[acidx][path_idx][edge_idx][t] = 1
                
                # Add the path time
                flight_times.append(path_times[path_idx][-1]-path_times[path_idx][0])
            
            #B_f_k wants flight times per path
            B_fk.append(flight_times)
            if acidx == 1:
                break
        return K_f, B_fk, xp_fket
            