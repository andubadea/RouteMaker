import numpy as np
from scipy.sparse import coo_matrix
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
        # Dictionary that links the aircraft ID to the index of the set
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
        self.E, self.idx2e, self.e2idx = self.compute_high_degree_edges()
        # Set of all flight levels
        self.Y = np.arange(self.fl_num, dtype = int)
        # The maximum flow
        self.C_e = self.C
        # For each flight level, the time it takes ascend and descend
        self.Dlt_y = self.compute_time_to_alt()
        # Set of all time windows
        self.W_t = self.compute_time_windows()
        
        # Process the paths to get the set of all paths for each flight f, the
        # travel time of each path for each flight f, and the binary variable
        # that shows whether flight f is on edge e at time t. Also update the
        # edges so they only contain the ones we actually use.
        self.K_f, self.B_fk, self.xp_fket, self.W_t \
                        = self.process_paths(self.path_dict)
        
    def compute_high_degree_edges(self) -> np.ndarray:
        """For this problem, we only care about the flow along edges that have
        nodes of degree 3 or higher. Basically, if the nodes of the edge only
        connect to one two other nodes, they are of degree 2, so the flow will
        automatically be enforced.

        Returns:
            np.ndarray: List of edges
        """
        node_deg = self.G.degree()
        idx2e = []
        e2idx = {}
        for u,v,_ in self.edges.index:
            # Check the degrees of nodes u and v
            if node_deg[u] > 2 or node_deg[v] > 2:
                # This edge counts
                idx2e.append((u,v,0))
                e2idx[(u,v,0)] = len(idx2e)-1
        return np.arange(len(idx2e)), idx2e, e2idx
        
        
    def compute_edge_flow_limit(self) -> np.ndarray:
        """The flow limit for each edge.
        """
        return np.array([self.C * self.time_window/60]*len(self.edges)
                        ).astype(int)

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
        
        # Initialise xp_fket as a dictionary
        xp_fket = {}
        
        
        # xp_fket = np.zeros((len(self.F),
        #                     len(list(self.path_dict.values())[0]['paths']),
        #                     len(self.E),
        #                     len(self.W_t)), dtype=bool)
        
        # Path_dict is of shape {acid:{paths,times}}
        # iterate over all flight idxs
        max_tw = 0
        for acidx in self.F:
            # Get the aircraft acid
            acid = self.idx2acid[acidx]
            # Get the paths and the path times
            paths = path_dict[acid]['paths']
            path_times = path_dict[acid]['times']
            # Let's ensure that we only account for unique paths
            flight_times = []
            path_list = []
            # Get the edges in the path, and then convert those edges to edge
            # indices
            for path_idx, path in enumerate(paths):
                # Check if this path is not the shortest path again
                if path_idx > 0 and path == paths[0]:
                    # We have reached a duplicate path, so we stop
                    break
                for path_edge_idx, edge in enumerate(zip(path[:-1], path[1:])):
                    u,v = edge
                    # Get the global edge idx only if we care about that edge
                    if (u,v,0) in self.e2idx.keys():
                        edge_idx = self.e2idx[(u,v,0)]
                    else:
                        # We skip this edge
                        continue
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
                        xp_fket[acidx,path_idx,edge_idx,t] = 1
                    # Set the maximum time window
                    if t_idx_int > max_tw:
                        max_tw = t_idx_int
                        
                # Add the path time and idx
                flight_times.append(path_times[path_idx][-1]-
                                                path_times[path_idx][0])
                path_list.append(path_idx)
            
            #B_f_k wants flight times per path
            B_fk.append(flight_times)
            K_f.append(path_list)
        return K_f, B_fk, xp_fket, np.arange(max_tw+1)
            