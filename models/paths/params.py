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
        self.paths = self.path_maker.get_paths()
        
        # Set of all flights
        self.F = np.arange(len(self.scenario), dtype = int)
        # Set of all nodes
        self.N = np.arange(len(self.nodes), dtype = int)
        # Set of all edges
        self.E = np.arange(len(self.edges), dtype = int)
        # Set of all flight levels
        self.Y = np.arange(self.fl_num, dtype = int)
        # Set of all time steps
        self.T = np.arange(np.ceil(self.time_horizon), 
                           dtype = int)
        # Departure time step for each flight
        self.Td_f = self.compute_dep_time()
        # Latest arrival time for each flight
        self.Ta_f = self.compute_arr_time()
        # For each flight, the origin node
        self.O_f = np.array([self.n2idx[self.scenario[acid][1]] 
                             for acid in self.scenario], dtype = int)
        # For each flight, the destination node
        self.D_f = np.array([self.n2idx[self.scenario[acid][2]] 
                             for acid in self.scenario], dtype = int)
        # For each edge, its estimated travel time
        self.B_e = self.compute_edge_travel_times()
        # For each edge, the maximum flow
        self.C_e = self.compute_edge_flow_limit()
        # For each node, its upstream edges in function of the index of E
        self.Up_n = self.get_upstream_edges()
        # For each node, its downstream edges in function of the index of E
        self.Down_n = self.get_downstream_edges()
        # The set of relevant time steps for each flight
        self.Mt_f = self.compute_mission_allowed_time()
        # For each flight level, the time it takes ascend and descend
        self.Dlt_y = self.compute_time_to_alt()
        # For each time step, its time window
        self.W_t = self.compute_time_windows()
        
        # Set of nodes without origin and destination for each aircraft
        self.N_f = self.get_nodes_without_origin_destination()
        
    def compute_dep_time(self) -> np.ndarray:
        """Compute the departure time step in function of the time step.
        """
        # Retrieve data
        dep_t = np.array([self.scenario[self.idx2acid[idx]][0] 
                          for idx in self.F])
        # Map onto time steps
        return np.ceil(dep_t/self.time_step).astype(int)
    
    def compute_arr_time(self) -> np.ndarray:
        """Compute the latest arrival time step in function of the time step
        using the existing self.Td_f.
        """
        # Get the max number of time steps per mission
        arr_max_t = int(np.ceil(self.max_flight_time))
        # Return array
        return np.clip(self.Td_f + arr_max_t, 0, 
                       self.time_horizon-1).astype(int)
    
    def compute_edge_travel_times(self) -> np.ndarray:
        """Edge time depends on the cruise speed and the length of the edge.
        Is in function of time steps
        """
        return np.ceil([self.edges.loc[e, 'length']/self.v_cruise
                         for e in self.edges.index]).astype(int)
        
    def compute_edge_flow_limit(self) -> np.ndarray:
        """The flow limit for each edge.
        """
        return np.array([self.C]*len(self.edges))
        
    def get_upstream_edges(self) -> dict:
        """For each node index, we get the upstream edge indices.
        """
        upstream_edges = {}
        for node in self.nodes.index:
            upstream_edges[self.n2idx[node]] = \
                [self.e2idx[(up_n, node, 0)] \
                            for up_n in self.G.predecessors(node)]
        return upstream_edges
        
    def get_downstream_edges(self) -> dict:
        """For each node index, we get the downstream edge indices.
        """
        downstream_edges = {}
        for node in self.nodes.index:
            downstream_edges[self.n2idx[node]] = \
                [self.e2idx[(node, down_n, 0)] \
                            for down_n in self.G.successors(node)]
        return downstream_edges
    
    def compute_mission_allowed_time(self) -> list:
        """The relevant time steps for each aircraft in function of the time
        step.
        """
        return [list(range(td, ta+1)) for td,ta in zip(self.Td_f, self.Ta_f)]

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
    
    def get_nodes_without_origin_destination(self) -> np.ndarray:
        new_nodes = np.empty((len(self.F), len(self.N)-2))
        for f in self.F:
            new_nodes[f] = self.N[
                np.logical_and(
                    np.logical_not(self.N == self.O_f[f]),
                    np.logical_not(self.N == self.D_f[f])
                )
            ]
        return new_nodes        