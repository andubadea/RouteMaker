import numpy as np
import geopandas as gpd
import osmnx as ox
import networkx as nx
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

class Parameters:
    def __init__(self, scenario:dict, G:nx.MultiDiGraph, nodes:gpd.GeoDataFrame, 
                 edges:gpd.GeoDataFrame, time_horizon:int, time_step:float, 
                 fl_num:int, fl_size:float, C:int|list, time_window:int,
                 v_cruise: float, v_turn: float, v_up: float, v_down: float,
                 a_hoz: float, max_flight_time: float,
                 overlap:bool = False) -> None:
        """Class that contains or calculates all problem parameters.
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
            max_flight_time (float):
                Maximum admissible flight time [s]
            overlap (bool, optional): 
                Whether the flow time windows overlap. Defaults to False.
        """
        self.scenario = scenario
        self.G = G
        self.nodes = nodes
        self.edges = edges
        self.time_step = time_step
        self.fl_num = fl_num
        self.fl_size = fl_size
        self.C = C
        self.a_hoz = a_hoz
        self.overlap = overlap
        
        # Convert these directly in function of number of time steps
        self.time_horizon = int(time_horizon/self.time_step)
        self.time_window = int(time_window/self.time_step)
        self.max_flight_time = int(max_flight_time/self.time_step)
        self.v_cruise = v_cruise * self.time_step
        self.v_turn = v_turn * self.time_step
        self.v_up = v_up * self.time_step
        self.v_down = v_down * self.time_step
        
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
        self.O_f = np.array([self.scenario[acid][1] for acid in self.scenario], 
                            dtype = int)
        # For each flight, the destination node
        self.D_f = np.array([self.scenario[acid][2] for acid in self.scenario], 
                            dtype = int)
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