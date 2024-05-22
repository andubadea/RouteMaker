import os
import tqdm
import copy
import random
import pyproj
import math
import pickle
import multiprocessing as mp
import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from shapely import convex_hull
from shapely.ops import linemerge, unary_union
from shapely.geometry import LineString, Polygon, Point

mp.set_start_method('fork')

# For plotting
colors = list(mcolors.CSS4_COLORS.keys())
random.shuffle(colors)
DEBUG = False

# Path generation
PATH_SKIP = 2 # Higher means less paths
N_RAND_PATHS = 5 # Number of random paths to generate
BUFFER_FACTOR = 1.5 # Higher means random path subgraph is smaller
PATH_ATTEMPTS = 50 # Higher means more attempts per random path
N_RAND_NODES = 1 # Number of random intermediate nodes
# Length limit for random routes in function of shortest route length
PATH_LENGTH_FACTOR = 1.5

class PathMaker():
    """Module to make alternative paths for each aircraft in a scenario.
    Class instance is specific to one scenario.
    """
    def __init__(self, scen_name:str, scenario:dict, G:nx.MultiDiGraph, 
                 nodes:gpd.GeoDataFrame, edges:gpd.GeoDataFrame, 
                 num_cpus:int = 1, seed:float = 42, 
                 force_gen:bool = False) -> None:
        """Class that creates alternative paths for one scenario.
        Args:
            scenario_name (str): 
                The name of the scenario, will be used to generate the cache
                for this scenario, and load it if it exists
            scenario (dict): 
                The scenario dictionary, created by the parser.
            G (nx.MultiDiGraph): The street graph.
            nodes (gpd.GeoDataFrame): The nodes in the graph.
            edges (gpd.GeoDataFrame): The edges in the graph.
            seed (float, optional): 
                The seed for the random number generator. Defaults to 42.
            force_gen (bool, optional):
                Whether to force the generation of alternative paths regardless
                of whether a cache already exists. Overwrites the existing 
                cache file.
        """
        self.scen_name = scen_name
        if DEBUG:
            # We're in debug mode, take the first two aircraft
            self.scenario = {'D1':scenario['D1'], 
                             'D2':scenario['D2']}
        else:
            self.scenario = scenario
            
        self.force_gen = force_gen
        # Store the graph information
        self.G = G
        self.nodes = nodes
        self.edges = edges
        
        # Create the coordinate transformers
        city_centre = self.nodes.dissolve().centroid
        utm_crs = self.convert_wgs_to_utm(city_centre.x[0], city_centre.y[0])
        self.geo2utm = pyproj.Transformer.from_crs(crs_from=4326, 
                                                   crs_to=utm_crs, 
                                                   always_xy = True).transform
        self.utm2geo = pyproj.Transformer.from_crs(crs_from=utm_crs, 
                                                   crs_to=4326, 
                                                   always_xy = True).transform
        
        # Get the rng
        self.rng = np.random.default_rng(seed)
        
        # Number of CPUs to use to generate paths
        self.num_cpus = num_cpus
    
    def get_paths(self) -> dict:
        """If existing cache file exists, loads it. If not, creates the paths
        and saves them, then returns the path dict of form acid:list of paths.
        """
        dirname = os.path.dirname(__file__)
        # Get the cache file name
        cache_path = os.path.join(dirname, f'cache/{self.scen_name}.pkl')
        # Check if there is an existing cache file
        if not self.force_gen and os.path.exists(cache_path) and not DEBUG:
            # Load it
            with open(cache_path, 'rb') as f:
                paths = pickle.load(f)
            # Simple check to see if this dictionary is complete. Otherwise, 
            # it will proceed with the path generation.
            if len(self.scenario) == len(paths):
                # return
                return paths
            else:
                print('Cache file incomplete, forcing path generation.')
        
        print(f'Generating paths for {self.scen_name}.')
        # We generate the paths for each aircraft
        if DEBUG:
            # Do single threaded to allow plots
            paths = {}
            for i, acid in enumerate(self.scenario.keys()):
                print(f'ACID: {acid} | {i+1}/{len(self.scenario)}')
                paths[acid] = self.make_paths(self.scenario[acid][1], 
                                            self.scenario[acid][2])
        else:
            # Create the input array of shape [[origin, dest, acid]...]
            input_arr = []
            for acid in self.scenario.keys():
                input_arr.append([self.scenario[acid][1], 
                                    self.scenario[acid][2], 
                                    acid])
            # Start the multiprocessing
            with mp.Pool(self.num_cpus) as p:
                results = list(tqdm.tqdm(p.imap(self.make_paths, 
                                                input_arr), 
                                                total=len(self.scenario)))
            # Transform the list into a dict
            paths = dict(results)
        # We cache them
        with open(cache_path, 'wb') as f:
            pickle.dump(paths, f)
            
        return paths
        
    def make_paths(self, args) -> list:
        """Generates the list of possible paths between the origin and
        destination nodes.

        Args:
            origin (int): Origin node
            destination (int): Destination node

        Returns:
            list: List of possible paths
        """
        # Unpack args
        origin, destination, acid = args
        # Get the shortest path for convenience
        sh_path = ox.shortest_path(self.G, origin, destination, 
                                       weight = "length")
        
        # Create the deterministic paths
        #ac_paths = self.make_deterministic_paths(origin, destination,sh_path)
        # Add the random routes
        ac_paths = self.make_random_routes(origin, destination, N_RAND_PATHS,
                                            sh_path)
        return [acid,ac_paths]
        
    def make_deterministic_paths(self, origin:int, destination:int, 
                                 sh_path:list = None) -> list:
        """Create a list of possible paths between the origin and destination
        nodes by setting the weight of groups of edges along the shortest route
        as a large value.

        Args:
            origin (int): Origin node
            destination (int): Destination node
            n_groups (list): 
                A list with the desired number of groups that the shortest path
                should be divided into. Multiple values possible.
            sh_path (list): 
                The shortest path between these two nodes to remove the need to
                recompute it here. If not provided, then it will be recomputed.

        Returns:
            list: List of paths
        """
        # Get the shortest path if it's not given
        if sh_path is None:
            # Get the shortest distance route
            sh_path = ox.shortest_path(self.G, origin, destination, 
                                       weight = "length")
            
        # Get the edges in the path
        sh_edges = list(zip(sh_path[:-1], sh_path[1:]))
        # Get the geometry in the path
        sh_geom = linemerge([self.edges.loc[(u, v, 0), 'geometry'] 
                             for u, v in sh_edges])

        if DEBUG:
            # Get the graph plot
            fig, ax = ox.plot_graph(self.G, node_alpha=0, edge_color='grey', 
                                    bgcolor='white', show = False)

        # Now let's generate some alternative paths
        alt_routes = []
        colori = 0
        # The number of parts we need to divide the path into. We go for powers
        # of two, and also include the case with one edge per group.
        n_groups = [2**x for x in range(int(np.log2(len(sh_edges))))] \
                    + [len(sh_edges)]
        for n in n_groups:
            # Divide list of edges into equal parts
            print(f'Dividing into {n} parts.')
            parts = self.split(sh_edges, n)
            # We go part by part and set the weight of each element of the part to a large value
            for part in parts:
                # Let's set the length of this guy to a lot
                G_local = copy.deepcopy(self.G)
                for u,v in part:
                    # 100 km should be enough
                    G_local.edges[u,v,0]['length'] = 100000
                # Now get the path again
                alt_route = ox.shortest_path(G_local, origin, destination, 
                                             weight = "length")
                # Is this route already in alternative routes?
                if alt_route in alt_routes:
                    continue
                else:
                    # Append it
                    alt_routes.append(alt_route)
                    # Plot it
                    alt_geom = linemerge([self.edges.loc[(i, j, 0), 'geometry'] 
                                          for i, j in zip(alt_route[:-1], 
                                                          alt_route[1:])])
                    if DEBUG:
                        ax.plot(alt_geom.xy[0], alt_geom.xy[1], 
                                color = colors[colori], linewidth = 2)
                    colori += 1
        if DEBUG:
            ax.plot(sh_geom.xy[0], sh_geom.xy[1], color = 'red', linewidth = 5)
            plt.show()
        return alt_routes
    
    def make_random_routes(self, origin:int, destination:int, n_paths:int, 
                           sh_path:list = None) -> list:
        """Creates randomly generated paths by selecting random nodes within
        a subgraph around the shortest path.

        Args:
            origin (int): Origin node
            destination (int): Destination node
            n_paths (int): The number of paths to generate
            sh_path (list): 
                The shortest path between these two nodes to remove the need to
                recompute it here. If not provided, then it will be recomputed.

        Returns:
            list: List of paths
        """
        # Get the shortest path if it's not given
        if sh_path is None:
            # Get the shortest distance route
            sh_path = ox.shortest_path(self.G, origin, destination, 
                                       weight = "length")
        
        # Get the edges in the path
        sh_edges = list(zip(sh_path[:-1], sh_path[1:]))
        # Get the geometry in the path
        sh_geom = linemerge([self.edges.loc[(u, v, 0), 'geometry'] 
                             for u, v in sh_edges])
        
        if DEBUG:
            # Get the graph plot
            fig, ax = ox.plot_graph(self.G, node_alpha=0, edge_color='grey', 
                                    bgcolor='white', show = False)

        # Get a subgraph based on a polygon buffer around the shortest path.
        sh_lon_utm, sh_lat_utm = self.geo2utm(sh_geom.xy[0], sh_geom.xy[1])
        sh_geom_utm = LineString(zip(sh_lon_utm, sh_lat_utm))
        
        # Buffer method
        # Just buffer the line
        #sh_poly_utm = sh_geom_utm.buffer(sh_geom_utm.length/BUFFER_FACTOR)
        
        # Weighted buffer method
        # Create the buffer such that its extent is greatest in the middle of
        # the route.
        buffers = []
        for i, lon, lat in zip(range(len(sh_lon_utm)),sh_lon_utm, sh_lat_utm):
            if i < len(sh_lon_utm)/2:
                b_factor = BUFFER_FACTOR * len(sh_lon_utm)/2/(i+1)**(0.75)
            else:
                b_factor = BUFFER_FACTOR * len(sh_lon_utm)/2/(len(sh_lon_utm)-i)**(0.75)
            # Make a point and buffer it in function of index and route length
            buffers.append(Point(lon,lat).buffer(sh_geom_utm.length/b_factor))
        sh_poly_utm = Polygon(convex_hull(unary_union(buffers)))
        sh_poly_lon, sh_poly_lat = self.utm2geo(
                                            sh_poly_utm.exterior.coords.xy[0], 
                                            sh_poly_utm.exterior.coords.xy[1])
        sh_poly = Polygon(zip(sh_poly_lon, sh_poly_lat))

        # Plot this polygon
        if DEBUG:
            ax.plot(sh_poly.exterior.coords.xy[0], 
                    sh_poly.exterior.coords.xy[1], color = 'orange', 
                    linewidth = 5)

        # Get the nodes within this polygon
        sh_poly_nodes = self.nodes[self.nodes.intersects(sh_poly)].index
        # Create the subgraph
        sh_G = self.G.subgraph(sh_poly_nodes)

        # The loop
        attempts = 0
        ac_paths = []
        colori = 0
        while attempts < PATH_ATTEMPTS and len(ac_paths) < n_paths:
            chosen_nodes = self.rng.choice(list(sh_G.nodes.keys()), N_RAND_NODES, 
                                           replace = True)
            
            # We want these nodes to not be somewhere in the shortest path
            if any([node in sh_path for node in chosen_nodes]):
                # Try again
                attempts+=1
                continue
            
            # We will pass through these nodes in order of path distance from
            # origin node.
            node_paths = [ox.shortest_path(sh_G, origin, node, 
                                    weight='length') for node in chosen_nodes]
            if any([x is None for x in node_paths]):
                # Try again
                attempts+=1
                continue
            
            node_paths_dist = [sum([self.edges.loc[(i,j,0), 'length'] 
                                    for i,j in zip(path[:-1], path[1:])]) 
                                    for path in node_paths]
            
            # Now sort these nodes in function of the length
            path_nodes = list(zip(node_paths_dist, chosen_nodes))
            path_nodes.sort()
            int_nodes = [x[1] for x in path_nodes]
            
            # Let's make a path through these nodes
            int_path = self.make_route_from_nodes(
                [origin] + int_nodes + [destination], sh_G
                )
            
            if int_path is None:
                # Try again
                attempts+=1
                continue
            
            # Extract the path and the geometry
            alt_path, alt_geom = int_path
            
            # Check the length requirement
            alt_length = sum([self.edges.loc[(i,j,0), 'length'] 
                                for i,j in zip(alt_path[:-1], alt_path[1:])])
            if alt_length > sh_geom_utm.length * PATH_LENGTH_FACTOR:
                # Try again
                attempts+=1
                continue
                
            # Plot this path
            if DEBUG:
                ax.plot(alt_geom.xy[0], alt_geom.xy[1], color = colors[colori], 
                        linewidth = 3)
            # Create the full node list and add it to the list of nodes
            ac_paths.append(alt_path)
            colori += 1
            attempts = 0
                
        # Plot the shortest route
        if DEBUG:
            ax.plot(sh_geom.xy[0], sh_geom.xy[1], color = 'red', linewidth = 5)
            plt.show()
            
        return ac_paths
    
    def make_route_from_nodes(self, node_list:list, 
                              G:nx.MultiDiGraph) -> list | None:
        """Creates a path in graph G that passes through each node in 
        node_list and is not self-intersecting.

        Args:
            node_list (list): List of nodes, in order.
            G (nx.MultiDiGraph): Graph
            edges (gpd.GeoDataFrame): Edges, for length calculation.

        Returns:
            list | None: 
                Returns the list, first element being the list of nodes, second 
                element being the linestring geometry. If route is not 
                possible, then returns None.
        """
        path = [node_list[0]] # Initialise with first node
        for node1,node2 in zip(node_list[:-1], node_list[1:]):
            small_path = ox.shortest_path(G, node1, node2,weight='length')
            if small_path is None:
                # Can't create path
                return None
            # Otherwise add these nodes to the path
            path += small_path[1:]
            
        # For this path, retrieve the geometry
        line = linemerge([self.edges.loc[(i,j,0), 'geometry'] for i,j in
                          zip(path[:-1], path[1:])])
        
        if not line.is_simple:
            # Line is self intersecting
            return None
        
        return path, line
        
        
    @staticmethod
    def split(a:list, n:int) -> list:
        """Split list a into n parts.

        Args:
            a (list): The list to be split.
            n (int): The number of parts.

        Returns:
            list: A list of sub-parts of a.
        """
        k, m = divmod(len(a), n)
        return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

    @staticmethod
    def convert_wgs_to_utm(lat: float, lon: float):
        """Based on lat and lng, return best utm epsg-code"""
        utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
        if len(utm_band) == 1:
            utm_band = '0'+utm_band
        if lat >= 0:
            epsg_code = '326' + utm_band
            return epsg_code
        epsg_code = '327' + utm_band
        return epsg_code