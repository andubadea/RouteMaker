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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from shapely import convex_hull
from shapely.ops import linemerge, unary_union, split
from shapely.geometry import LineString, Polygon, Point


# For plotting
colors = list(mcolors.CSS4_COLORS.keys())
random.shuffle(colors)
DEBUG = False

# Path generation
N_RAND_PATHS = 2 # Number of random paths to generate
N_PATHS = 10 # Number of non-random paths to generate
BUFFER_FACTOR = 1.5 # Higher means random path subgraph is smaller
PATH_ATTEMPTS = 50 # Higher means more attempts per random path
N_RAND_NODES = 1 # Number of random intermediate nodes
# Length limit for random routes in function of shortest route length
PATH_LENGTH_FACTOR = 2
PATH_SIMILARITY_FACTOR = 0.9
TURN_ANGLE = 25

class PathCacheMaker:
    """Makes the path for all scenarios in one go.
    """
    def __init__(self, **kwargs) -> None:
        """Class that creates alternative paths for each aircraft in one 
        scenario.
        """
        self.__dict__.update(kwargs)
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
        self.rng = np.random.default_rng(self.seed)
        
        # Vehicle properties
        self.t_dcel = (self.v_cruise - self.v_turn) / self.a_hoz
        self.d_dcel = (self.v_cruise**2 - self.v_turn**2) / (2*self.a_hoz)
    
    def create_cache(self, scenarios:list) -> None:
        """Create the caches for all the scenarios in the list. The scenario
        list must be of form [[scenario dict, name]...]
        """
        with mp.Pool(self.num_cpus) as p:
            _ = list(tqdm.tqdm(p.imap(self.get_paths, scenarios), 
                                                total=len(scenarios)))
        return
        
        
    def get_paths(self, args) -> None:
        """Create the paths for one scenario.
        """
        scenario,name = args
        dirname = os.path.dirname(__file__)
        # Get the cache file name
        cache_path = os.path.join(dirname, f'cache/{name}.pkl')
        
        # We generate the paths for each aircraft
        paths = {}
        for i, acid in enumerate(scenario.keys()):
            #print(f'ACID: {acid} | {i+1}/{len(scenario)}')
            paths[acid] = self.make_paths([scenario[acid][1], 
                                        scenario[acid][2],
                                        scenario[acid][0]])
            
        # We cache them
        with open(cache_path, 'wb') as f:
            pickle.dump(paths, f)
            
        return
        
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
        origin, destination, dep_time = args
        # Get the shortest path for convenience
        sh_path = ox.shortest_path(self.G, origin, destination, 
                                       weight = "length")
        
        # Create the deterministic paths
        ac_paths = self.make_deterministic_paths(origin, destination,sh_path)
        if len(ac_paths) > (N_PATHS - N_RAND_PATHS):
            # Take the first N_PATHS routes
            ac_paths = ac_paths[:(N_PATHS - N_RAND_PATHS)]

        # Add some random routes up to N_PATHS
        ac_paths += self.make_random_routes(origin, destination, 
                                            N_PATHS - len(ac_paths), 
                                            sh_path)
        
        # Create the complete path dictionary by adding time at each edge.
        ac_paths_dict = self.create_path_dict(ac_paths, dep_time)
        return ac_paths_dict
        
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

        # Now let's generate some alternative paths
        alt_routes = [sh_path]
        # The number of parts we need to divide the path into. We go for powers
        # of two, and also include the case with one edge per group.
        n_groups = [2**x for x in range(int(np.log2(len(sh_edges))))] \
                    + [len(sh_edges)]
        for n in n_groups:
            # Stop earlier if we have enough routes
            if len(alt_routes) >= N_PATHS - N_RAND_PATHS:
                break
            # Divide list of edges into equal parts
            parts = self.split(sh_edges, n)
            # We go part by part and set the weight of each element of the part to a large value
            for part in parts:
                # Stop earlier if we have enough routes
                if len(alt_routes) >= N_PATHS - N_RAND_PATHS:
                    break
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
                
                # Check how similar it is compared to the shortest path
                similarity = sum([node in sh_path 
                                  for node in alt_route]) / len(alt_route)
                
                if similarity > PATH_SIMILARITY_FACTOR:
                    # Skip this route
                    continue
                
                # Append it
                alt_routes.append(alt_route)
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

        # Get a subgraph based on a polygon buffer around the shortest path.
        sh_lon_utm, sh_lat_utm = self.geo2utm(sh_geom.xy[0], sh_geom.xy[1])
        sh_geom_utm = LineString(zip(sh_lon_utm, sh_lat_utm))
        
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

        # Get the nodes within this polygon
        sh_poly_nodes = self.nodes[self.nodes.intersects(sh_poly)].index
        # Create the subgraph
        sh_G = self.G.subgraph(sh_poly_nodes)
        # Within the subgraph, increase the weights of the edges of the shortest
        # path to discourage their use.
        for u,v in sh_edges:
            # Make these edges undesirable
            sh_G.edges[u,v,0]['length'] *= 3

        # The loop
        attempts = 0
        ac_paths = []
        while attempts < PATH_ATTEMPTS and len(ac_paths) < n_paths:
            chosen_nodes = self.rng.choice(list(sh_G.nodes.keys()), 
                                           N_RAND_NODES)
            
            # We want these nodes to not be in the shortest path
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
                
            # Create the full node list and add it to the list of nodes
            ac_paths.append(alt_path)
            attempts = 0
            
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
    
    def create_path_dict(self, ac_paths:list, dep_time:float) -> dict:
        """Creates the path dictionary for an aircraft.

        Args:
            acid (str): Aircraft ID
            ac_paths (list): List of paths in nodes.

        Returns:
            List: 
                List of lists corresponding to the ac_paths list of lists with
                the estimated time at each node for each path.
        """
        path_times = [self.calc_path_times(path, dep_time) for path in ac_paths]
        return {'paths':ac_paths, 'times':path_times}
            
        
    def calc_path_times(self, path:list, dep_time:float) -> list:
        """Calculates the time at which an aircraft will be at each node
        of this path.

        Args:
            acid (str): Aircraft ID
            path (list): List of nodes.

        Returns:
            list: List of times for each node.
        """
        # Initialise the times list with the departure time
        timestamps = [dep_time]
        # Get the path geometry and the indices within it of the nodes
        node_coord_idx = [0] # First node will of course have idx 0
        edge_geoms = []
        for u,v in zip(path[:-1], path[1:]):
            edge_line = self.edges.loc[(u,v,0), 'geometry']
            node_coord_idx.append(node_coord_idx[-1] + len(edge_line.xy[0])-1)
            edge_geoms.append(edge_line)
            
        path_geom = linemerge(edge_geoms)
        prev_turn = False
        # Now find the idx's of the turns in the geometry
        for i in range(1, len(path_geom.coords)-1):
            lon_prev, lat_prev = path_geom.coords[i-1]
            lon_next, lat_next = path_geom.coords[i+1]
            lon, lat = path_geom.coords[i]
            # Get the angle and distance
            a1, d1=self.kwikqdrdist(lat_prev,lon_prev,lat,lon)
            a2, d2=self.kwikqdrdist(lat,lon,lat_next,lon_next)
            angle=abs(a2-a1)

            if angle>180:
                angle=360-angle
                
            # This is a turn if angle is greater than 25
            if angle < 25:
                # This isn't a turn, but did we have a turn before?
                if prev_turn:
                    # Then we must account for the time to accelerate
                    d1 -= self.d_dcel
                    if d1 < 0:
                        d1 = 0 # We never accelerated enough then
                    timestamps.append(timestamps[-1] + d1/self.v_cruise 
                                      + self.t_dcel)
                else:
                    timestamps.append(timestamps[-1] + d1/self.v_cruise)
                prev_turn = False
                
            else:
                # This is a turn, was the previous also a turn?
                if prev_turn:
                    # Account for double turn
                    d1 -= self.d_dcel * 2
                    if d1 < 0:
                        d1 = 0
                    timestamps.append(timestamps[-1] + d1/self.v_cruise 
                                      + 2 * self.t_dcel)
                else:
                    # Only account for one turn
                    d1 -= self.d_dcel
                    if d1 < 0:
                        d1 = 0 # We never accelerated enough then
                    timestamps.append(timestamps[-1] + d1/self.v_cruise 
                                      + self.t_dcel)
                prev_turn = True

            if i == len(path_geom.coords)-1:
                # Also add the destination time
                if prev_turn:
                    d2 -= self.d_dcel
                    if d2 < 0:
                        d2 = 0 # We never accelerated enough then
                    timestamps.append(timestamps[-1] + d2/self.v_cruise 
                                      + self.t_dcel)
                else:
                    timestamps.append(timestamps[-1] + d2/self.v_cruise)
        return timestamps
    
    @staticmethod
    def kwikqdrdist(lata: float, lona: float, latb: float, lonb: float)-> float:
        """Gives quick and dirty qdr[deg] and dist [m]
        from lat/lon. (note: does not work well close to poles)"""
        re      = 6371000.  # radius earth [m]
        dlat    = np.radians(latb - lata)
        dlon    = np.radians(((lonb - lona)+180)%360-180)
        cavelat = np.cos(np.radians(lata + latb) * 0.5)
        qdr     = np.degrees(np.arctan2(dlon * cavelat, dlat)) % 360
        dangle  = np.sqrt(dlat * dlat + dlon * dlon * cavelat * cavelat)
        dist    = re * dangle
        return qdr, dist
        
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
    
    @staticmethod
    def perpendicular_line(line:LineString, length:float) -> LineString:
        """Creates a linestring perpendicular to another line.

        Args:
            line (LineString): Original line.
            length (float): Length of the new perpendicular line.

        Returns:
            LineString: Perpendicular line.
        """
        left = line.offset_curve(length / 2)
        right = line.offset_curve(-length / 2)
        c = left.boundary.geoms[1]
        d = right.boundary.geoms[0]
        return LineString([c, d])