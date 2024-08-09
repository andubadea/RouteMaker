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
colors = ['red','blue','green','orange','purple']
DEBUG = False
DEBUG2 = True
PLOT = False

# Path generation
N_PATHS = 5 # Number of non-random paths to generate
N_RAND_PATHS = 2 # Number of random paths to generate
BUFFER_FACTOR = 2 # Higher means random path subgraph is smaller
PATH_ATTEMPTS = 50 # Higher means more attempts per random path
N_RAND_NODES = 1 # Number of random intermediate nodes
# Length limit for random routes in function of shortest route length
PATH_LENGTH_FACTOR = 1.25
PATH_LENGTH_RANDOM_FACTOR = 1.5
PATH_SIMILARITY_FACTOR = 0.8
TURN_ANGLE = 25
n_groups = [2,1]

class PathMaker():
    """Module to make alternative paths for each aircraft in a scenario.
    Class instance is specific to one scenario.
    """
    def __init__(self, **kwargs) -> None:
        """Class that creates alternative paths for each aircraft in one 
        scenario.
        """
        self.__dict__.update(kwargs)
        
        # Create the coordinate transformers
        city_centre = self.nodes.to_crs('+proj=cea').dissolve(
                                            ).centroid.to_crs(self.nodes.crs)
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
        
        self.get_paths()
    
    def get_paths(self) -> dict:
        """If existing cache file exists, loads it. If not, creates the paths
        and saves them, then returns the path dict of form acid:list of paths.
        """
        # Get the cache file name
        cache_path = os.path.join(
            f'data/cities/{self.city.name}/cache/{self.scen_name}_paths.pkl')
        
        # We generate the paths for each aircraft
        if DEBUG or self.num_cpus == 1:
            print(f'> Generating paths for {self.scen_name}...')
            # Do single threaded to allow plots
            paths = {}
            for i, acid in enumerate(self.scenario.keys()):
                #print(f'ACID: {acid} | {i+1}/{len(self.scenario)}')
                paths[acid] = self.make_paths([self.scenario[acid][1], 
                                            self.scenario[acid][2],
                                            acid])[1]
        else:
            print(f'> Generating paths for {self.scen_name}...')
            # Create the input array of shape [[origin, dest, acid]...]
            input_arr = []
            for acid in self.scenario.keys():
                input_arr.append([self.scenario[acid][1], 
                                    self.scenario[acid][2], 
                                    acid])
            # Start the multiprocessing
            with mp.Pool(self.num_cpus) as p:
                results = list(tqdm.tqdm(p.imap(self.make_paths, input_arr), 
                                    total=len(input_arr)))
            # Transform the list into a dict
            paths = dict(results)
        # We cache them
        print('> Saving cache file...')
        with open(cache_path, 'wb') as f:
            pickle.dump(paths, f)
        print('> Cache file saved.')
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
        ac_paths = self.make_deterministic_paths(origin, destination,sh_path)
        if len(ac_paths) < N_PATHS - N_RAND_PATHS:
            # We came short of the target, compensate with random paths
            n_rands = N_RAND_PATHS + (N_PATHS - N_RAND_PATHS - len(ac_paths))
        else:
            n_rands = N_RAND_PATHS

        # Add the random paths
        ac_paths += self.make_random_routes(origin, destination, 
                                            n_rands, sh_path, ac_paths)
        if len(ac_paths) > (N_PATHS):
            # We overshot, take the required number of paths
            ac_paths = ac_paths[:N_PATHS]
        elif len(ac_paths) < (N_PATHS):
            # Add the shortest path a few more times and done
            ac_paths += [ac_paths[0]]*(N_PATHS-len(ac_paths))
            
        # Plot em
        if DEBUG2:
            fig, ax = ox.plot_graph(self.G, node_alpha=0, edge_color='grey', 
                                        bgcolor='white', show = False)
            for i in range(len(ac_paths)):
                # Plot it
                if i == 0:
                    # Shortest path
                    label = 'Shortest path'
                    linewidth = 6
                    linestyle = 'solid'
                # elif i == len(ac_paths)-1:
                #     # Random route
                #     label = 'Random path'
                #     linewidth = 3
                #     linestyle = 'dotted'
                else:
                    label = f'Alternative {i}'
                    linewidth = 3
                    linestyle = 'dashed'
                geom = linemerge([self.edges.loc[(i, j, 0), 'geometry'] 
                                    for i, j in zip(ac_paths[i][:-1], 
                                                    ac_paths[i][1:])])
                ax.plot(geom.xy[0], geom.xy[1], 
                        color = colors[i], linewidth = linewidth,
                        label = label, linestyle = linestyle)
            
            # ax.plot(self.sh_poly.exterior.coords.xy[0], 
            #         self.sh_poly.exterior.coords.xy[1], color = 'black', 
            #         linewidth = 2, label = 'Random path polygon')
            
            plt.legend()
            plt.show()
            plt.close()
        
        # Create the complete path dictionary by adding time at each edge.
        ac_paths_dict = self.create_path_dict(acid, ac_paths)
        return (acid, ac_paths_dict)
        
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

        if DEBUG or PLOT:
            # Get the graph plot
            fig, ax = ox.plot_graph(self.G, node_alpha=0, edge_color='grey', 
                                    bgcolor='white', show = False)
            colori = 0

        # Now let's generate some alternative paths
        alt_routes = [sh_path]
        for n in n_groups:
            # Stop earlier if we have enough routes
            if len(alt_routes) >= N_PATHS - N_RAND_PATHS:
                break
            # We don't want to touch the first 10% and last 10% of the path
            ten_p_idx = int(len(sh_edges)/10)
            # Divide list of edges into equal parts
            parts = self.split(sh_edges[ten_p_idx:-ten_p_idx], n)
            # We go part by part and set the weight of each element of the part to a large value
            skips = 0
            for part in parts:
                # Stop earlier if we have enough routes
                if len(alt_routes) >= N_PATHS - N_RAND_PATHS or skips >= n/2:
                    break
                # Let's set the length of this guy to a lot
                G_local = copy.deepcopy(self.G)
                for u,v in part:
                    # Avoid setting this for the 10% and last 10% of the path
                    # 100 km should be enough
                    G_local.edges[u,v,0]['length'] = 100000
                # Now get the path again
                alt_route = ox.shortest_path(G_local, origin, destination, 
                                             weight = "length")
                # Is this route already in alternative routes?
                if alt_route in alt_routes:
                    # Skip this route
                    skips += 1
                    continue
                
                # Check how similar it is compared to the shortest path
                similarity = sum([node in sh_path 
                                  for node in alt_route]) / len(alt_route)
                
                if similarity > PATH_SIMILARITY_FACTOR:
                    # Skip this route
                    skips += 1
                    continue
                
                # Append it
                alt_routes.append(alt_route)
                
                if DEBUG or PLOT:
                    # Plot it
                    alt_geom = linemerge([self.edges.loc[(i, j, 0), 'geometry'] 
                                        for i, j in zip(alt_route[:-1], 
                                                        alt_route[1:])])
                    ax.plot(alt_geom.xy[0], alt_geom.xy[1], 
                            color = colors[colori], linewidth = 3,
                            label = f'Alternative {colori + 1}')
                    colori += 1
        if DEBUG or PLOT:
            # Get the geometry in the path
            sh_geom = linemerge([self.edges.loc[(u, v, 0), 'geometry'] 
                             for u, v in sh_edges])
            ax.plot(sh_geom.xy[0], sh_geom.xy[1], color = 'red', linewidth = 6,
                    label = 'Shortest path')
            plt.legend()
            plt.show()
        return alt_routes
    
    def make_random_routes(self, origin:int, destination:int, n_paths:int, 
                           sh_path:list = None, ac_paths:list = []) -> list:
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
        
        if DEBUG or PLOT:
            # Get the graph plot
            fig, ax = ox.plot_graph(self.G, node_alpha=0, edge_color='grey', 
                                    bgcolor='white', show = False)

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

        # Plot this polygon
        if DEBUG or PLOT:
            ax.plot(sh_poly.exterior.coords.xy[0], 
                    sh_poly.exterior.coords.xy[1], color = 'orange', 
                    linewidth = 5)
            
        self.sh_poly = sh_poly

        # Get the nodes within this polygon
        sh_poly_nodes = self.nodes[self.nodes.intersects(sh_poly)].index
        # Create the subgraph
        sh_G = self.G.subgraph(sh_poly_nodes)
        # Within the subgraph, increase the weights of the edges of the shortest
        # path to discourage their use.
        # for u,v in sh_edges:
        #     # Make these edges less desirable
        #     sh_G.edges[u,v,0]['length'] *= 1.1

        # The loop
        attempts = 0
        added = 0
        colori = 0
        while attempts < PATH_ATTEMPTS and added < n_paths:
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
            if alt_length > sh_geom_utm.length * PATH_LENGTH_RANDOM_FACTOR:
                # Try again
                attempts+=1
                continue
            
            if alt_path in ac_paths:
                # Try again
                attempts+=1
                continue
                
            # Plot this path
            if DEBUG or PLOT:
                ax.plot(alt_geom.xy[0], alt_geom.xy[1], color = colors[colori], 
                        linewidth = 3)
            # Create the full node list and add it to the list of nodes
            ac_paths.append(alt_path)
            added += 1
            colori += 1
            attempts = 0
                
        # Plot the shortest route
        if DEBUG or PLOT:
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
    
    def create_path_dict(self, acid:str, ac_paths:list) -> dict:
        """Creates the path dictionary for an aircraft.

        Args:
            acid (str): Aircraft ID
            ac_paths (list): List of paths in nodes.

        Returns:
            List: 
                List of lists corresponding to the ac_paths list of lists with
                the estimated time at each node for each path.
        """
        path_times = [self.calc_path_times(acid, path) for path in ac_paths]
        return {'paths':ac_paths, 'times':path_times}
    
    
    def calc_path_times(self, acid:str, path:list) -> list:
        # Get the aircraft departure time
        dep_time = self.scenario[acid][0]
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
        prev_turn_time = self.t_dcel
        prev_turn_dist = self.d_dcel
        prev_wplon, prev_wplat = path_geom.coords[0]
        for i in range(1, len(path_geom.coords)):
            # If the previous waypoint was a turn one, add extra time
            leg_time = prev_turn_time
            wplon, wplat = path_geom.coords[i]
            # If destination, it's a turn
            if i == (len(path_geom.coords)-1):
                # Get the current distance to this waypoint
                _, totaldist = self.kwikqdrdist(prev_wplat, prev_wplon, 
                                      wplat, wplon)
                cruise_dist = totaldist - self.d_dcel - prev_turn_dist
                if cruise_dist < 0:
                    # Just take it as 0
                    cruise_dist = 0
                time_cruise = cruise_dist / self.v_cruise
                
                # Finally add to estimated time
                leg_time += time_cruise + self.t_dcel
                timestamps.append(timestamps[-1] + leg_time)
                break
            
            next_wplon, next_wplat = path_geom.coords[i+1]
            a1,_=self.kwikqdrdist(prev_wplat,prev_wplon,wplat,wplon)
            a2,_=self.kwikqdrdist(wplat,wplon,next_wplat,next_wplon)
            angle=abs(a2-a1)
            if angle > 25:
                time_turn = angle/self.yaw_r*0
                # Get the current distance to this waypoint
                _, totaldist = self.kwikqdrdist(prev_wplat, prev_wplon, 
                                      wplat, wplon)
                cruise_dist = totaldist - self.d_dcel - prev_turn_dist
                if cruise_dist < 0:
                    # Didn't have time to cruise. So we accelerated and
                    # decelerated for half the distance each
                    # Do we both accel and decel?
                    if prev_turn_dist > 0:
                        leg_time = 2*((totaldist*self.a_hoz/2+self.v_turn**2
                                )**0.5 - self.v_turn)/self.a_hoz + time_turn/2
                        timestamps.append(timestamps[-1] + leg_time)
                        prev_turn_time = self.t_dcel + time_turn/2
                        prev_turn_dist = self.d_dcel
                        continue
                    else:
                        # We only decelerated
                        leg_time = ((totaldist*self.a_hoz+self.v_turn**2
                                )**0.5 - self.v_turn)/self.a_hoz + time_turn/2
                        timestamps.append(timestamps[-1] + leg_time)
                        prev_turn_time = self.t_dcel + time_turn/2
                        prev_turn_dist = self.d_dcel
                        continue
                        
                        
                time_cruise = cruise_dist / self.v_cruise
                
                # Finally add to estimated time
                leg_time += time_cruise + self.t_dcel + time_turn/2
                timestamps.append(timestamps[-1] + leg_time)
                
                # Acceleration is symmetric to deceleration
                prev_turn_time = self.t_dcel + time_turn/2
                prev_turn_dist = self.d_dcel
                prev_wplat, prev_wplon = wplat, wplon
            else:
                # Normal cruise calculation
                _, totaldist = self.kwikqdrdist(prev_wplat, prev_wplon, 
                                      wplat, wplon)
                cruise_dist = totaldist - prev_turn_dist
                if cruise_dist < 0:
                    cruise_dist = 0
                time_cruise = cruise_dist / self.v_cruise
                leg_time += time_cruise
                timestamps.append(timestamps[-1] + leg_time)
                
                prev_turn_time = 0
                prev_turn_dist = 0
                prev_wplat, prev_wplon = wplat, wplon
        
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