import numpy as np
import osmnx as ox
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from shapely.ops import linemerge
from shapely.geometry import LineString, Polygon
from shapely.plotting import plot_polygon
import copy
import random
import pyproj
import math

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def convert_wgs_to_utm( lat: float, lon: float):
        """Based on lat and lng, return best utm epsg-code"""
        utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
        if len(utm_band) == 1:
            utm_band = '0'+utm_band
        if lat >= 0:
            epsg_code = '326' + utm_band
            return epsg_code
        epsg_code = '327' + utm_band
        return epsg_code

rng = np.random.default_rng(42)

#Coord converter
city_centre_coords = [48.208758, 16.372449]
utm_crs = convert_wgs_to_utm(*city_centre_coords)
geo2utm = pyproj.Transformer.from_crs(crs_from=4326, crs_to=utm_crs, always_xy = True).transform
utm2geo = pyproj.Transformer.from_crs(crs_from=utm_crs, crs_to=4326, always_xy = True).transform

city = 'Vienna'

nodes = gpd.read_file(f'data/cities/{city}/streets.gpkg', layer='nodes')
edges = gpd.read_file(f'data/cities/{city}/streets.gpkg', layer='edges')

# set the indices 
edges.set_index(['u', 'v', 'key'], inplace=True)
nodes.set_index(['osmid'], inplace=True)

# ensure that it has the correct value
nodes['x'] = nodes['geometry'].apply(lambda x: x.x)
nodes['y'] = nodes['geometry'].apply(lambda x: x.y)

G = ox.graph_from_gdfs(nodes, edges)

colors = list(mcolors.CSS4_COLORS.keys())
random.shuffle(colors)

# Let's plan some routes
flight_intentions = np.genfromtxt(f'data/scenarios/Flight_intention_120_1.txt', delimiter = ';')

acidx = 0
origin = 4502 #flight_intentions[acidx][3]
destination = 4587 #flight_intentions[acidx][4]

# Get the shortest distance route
short_route = ox.shortest_path(G, origin, destination, weight = "length")
# Get the edges
short_edges = list(zip(short_route[:-1], short_route[1:]))
# Get the geometry
short_geom = linemerge([edges.loc[(u, v, 0), 'geometry'] for u, v in short_edges])

# Get the graph plot
fig, ax = ox.plot_graph(G, node_alpha=0, edge_color='grey', bgcolor='white', show = False)

# Now let's generate some alternative paths
alt_routes = []
colori = 0
part_step = 4
for n_parts in set(np.array(len(short_edges) / np.arange(1,len(short_edges),part_step), dtype = int).flatten()):
    # Divide list of edges into equal parts
    print(f'Dividing into {n_parts} parts.')
    parts = split(short_edges, n_parts)
    # We go part by part and set the weight of each element of the part to a large value
    for part in parts:
        # Let's set the length of this guy to a lot
        G_local = copy.deepcopy(G)
        for u,v in part:
            G_local.edges[u,v,0]['length'] = 10000
        # Now get the path again
        alt_route = ox.shortest_path(G_local, origin, destination, weight = "length")
        # Is this route already in alternative routes?
        if alt_route in alt_routes:
            continue
        else:
            # Append it
            alt_routes.append(alt_route)
            # Plot it
            alt_geom = linemerge([edges.loc[(i, j, 0), 'geometry'] for i, j in zip(alt_route[:-1], alt_route[1:])])
            ax.plot(alt_geom.xy[0], alt_geom.xy[1], color = 'black', linewidth = 2)#, linestyle = 'dashed')
            colori += 1

# Let's get a subgraph based on a polygon buffer around the shortest path.
short_lon_utm, short_lat_utm = geo2utm(short_geom.xy[0], short_geom.xy[1])
short_geom_utm = LineString(zip(short_lon_utm, short_lat_utm))
short_poly_utm = short_geom_utm.buffer(short_geom_utm.length/3)
short_poly_lon, short_poly_lat = utm2geo(short_poly_utm.boundary.coords.xy[0], short_poly_utm.boundary.coords.xy[1])
short_poly = Polygon(zip(short_poly_lon, short_poly_lat))

# Plot this polygon
#ax.plot(short_poly.boundary.coords.xy[0], short_poly.boundary.coords.xy[1], color = 'orange', linewidth = 5)

# Get the nodes within this polygon
short_poly_nodes = nodes[nodes.intersects(short_poly)].index
short_G = G.subgraph(short_poly_nodes)

# Select two random nodes, and route through them
attempts = 0
num_paths = 5
path_counter = 0
while attempts < 50 and path_counter < num_paths:
    node1, node2 = rng.choice(list(short_G.nodes.keys()), 2, replace = True)
    if node1 in short_route or node2 in short_route:
        # Try again
        attempts+=1
        continue
    
    # Compute shortest path to these two nodes from origin and destination\
    path1_o = ox.shortest_path(short_G, origin, node1, weight='length')
    path1_d = ox.shortest_path(short_G, node1, destination, weight='length')
    path2_o = ox.shortest_path(short_G, origin, node2, weight='length')
    path2_d = ox.shortest_path(short_G, node2, destination, weight='length')

    if path1_o is None or path1_d is None or path2_o is None or path2_d is None:
        # Try again
        attempts+=1
        continue
    
    # Get the lengths of these paths
    path1_ol = sum([edges.loc[(i, j, 0), 'length'] for i, j in zip(path1_o[:-1], path1_o[1:])])
    path1_dl = sum([edges.loc[(i, j, 0), 'length'] for i, j in zip(path1_d[:-1], path1_d[1:])])
    path2_ol = sum([edges.loc[(i, j, 0), 'length'] for i, j in zip(path2_o[:-1], path2_o[1:])])
    path2_dl = sum([edges.loc[(i, j, 0), 'length'] for i, j in zip(path2_d[:-1], path2_d[1:])])
    
    # We want either one of these to be smaller than the original route path
    if not(path1_ol < short_geom_utm.length and \
        path1_dl < short_geom_utm.length and \
            path2_ol < short_geom_utm.length and \
                path2_dl < short_geom_utm.length):
        # try again
        attempts += 1
        continue
    
    # Ok now make the shortest path
    if path1_ol < path2_ol:
        # node 1 first
        part1 = ox.shortest_path(short_G, origin, node1, weight='length')
        part2 = ox.shortest_path(short_G, node1, node2, weight='length')
        part3 = ox.shortest_path(short_G, node2, destination, weight='length')
        # make path
        part1_g = linemerge([edges.loc[(i, j, 0), 'geometry'] for i, j in zip(part1[:-1], part1[1:])])
        part2_g = linemerge([edges.loc[(i, j, 0), 'geometry'] for i, j in zip(part2[:-1], part2[1:])])
        part3_g = linemerge([edges.loc[(i, j, 0), 'geometry'] for i, j in zip(part3[:-1], part3[1:])])
        alt_geom = linemerge([part1_g, part2_g, part3_g])
    else:
        # node 2 first
        part1 = ox.shortest_path(short_G, origin, node2, weight='length')
        part2 = ox.shortest_path(short_G, node2, node1, weight='length')
        part3 = ox.shortest_path(short_G, node1, destination, weight='length')
        # make path
        part1_g = linemerge([edges.loc[(i, j, 0), 'geometry'] for i, j in zip(part1[:-1], part1[1:])])
        part2_g = linemerge([edges.loc[(i, j, 0), 'geometry'] for i, j in zip(part2[:-1], part2[1:])])
        part3_g = linemerge([edges.loc[(i, j, 0), 'geometry'] for i, j in zip(part3[:-1], part3[1:])])
        alt_geom = linemerge([part1_g, part2_g, part3_g])
    # Plot this guy
    ax.plot(alt_geom.xy[0], alt_geom.xy[1], color = colors[colori], linewidth = 5, linestyle = 'dashed')
    colori += 1
    path_counter += 1
    attempts = 0
        
# Plot the shortest route
ax.plot(short_geom.xy[0], short_geom.xy[1], color = 'red', linewidth = 5)
plt.show()