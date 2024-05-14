import numpy as np
import osmnx as ox
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from shapely.ops import linemerge
import copy

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

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
for n_parts in np.array(len(short_edges) / np.arange(1,len(short_edges),4), dtype = int):
    # Divide list of edges into equal parts
    print(n_parts)
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
            ax.plot(alt_geom.xy[0], alt_geom.xy[1], color = colors[colori], linewidth = 5)
            colori += 1

print(f'Number of alternative routes: {len(alt_routes)}')
        
# Plot the shortest route
ax.plot(short_geom.xy[0], short_geom.xy[1], color = 'red', linewidth = 5)
plt.show()