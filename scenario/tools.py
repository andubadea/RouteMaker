import os
import re
import numpy as np
from time import strftime, gmtime
from typing import Any


def natural_sort(l:list) -> list: 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def kwikqdr(lata: float, lona: float, latb: float, lonb: float)-> float:
        """Gives quick qdr[deg] from lat/lon."""
        dlat    = np.radians(latb - lata)
        dlon    = np.radians(((lonb - lona)+180)%360-180)
        cavelat = np.cos(np.radians(lata + latb) * 0.5)
        qdr = np.degrees(np.arctan2(dlon * cavelat, dlat)) % 360
        return qdr

def tim2txt(t:float) -> str:
    """Convert time to timestring: HH:MM:SS"""
    return strftime("%H:%M:%S", gmtime(t))

def makescen(model:Any) -> None:
    """Generates a scenario using the path and altitude allocation provided
    by the optimisation.

    Args:
        model (PathModel): The path model
    """
    # Extract the needed information
    F = model.params.F
    K_f = model.params.K_f
    Y = model.params.Y
    fl_size = model.params.fl_size
    nodes = model.params.city.nodes
    idx2acid = model.params.idx2acid
    path_dict = model.params.path_dict
    v_turn = model.params.v_turn
    
    # Imperial units
    m2ft = 1/0.3048
    mps2kts = 0.514444
    
    # Generate scenario lines from model solution.
    z_sol = []
    for f in F:
        for k in K_f[f]:
            for y in Y:
                if model.problem.z[f,k,y].X > 0.5:
                    z_sol.append((f,k,y))
                    
    # Generate the scenario text for each entry
    scen_text = ''
    
    for f,k,y in z_sol:
        # Find the acid
        acid = idx2acid[f]
        # Get its route nodes
        rte_nodes = path_dict[acid]['paths'][k]
        # Get the departure time
        dep_time = tim2txt(path_dict[acid]['times'][k][0])
        # Get the departure altitude
        alt = y * fl_size * m2ft
        
        # Compile the create command
        prev_lon = nodes.at[rte_nodes[0],'geometry'].x
        prev_lat = nodes.at[rte_nodes[0],'geometry'].y
        prev_node = rte_nodes[0]
        next_lon = nodes.at[rte_nodes[1],'geometry'].x
        next_lat = nodes.at[rte_nodes[1],'geometry'].y
        street_number = f'{rte_nodes[0]}-{rte_nodes[1]}'
        hdg = kwikqdr(prev_lat, prev_lon, next_lat, next_lon)
        # Split it in too parts for it to be tidy
        scen_text += f'{dep_time}>M22CRE {acid},M600,{prev_lat},'
        scen_text += f'{prev_lon},{hdg},{alt},{v_turn * mps2kts}'
        
        # Now go node by node
        for i, node in enumerate(rte_nodes):
            if i == 0:
                # Add origin
                scen_text += f',{prev_lat},{prev_lon},,,,FLYBY,{street_number}'
                continue
            lon = nodes.at[node,'geometry'].x 
            lat = nodes.at[node,'geometry'].y
            # Street number
            street_number = f'{prev_node}-{node}'
            if i == len(rte_nodes)-1:
                # Last waypoint, it's by default marked as a turn and add a \n
                scen_text += f',{lat},{lon},,,,FLYTURN,{street_number}\n'
            else:
                # Check if turn
                next_node = rte_nodes[i+1]
                next_lat = nodes.at[next_node,'geometry'].x
                next_lon = nodes.at[next_node,'geometry'].y
                # Get the angle
                a1 = kwikqdr(prev_lat, prev_lon, lat, lon)
                a2 = kwikqdr(lat, lon, next_lat, next_lon)
                angle = abs(a2-a1)
                if angle>180:
                    angle=360-angle
                    
                # This is a turn if angle is greater than 25
                if angle > 25:
                    scen_text += f',{lat},{lon},,,,FLYTURN,{street_number}'
                else:
                    scen_text += f',{lat},{lon},,,,FLYBY,{street_number}'
                    
            # Store prev values
            prev_lat = lat
            prev_lon = lon
            prev_node = node
        
    # Save this scenario
    with open(model.notypename + '.scn', 'w') as f:
        f.write(scen_text)