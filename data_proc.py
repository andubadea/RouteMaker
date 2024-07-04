log_file = 'data/output/outh.log'

with open(log_file, 'r') as f:
    lines = f.readlines()
    
data_dict = {}

time_flow_W = None
demand_level = None
repetition = None
correct_tw = False
for line in lines:
    if line [0] == '@':
        #Set the flow time window
        time_flow_W = int(line.split('s @')[0].split(': ')[-1])
    elif line[0] == '#' and '!' not in line:
        # Set the density and repetition
        demand_level = int(line.split('_')[2])
        repetition = int(line.split('_')[3][0])
    elif line[0] == '*' and '5700s' in line:
        # This is the correct time window
        correct_tw = True
    elif correct_tw and 'Best objective' in line:
        # Extract the gap
        gap = float(line.split(' gap ')[-1].replace('%',''))
        data_dict[time_flow_W,demand_level,repetition] = gap
        # Reset correct tw
        correct_tw = False
    else:
        continue
    
print(data_dict)