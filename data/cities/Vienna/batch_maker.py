"""Make scenarios based on base scenarios and generate a batch file."""

import os

base_dir = 'ENAC/Base'
base_scens = [x for x in os.listdir(base_dir) if '.scn' in x]

wind_mags = [2,4,6]
wind_dirs = [0,90,180,270]
delay_mags = [10,30]
delay_prob = 30
CDRs = [['M22CD', 'M22CR'], ['DEFENSIVECD', 'DEFENSIVECR'], ['M22CD', 'HALTSBCR']]
batch_lines = ''

for base_scen in base_scens:
    # We test them with SB + DB, at various winds and delays
    for cdmethod, crmethod in CDRs:
        # Make a new file
        scen_name = base_scen.replace('.scn','')
        scen_name += f'_0'
        scen_name += f'_0'
        scen_name += f'_0'
        scen_name += f'_{crmethod}'
        scen_name += f'.scn'
        
        # Create the lines
        lines = '00:00:00>SEED 42\n00:00:00>ASAS ON\n'
        lines += f'00:00:00>RESO {crmethod}\n'
        lines += f'00:00:00>CDMETHOD {cdmethod}\n'
        lines += '00:00:00>IMPL WINDSIM M22WIND\n'
        lines += f'00:00:00>SETM22WIND 0 0\n'
        lines += '00:00:00>SETM22DELAY 0 0\n'
        lines += f'00:00:00>ENABLESPAWNPROTECTION\n'
        lines += '00:00:00>STARTLOGS\n'
        lines += f'00:00:00>PCALL {base_dir}/{base_scen}\n'
        lines += '00:00:00>SCHEDULE 02:00:00 DELETEALL\n'
        lines += '00:00:00>SCHEDULE 02:00:01 HOLD'
        
        # Open the new file
        with open(f'ENAC/{scen_name}', 'w') as f:
            f.write(lines)
            
        # Write the batch lines as well
        batch_lines += f'00:00:00.00>SCEN {scen_name.replace('.scn','')}\n'
        batch_lines += f'00:00:00.00>PCALL ENAC/{scen_name}\n'
        batch_lines += f'00:00:00.00>FF\n\n'
        
        for wind_mag in wind_mags:
            for wind_dir in wind_dirs:
                # Make a new file
                scen_name = base_scen.replace('.scn','')
                scen_name += f'_{wind_mag}'
                scen_name += f'_{wind_dir}'
                scen_name += f'_0'
                scen_name += f'_{crmethod}'
                scen_name += f'.scn'
                
                # Create the lines
                lines = '00:00:00>SEED 42\n00:00:00>ASAS ON\n'
                lines += f'00:00:00>RESO {crmethod}\n'
                lines += f'00:00:00>CDMETHOD {cdmethod}\n'
                lines += '00:00:00>IMPL WINDSIM M22WIND\n'
                lines += f'00:00:00>SETM22WIND {wind_mag} {wind_dir}\n'
                lines += '00:00:00>SETM22DELAY 0 0\n'
                lines += f'00:00:00>ENABLESPAWNPROTECTION\n'
                lines += '00:00:00>STARTLOGS\n'
                lines += f'00:00:00>PCALL {base_dir}/{base_scen}\n'
                lines += '00:00:00>SCHEDULE 02:00:00 DELETEALL\n'
                lines += '00:00:00>SCHEDULE 02:00:01 HOLD'
                
                # Open the new file
                with open(f'ENAC/{scen_name}', 'w') as f:
                    f.write(lines)
                    
                # Write the batch lines as well
                batch_lines += f'00:00:00.00>SCEN {scen_name.replace('.scn','')}\n'
                batch_lines += f'00:00:00.00>PCALL ENAC/{scen_name}\n'
                batch_lines += f'00:00:00.00>FF\n\n'
        
        for delay_mag in delay_mags:
            # Make a new file
            scen_name = base_scen.replace('.scn','')
            scen_name += f'_0'
            scen_name += f'_0'
            scen_name += f'_{delay_mag}'
            scen_name += f'_{crmethod}'
            scen_name += f'.scn'
            
            # Create the lines
            lines = '00:00:00>SEED 42\n00:00:00>ASAS ON\n'
            lines += f'00:00:00>RESO {crmethod}\n'
            lines += f'00:00:00>CDMETHOD {cdmethod}\n'
            lines += '00:00:00>IMPL WINDSIM M22WIND\n'
            lines += '00:00:00>SETM22WIND 0 0\n'
            lines += f'00:00:00>SETM22DELAY {delay_mag} {delay_prob}\n'
            lines += f'00:00:00>ENABLESPAWNPROTECTION\n'
            lines += '00:00:00>STARTLOGS\n'
            lines += f'00:00:00>PCALL {base_dir}/{base_scen}\n'
            lines += '00:00:00>SCHEDULE 02:00:00 DELETEALL\n'
            lines += '00:00:00>SCHEDULE 02:00:01 HOLD'
            
            # Open the new file
            with open(f'ENAC/{scen_name}', 'w') as f:
                f.write(lines)
                
            # Write the batch lines as well
            batch_lines += f'00:00:00.00>SCEN {scen_name.replace('.scn','')}\n'
            batch_lines += f'00:00:00.00>PCALL ENAC/{scen_name}\n'
            batch_lines += f'00:00:00.00>FF\n\n'
# Create the final batch file
with open('enacbatch.scn', 'w') as f:
    f.write(batch_lines)