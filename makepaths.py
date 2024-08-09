import multiprocessing as mp
from data.parser import CityParser, parse_scenario
from scenario.pcache import PathMaker
import tqdm
mp.set_start_method('fork')

city = CityParser('Vienna')

def makepaths(scen):
    name, scenario = parse_scenario(scen)
    # Generate the cache files for this scenario
    _ = PathMaker(scenario=scenario,
        G=city.G,
        nodes=city.nodes,
        edges=city.edges,
        city = city,
        time_horizon=7200,
        time_step=1,
        fl_num=10,
        fl_size=20,
        C=6,
        time_window=60,
        v_cruise=15,
        v_turn=4.78,
        v_up=5,
        v_down=3,
        a_hoz=3,
        yaw_r=55,
        max_flight_time=1800,
        overlap=True,
        scen_name=name,
        num_cpus=1,
        seed=42,
        force_path_gen = True
        )
    
if __name__ == "__main__":
    # Start the multiprocessing
    with mp.Pool(1) as p:
        results = list(tqdm.tqdm(p.imap(makepaths, city.scenarios), 
                            total=len(city.scenarios)))