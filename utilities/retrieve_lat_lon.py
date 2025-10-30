import os
from utils import exp_data_dir

def retrieve_lat_lon():

    with open(os.path.join(exp_data_dir, 'lat_lon_boundaries.txt'), 'r') as f:
        lines = f.readlines()
        init_lat = lines[0].strip()
        final_lat = lines[1].strip()
        init_lon = lines[2].strip()
        final_lon = lines[3].strip()

    return init_lat, final_lat, init_lon, final_lon

if __name__ == "__main__":
    
    ilat, flat, ilon, flon = retrieve_lat_lon()
    print(ilat, flat, ilon, flon)