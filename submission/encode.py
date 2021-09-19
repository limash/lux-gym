import pickle
import bz2
import base64

with open('city_tiles.pickle', 'rb') as file:
    cts_data = pickle.load(file)
with open('units.pickle', 'rb') as file:
    units_data = pickle.load(file)
    
cts_string = base64.b64encode(bz2.compress(pickle.dumps(cts_data)))
units_string = base64.b64encode(bz2.compress(pickle.dumps(units_data)))

with open("params_city_tiles.py", "w") as text_file:
    print(f"PARAM_CT = {cts_string}", file=text_file)

with open("params_units.py", "w") as text_file:
    print(f"PARAM_UNITS = {units_string}", file=text_file)
