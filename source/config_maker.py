# Can be used to modify or generate config files run either with the path to the config file or the station name
# to erase the old config file before writing a new one do "python3 config_maker.py --new <stationname> "

import json
import sys
import os

main_path = os.dirname(os.path.dirname(os.path.realpath(__file__))) 
sys.path.append(main_path)

par = {}
#read configs

CONFIG_PATH = sys.argv[2] 

if(sys.argv[1] == '--mod'):
    with open(sys.argv[2],'r') as input: 
        CONFIG_PATH = sys.argv[2]
        par = json.load(input)

#when modifying a config file only select the parameters that are changed or creat new ones, the other will stay the same 
elif(sys.argv[1] == '--new'):
    print(CONFIG_PATH + ' has been reset before writing')
    par = {}
else:
    print('No option selected either use either --mod or --new (WARNING: This  erases the old config file)')

# INSERT MODIFICATIONS FOR THE CONFIG HERE
# ============================================
par["dropout_fc"] = 0.3
par["dropout_core"] = 0.2
par["core_model"] = 'gru'

with open(CONFIG_PATH,'w') as output: 
    json.dump(par, output, indent = 4)
