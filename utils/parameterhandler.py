"""
    parameterhandler creates combinations of parameters and handles them
"""

import json
import numpy as np
import importlib
import os
import pandas as pd
from fc_util.config_util import read_config


class parameter_combination:
    def __init__(self, param_settings:dict = {}, num = 0): #, base_path = None, create_from_base = False):
        """
        :param param_settings:      dict, every key corresponds to a parameter name.
                                    The value is either the value of that parameter that should be chosen every time or
                                    a list where the first entry is a function that returns a value or list of values and the second entry
                                    is a dict with the arguments for the function. If the value is a list or the function returns a list of
                                    values they are used in sequence and looped if necessary
        :param num:                 number of setups that are being created in addition to the potential base one
        :param base_path:           path to a json file specifing the base param_settings
        :param create_from_base:    Wether to create a reference setup from the base settings in addition to the num others
        """
        self.settings = {}
        self.parameters = pd.DataFrame()
        # self.base_path = base_path
        # self.best_params = None
        # self.best_loss = np.inf

        # if self.base_path is not None:
        #     if os.path.isfile(self.base_path):
        #         base = self.read_json(self.base_path)
        #         if 'best_loss' in base and base['best_loss'] is not None:
        #             self.best_loss = base['best_loss']
        #         if 'settings' in base and base['settings'] is not None:
        #             self.settings = base['settings']
        #             self.best_params = base['settings']
        #         if create_from_base:
        #             self.generate(1)
        #             num -= 1
        #     else:
        #         print(self.base_path, ' does not exist, no basic configuration was loaded.')
        #for key in param_settings.keys():
        self.settings.update(param_settings)
        
        self.generate(num)
    
    def add_parameters(self, param_settings:dict):
        """
        :param param_settings: keys and values are added to the settings of the parameter handler
        """
        self.settings.update(param_settings)
        # self.fill()

    def clear_values(self):
        """
        clears all values from the parameterhandler but keeps the settings untouched
        """
        self.parameters = pd.DataFrame()

    def reset(self):
        """
        clears all settings and values
        """
        self.clear_values()
        self.settings = {}

    def remove(self, key):
        """
        removes all values of the specified key(s) from the parameter combinations and from the settings
        """
        if not isinstance(key, list):
            key=[key]
        for k in key:
            try:
                self.settings.pop(k) #= {k:self.settings[k] for k in self.settings.keys() if k not in key}
            except KeyError:
                print(k, ' was not recognized and thus not removed')
        self.parameters.drop(columns=key,inplace=True)

    # def fill(self):
    #     '''fills the not generated parameters up to the needed length'''
    #     for idx in range(len(self)):
    #         keys = [k for k in self.settings.keys() if k not in self.parameters[idx].keys()]
    #         # for key in self.settings.keys():
    #         #     if key not in self.parameters[idx].keys():
    #         for key in keys:
    #             # generator should be a callable that can be called with the arguments in self.settings[key][1]
    #             # or a set value that can should be chosen.
    #             # TODO implement method to select a random entry from a list.
    #             if isinstance(self.settings[key], dict) and 'function' in self.settings[key].keys():
    #                 generator = self.settings[key]['function']
    #                 if hasattr(generator, '__call__'):
    #                     pass
    #                 elif isinstance(generator, str):
    #                     parse = generator.rsplit('.',1)
    #                     mod = importlib.__import__(parse[0],fromlist=parse[1])
    #                     generator = getattr(mod, parse[1])
    #                 else:
    #                     raise AttributeError('a function was specified but could not be parsed')
    #                 if 'kwargs' in self.settings[key].keys():
    #                     self.parameters[idx][key] = generator(**(self.settings[key]['kwargs']))
    #                 else:
    #                     self.parameters[idx][key] = generator()
    #             else:
    #                 self.parameters[idx][key] = self.settings[key]

    def generate(self, num=1):
        '''generates num new values for the parameters in self.settings'''
        if num < 1:
            return

        tmp_df = pd.DataFrame()
        for key in self.settings.keys():
            parameters = []
            while(len(parameters) < num):
                if isinstance(self.settings[key], dict) and 'function' in self.settings[key].keys():
                    # parse function and arguments
                    generator = self.settings[key]['function']
                    if hasattr(generator, '__call__'):
                        pass
                    elif isinstance(generator, str):
                        parse = generator.rsplit('.',1)
                        mod = importlib.__import__(parse[0],fromlist=parse[1])
                        generator = getattr(mod, parse[1])
                    else:
                        raise AttributeError('a function was specified but could not be parsed')

                    if 'kwargs' in self.settings[key].keys():
                        single_call_params = generator(**(self.settings[key]['kwargs']))
                    else:
                        single_call_params = generator()
                else:
                    single_call_params = self.settings[key]
                
                if isinstance(single_call_params,list):
                    parameters += single_call_params
                else:
                    parameters += [single_call_params]

            tmp_df[key] = parameters[:num]

        self.parameters = pd.concat([self.parameters,tmp_df], axis=0, ignore_index=True)

    # def write_new_base(self, best_loss = None, params:dict = None, path = None):
    #     """
    #     either updates the base settings or writes a new one, can not write functions
    #     :param params:  dictonary that will be saved.
    #     :param path:    path to a file where the base configuration is saved.
    #     """
    #     if best_loss is None:
    #         best_loss = self.best_loss
    #     if params is None:
    #         params = self.best_params
    #     config = dict(settings = params, best_loss=best_loss)
    #     if path is None:
    #         path = self.base_path
    #     with open(path,'w') as output: 
    #         json.dump(config, output, indent = 4)

    def read_json(self, path):
        """
        reads settings from json
        """
        with open(path,'r') as input:
            base = json.load(input) 
            return base

    def __len__(self):
        """
        len corresponds to the number of parameter configurations generated
        """
        return self.parameters.shape[0]

    def __getitem__(self,index):
        """
        returns a parameter configuaration
        """
        return self.parameters.take([index], axis=0).dropna(axis=1).to_dict(orient='index')[index]

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if(self.iter_index < self.parameters.shape[0]):  
            ret = self[self.iter_index]
            self.iter_index += 1
            return ret
        else:
            raise StopIteration
    
    
def make_from_config(model_name, config_path, main_path):
    """
    generates a parameter configuration from a json file defining a path to the base values 'rel_base_path':str, and optional 'number_of_test': int
    and a 'setting' that defines parameter settings as defined in __init__.
    """
    with open(config_path,'r') as input: 
        config = json.load(input)

    config = read_config(model_name, config_path, main_path, suffix = 'parameterhandler')
    # base = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # base = os.path.join(base, config['rel_base_path'])
    
    if 'number_of_tests' in config.keys():
        num = config['number_of_tests']
    else:
        num = 1
        print('number_of_tests was not defined, set to 1')
    ret = parameter_combination(config['settings'], num=num)
    return ret
