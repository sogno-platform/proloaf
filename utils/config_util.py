import os
import json
import sys
import argparse
import plf_util.eval_metrics as metrics

#TODO test if configmaker is still working after refactoring the directories

def read_config(model_name = None, config_path = None, main_path=''):
    if config_path is None:
        config_path = os.path.join(main_path, 'targets',  model_name, 'config.json')
    with open(config_path,'r') as input:
        return json.load(input)

def write_config(config, model_name = None, config_path = None, main_path=''):
    if config_path is None:
        config_path = os.path.join(main_path, 'targets',  model_name, 'config.json')
    with open(config_path,'w') as output:
        return json.dump(config, output, indent=4)

class flag_and_store(argparse._StoreAction):
    def __init__(self, option_strings, dest, dest_const, const, nargs = 0, **kwargs):
        self.dest_const = dest_const
        if isinstance(const, list):
            self.val = const[1:]
            self.flag = const[0]
        else:
            self.val = []
            self.flag = const
        if isinstance(nargs, int):
            nargs_store = nargs+len(self.val)
        else:
            nargs_store = nargs
        super(flag_and_store, self).__init__(option_strings, dest, const = None, nargs = nargs_store, **kwargs)
        self.nargs = nargs

    def __call__(self, parser, namespace, values, option_strings = None):
        setattr(namespace, self.dest_const,self.flag)
        if isinstance(values,list):
            self.val.extend(values)
        elif values is not None:
            self.val.append(values)

        if len(self.val) == 1:
            self.val = self.val[0]
        super().__call__(parser, namespace, self.val, option_strings)


def parse_basic(args = sys.argv[1:]):
    parser = argparse.ArgumentParser()
    # TODO this should be a requiered argument (also remove default below)
    ident = parser.add_mutually_exclusive_group()#required=True)
    ident.add_argument("-s","--station", help = "station to be trained for (e.g. gefcom2017/nh_data)", default='gefcom2017/nh_data')
    ident.add_argument("-c", "--config", help = "path to the config file relative to the project root")
    return parser.parse_args(args)

def parse_with_loss(args = sys.argv[1:]):
    parser = argparse.ArgumentParser()
    # TODO this should be a requiered argument (also remove default below)
    ident = parser.add_mutually_exclusive_group()#required=True)
    ident.add_argument("-s","--station", help = "station to be trained for (e.g. gefcom2017/nh_data)", default='gefcom2017/nh_data')
    ident.add_argument("-c", "--config", help = "path to the config file relative to the project root")

    losses = parser.add_mutually_exclusive_group()
    losses.add_argument("--ci", help="Enables execution mode optimized for GitLab's CI", action='store_true', default=False)
    #losses.add_argument("--hyper", help="turn hyperparam-tuning on/off next time (int: 1=on, else=off)", type=int, default=0)
    #losses.add_argument("-o", "--overwrite", help = "overwrite config with new training parameter (int: 1=True=default/else=False)", type=int, default=0)
    losses.add_argument("--nll_gauss", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.nll_gauss,2], help = "train with nll guassian loss", action=flag_and_store)
    losses.add_argument("--quantiles", dest_const='loss', dest='quantiles',metavar=' q1 q2', nargs='+', type=float, const=metrics.quantile_score, help = "train with pinball loss and MSE with q1 and q2 being the upper and lower quantiles", action=flag_and_store)
    losses.add_argument("--crps","--crps_gaussian", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.crps_gaussian,2], help = "train with crps gaussian loss", action=flag_and_store)
    losses.add_argument("--mse", "--mean_squared_error", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.mse,1], help = "train with mean squared error", action=flag_and_store)
    losses.add_argument("--rmse", "--root_mean_squared_error", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.rmse,1], help = "train with root mean squared error", action=flag_and_store)
    losses.add_argument("--mape", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.mape,1], help = "train with root mean absolute error", action=flag_and_store)
    # losses.add_argument("--mase", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.mase,2], help = "train with root mean absolute scaled error", action=flag_and_store)
    # losses.add_argument("--picp", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.picp_loss,2], help = "train with 1 - prediction intervall coverage",action=flag_and_store)
    losses.add_argument("--sharpness", dest_const='loss', dest='num_pred', nargs=0, type=int, const=[metrics.sharpness,1], help = "train with sharpness", action=flag_and_store)
    losses.add_argument("--mis", "--mean_interval_score", dest_const='loss', dest='alpha', nargs=1,metavar = 'A', type=float, const=metrics.mis, help = "train with mean intervall score, A corresponding to the inverse weight", action=flag_and_store)
    #TODO write mis as default argument as soon as evaluate works with it
    parser.set_defaults(loss=metrics.nll_gauss, num_pred=2)
    ret = parser.parse_args(args)

    if ret.loss == metrics.mis:
        ret.num_pred = 2
        options = dict(alpha=ret.alpha)
    if ret.loss == metrics.quantile_score:
        ret.num_pred = len(ret.quantiles) + 1
        options = dict(quantiles = ret.quantiles)
    else:
        options = {}
    return ret, options

def query_true_false(question, default="yes"):
    ## Source: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": False, "y": False, "ye": False,
             "no": True, "n": True}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
