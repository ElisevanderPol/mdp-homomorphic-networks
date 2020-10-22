import csv
import os
import json

def load_values(folder, value_list, remove_nan=False):
    """
    Load logged values from csv
    """
    with open(os.path.join(folder, "progress.csv")) as csv_file:
        csv_reader = csv.DictReader((l.replace('\0', '') for l in csv_file), delimiter=',')
        val_dict = {key: [] for key in value_list}
        for c, row in enumerate(csv_reader):
            for key in value_list:
                if remove_nan:
                    if row[key] != "nan":
                        val_dict[key].append(float(row[key]))
                else:
                    val_dict[key].append(float(row[key]))
    return val_dict


def load_runs(folder, values, remove_nan=False):
    """
    Load run logs
    """
    runs = os.listdir(folder)

    run_dict = {run: None for run in runs}
    for run in runs:

        run_location = os.path.join(folder, run)
        val_dict = load_values(run_location, values, remove_nan=remove_nan)
        run_dict[run] = val_dict
    return run_dict


def get_params(folder):
    runs = os.listdir(folder)
    for run in runs:
        with open(os.path.join(folder, run, "params.json")) as file:
            opened = file.read()
            data = json.loads(opened)
            return data



class Folder:
    """
    """
    def __init__(self, top_folders, name):
        """
        """
        self.top_folders = self.search(top_folders, name)
        self.name = name
        param_loc = os.path.join(self.top_folders[0], name)
        self.param_dict = get_params(param_loc)

    def search(self, top_folders, name):
        """
        """
        tops = []
        for folder in top_folders:
            full_folder = os.path.join(folder, name)
            if os.path.exists(full_folder):
                tops.append(folder)
        return tops

    def get_param_value(self, key):
        """"
        """
        return self.param_dict[key]

    def has_param_value(self, key, value):
        """
        """
        return self.param_dict[key] == value

    def get_runs(self, vals, remove_nan=False):
        """
        get runs for stats for each run over multiple top folders for the same
        experiment
        """
        all_run_dict = {}
        for tf in self.top_folders:
            run_dict = load_runs(os.path.join(tf, self.name), vals,
                                 remove_nan=remove_nan)
            for key in run_dict.keys():
                all_run_dict[key] = run_dict[key]
        tot_dict = {}
        for key in all_run_dict.keys():
            run_dict = all_run_dict[key]
            for k in run_dict.keys():
                try:
                    tot_dict[k].append(run_dict[k])
                except KeyError:
                    tot_dict[k] = [run_dict[k]]
        return tot_dict



def get_run_collector(split_on, top_folders, folders, values):
    """
    Collect the relevant run logs
    """
    run_collector = {key: {} for key in values}
    for folder in folders:
        fldr = Folder(top_folders, folder)
        splitter_key = ""
        for so in split_on:
            splitter_key += str(so) + "_" + str(fldr.get_param_value(so)) +";"
        tot_dict = fldr.get_runs(values, remove_nan=True)

        for key in tot_dict.keys():
            try:
                run_collector[key][splitter_key] += tot_dict[key]
            except KeyError:
                run_collector[key][splitter_key] = tot_dict[key]
    return run_collector
