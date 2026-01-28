import os
import json

import numpy as np
from numpy.typing import NDArray

from .specie import Species

SAVE_FREQUENCY = None # None if no intermediate save

class VariableTracker:

    def __init__(self, log_folder_path, log_file_name, enabled: bool = True):
        self.log_folder_path = log_folder_path
        self.file_name = log_file_name
        self.log_file_path = os.path.join(log_folder_path, log_file_name)
        self.tracked_variables = {}
        self.step_number = 0
        self.enabled = enabled

    def add_value_to_variable(self, key, value):
        if not self.enabled:
            return
        if key in self.tracked_variables:
            self.tracked_variables[key].append(value)
        else:
            self.tracked_variables[key] = [value]
        if key == "time" and SAVE_FREQUENCY:
            self.step_number += 1
            if self.step_number % 2*SAVE_FREQUENCY == 0 :
                self.save_tracked_variables(filename=self.file_name+"_temp_0")
            elif self.step_number % 2*SAVE_FREQUENCY == SAVE_FREQUENCY :
                self.save_tracked_variables(filename=self.file_name+"_temp_1")

    def add_value_to_variable_list(self, prefix: str, values: list[float] | NDArray[np.float64], suffix=""):
        if not self.enabled:
            return
        for i in range(len(values)):
            self.add_value_to_variable(prefix+str(i)+suffix, values[i])

    def add_all_densities_and_temperatures(self, state, species: Species, prefix=""):
        if not self.enabled:
            return
        for i, specie in enumerate(species.species):
            self.add_value_to_variable(prefix + specie.name+"_density", state[i])
        for i in range(len(state) - species.nb):
            self.add_value_to_variable(prefix + f"temperature_{i}_atom", state[species.nb + i])

    def save_tracked_variables(self, filename=None):
        """Save in json file"""
        if not self.enabled:
            return
        if filename is not None :
            log_file_path = os.path.join(self.log_folder_path, filename)
        else:
            log_file_path = self.log_file_path
        with open(log_file_path, 'w') as file:
            json.dump(self.tracked_variables, file, indent=4)

    def update_filename(self, filename):
        self.log_file_path = os.path.join(self.log_folder_path, filename)
