import os
import json

class ParamLogging:

    def __init__(self):
        self.log_list = []

    def emit(self, log):
        self.log_list.append(log)

    def save(self, save_path):
        with open(save_path, "w") as f:
            json.dump(self.log_list, f, indent=4)
