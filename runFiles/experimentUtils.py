import json
import os

def saveVars(vars, save_file):
    """
    Save experiment variables to a file along with date it was ran and
    """
    with open(save_file, "w+") as file:
        json.dumps(vars, file)
