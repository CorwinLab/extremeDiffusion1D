import json


def saveVars(vars, save_file):
    """
    Save experiment variables to a file along with date it was ran and
    """
    if os.path.exists(save_file):
        return None

    with open(save_file, "w+") as file:
        json.dumps(vars, file)
