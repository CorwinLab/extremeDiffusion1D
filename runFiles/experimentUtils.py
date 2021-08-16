import json
import glob

def saveVars(vars, save_file):
    """
    Save experiment variables to a file along with date it was ran and
    """
    with open(save_file, "w+") as file:
        json.dump(vars, file)

def condenseSlurmLogs(directory, delete_output=True):
    """
    Slurm usually creates a ton of log files. This functions should condense down
    the logs to a single file and delete any empty files.
    """

    error_files = glob.glob("*.err")
    output_files = glob.glob("*.out")

    if delete_output:
        print("Deleting Output Files")
        for f in output_files:
            os.remove(f)

    for f in error_files:
        if os.stat(f).st_size == 0:
            os.remove(f)
