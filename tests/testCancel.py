import sys
import os

src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.append(src_path)

from pydiffusionPDF import DiffusionPDF


def runExperiment(save_dir, id):
    d = DiffusionPDF(int(1e7), 1, int(1e7))
    d.id = id
    d.save_dir = save_dir
    d.evolveToTime(int(1e7))
    d._save_interval = 1


if __name__ == "__main__":
    (topDir, sysID) = sys.argv[1:]
    runExperiment(topDir, sysID)
