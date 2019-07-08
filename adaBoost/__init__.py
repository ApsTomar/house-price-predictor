import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import preprocessing

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
preprocessing.set_path(path)
