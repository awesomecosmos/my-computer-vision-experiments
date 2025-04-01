import time
import os
import json
import sys
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
import pickle

import utils

with open("../data/results/pretrained_experiments/all_results.pkl", "rb") as f:
    data = pickle.load(f)

results_df = pd.DataFrame(data)
print(results_df.head())