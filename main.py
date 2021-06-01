from AltWOA import AltWOA
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def correct_target(target):
  new_t = np.zeros(target.shape)
  target = np.array(target)
  unique = np.unique(target)
  num_classes = unique.shape[0]
  for i,u in enumerate(unique):
    new_t[np.where(target==u)[0]] = i
  return new_t

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, default = './', help='Directory where the csv file of features is stored')
parser.add_argument('--csv', type=str, required = True, help='Name of the csv file')
parser.add_argument('--test_size', type=int, default = 0.2, help='Size of test set(Absolute value, i.e., 0.2, 0.3, etc.)')
parser.add_argument('--num_agents', type=int, default = 20, help='Population size')
parser.add_argument('--max_iter', type=int, default = 20, help='Maximum number of iterations to run AWOA')
parser.add_argument('--altruism_indi', type=int, default = 8, help='Number of altruistic individuals')
parser.add_argument('--save_conv_graph', type=bool, default = True, help='Save the convergence plots?')
args = parser.parse_args()

root = args.data_directory
if root[-1] != '/':
    root = root+'/'
csv = args.csv
if '.csv' not in csv:
    csv=csv+'.csv'
df = np.asarray(pd.read_csv(root+csv))
data = df[:,:-1]
target = correct_target(df[:,-1])

train_data, test_data, train_label, test_label = train_test_split(data, target, test_size = args.test_size)

solution = AltWOA(num_agents=args.num_agents,
                max_iter=args.max_iter,
                train_data=train_data, train_label=train_label,
                test_data = test_data, test_label = test_label,
                altruism_indi = args.altruism_indi,
                save_conv_graph=True)
