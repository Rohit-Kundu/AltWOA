import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.filter import PasiLuukka
from utils.AltWOA import AltWOA

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
parser.add_argument('--csv_path', type=str, required = True, help='Path to where the csv file of features')
parser.add_argument('--test_size', type=float, default = 0.2, help='Size of test set(Absolute value, i.e., 0.2, 0.3, etc.)')
parser.add_argument('--filter_reduction', type=int, default = 100, help='Number of features to retain using filter method.')
parser.add_argument('--num_agents', type=int, default = 100, help='Population size')
parser.add_argument('--max_iter', type=int, default = 20, help='Maximum number of iterations to run AWOA')
parser.add_argument('--altruism_indi', type=int, default = 10, help='Number of altruistic individuals')
parser.add_argument('--save_conv_graph', type=bool, default = True, help='Save the convergence plots?')
args = parser.parse_args()


df = np.asarray(pd.read_csv(args.csv_path,header=None))
data = df[:,:-1]
target = correct_target(df[:,-1])

train_data, test_data, train_label, test_label = train_test_split(data, target, test_size = args.test_size)

#Initial reduction using Pasi Luukka Filter method
sol = PasiLuukka(train_data, train_label)
pos = np.where(sol.ranks<args.filter_reduction)[0]
train_reduced = train_data[:,pos]
test_reduced = test_data[:,pos]


solution = AltWOA(num_agents=args.num_agents,
                  max_iter=args.max_iter,
                  train_data=train_reduced, train_label=train_label,
                  test_data = test_reduced, test_label = test_label,
                  altruism_indi = args.altruism_indi,
                  save_conv_graph=True)
