# Altruistic Whale Optimization Algorithm (AltWOA)
AltWOA is a novel feature selection algorithm where Altruistic properties of candidate whales are embedded to the Whale Optimization Algorithm (WOA). 

## Dependencies 
    pip install -r requirements.txt
## Arguments
    PS E:\Ramsir\Altruism\code\AltWOA> python main.py -h
    usage: main.py [-h] --csv_path CSV_PATH [--test_size TEST_SIZE] [--filter_reduction FILTER_REDUCTION] [--num_agents NUM_AGENTS] [--max_iter MAX_ITER]
                   [--altruism_indi ALTRUISM_INDI] [--save_conv_graph SAVE_CONV_GRAPH]

    optional arguments:
      -h, --help            show this help message and exit
      --csv_path CSV_PATH   Path to where the csv file of features
      --test_size TEST_SIZE
                            Size of test set(Absolute value, i.e., 0.2, 0.3, etc.)
      --filter_reduction FILTER_REDUCTION
                            Number of features to retain using filter method.
      --num_agents NUM_AGENTS
                            Population size
      --max_iter MAX_ITER   Maximum number of iterations to run AWOA
      --altruism_indi ALTRUISM_INDI
                            Number of altruistic individuals
      --save_conv_graph SAVE_CONV_GRAPH
                            Save the convergence plots?
## Code Execution
    python main.py --csv_path "data/alon.csv"
