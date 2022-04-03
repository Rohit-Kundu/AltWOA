# Altruistic Whale Optimization Algorithm (AltWOA)
AltWOA is a novel feature selection algorithm where `Altruistic` property of candidate whales are embedded to the Whale Optimization Algorithm (WOA)

**_Note_**:  
- Refer to our paper published in _Computers in Biology and Medicine, Elsevier_: [AltWOA: Altruistic Whale Optimization Algorithm for feature selection on microarray datasets](https://doi.org/10.1016/j.compbiomed.2022.105349)
- Access the preprint [here](https://raw.githubusercontent.com/Rohit-Kundu/AltWOA/main/preprint/AltWOA.pdf).
- For the PDF of the accepted version please email the first author at rohitkunduju@gmail.com or request the full text in [ResearchGate](https://www.researchgate.net/publication/359162107_AltWOA_Altruistic_Whale_Optimization_Algorithm_for_feature_selection_on_microarray_datasets). 

- The overall flowchart of **AltWOA** and the Pseudo-algorithm of the algorithm of **Altruism** are given as follow :  ![AltWOA](https://github.com/Rohit-Kundu/AltWOA/blob/main/Pictures/Slide1.JPG) where Equation 13 and Equation 14 are
    -  ![Eq13](https://github.com/Rohit-Kundu/AltWOA/blob/main/Pictures/Screenshot%20(61)%20eq13.png) ![Eq14](https://github.com/Rohit-Kundu/AltWOA/blob/main/Pictures/Screenshot%20(62)%20eq14.png) 

## Dependencies 
    directory\AltWOA> pip install -r requirements.txt
## Arguments
    directory\AltWOA> python main.py -h
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
    directory\AltWOA> python main.py --csv_path "data/name_of_file.csv"
## Description
    directory\AltWOA
            |
            +--> utils
            |      |
            |      +--> AltWOA.py           # The python file of the proposed feature selection algorithm
            |      +--> altruism.py         # The python file for performing altruism operation
            |      +--> filter.py           # The python file for performing Pasi-Luukka operation
            |
            +--> main.py                    # The main function. Where initial feature selection is performed 
                                              via the Pasi-Luukka filter. After that final feature selection is
                                              performed via proposed AltWOA algorithm. 

# Citation
If you find this repository useful in any way, please consider citing our work:
```
@article{kundu2022altwoa,
  title={AltWOA: Altruistic Whale Optimization Algorithm for feature selection on microarray datasets},
  author={Kundu, Rohit and Chattopadhyay, Soham and Cuevas, Erik and Sarkar, Ram},
  journal={Computers in Biology and Medicine},
  pages={105349},
  year={2022},
  publisher={Elsevier}
}
```
