Requirements:
    
    Python 3.7.3 (v3.7.3:ef4ec6ed12, Mar.25 2019, 16:52:21)
    numpy 1.16.4
    scipy
    
Run:
    
    1, First you should copy the feature file into one specified directory, 
    and we input the directory as a parameter in function run_duti(...).
    
    2, function run_duti(...) is an example to run the algorithm, 
    including load the .txt file, init data for duti, run duti and 
    save the result. You can just change the path in the __main__ script 
    of the Main.py.
    
Result:

    1, I have test the algorithm on the two given dataset 2522 and 2606, 
    it can quickly finished the correction on the 2522 dataset, but it seems 
    blocked when correcting the 2522.
    
    2, I have done a simple check, it cost too much time to train a LR classifier. 
    It worked well if we only use the top 10000 items of the training data. So, as 
    we expected, we still face the speed problem.
    