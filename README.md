# Triad-based Neural Network for Coreference Resolution

This code is the implementation of our paper "Triad-based Neural Network for Coreference Resolution" https://arxiv.org/abs/1809.06491
It is a modified version of our COLING 2018 paper http://www.aclweb.org/anthology/C18-1004
In the new paper, we corrected some errors and added more analyses.

## How to install
    1. Clone our repository.
    2. Download data from CoNLL 2012 Shared Task webpage http://conll.cemantix.org/2012/data.html
       You may need license from OntoNotes to obtain all data files. The license should be free for academic uses.
       (Contact me if you have any problem downloading the data files.)
    3. For evaluation, you can use the official scorer. https://github.com/conll/reference-coreference-scorers
       By default, the scorer should be installed in the repository home directory (the same directory as train_triad.py).
       
We developed the system with Python 3.6. You also need pytorch. We used 0.4.0. 
You could also use Keras with TensorFlow, but we did not test the final performance on Keras. 
We used Keras 2.1.5 and TensorFlow 1.7.0  
       
## How to use
To train the model, run the following script:
    
    $python train_triad.py training_dir/ model_destination/ --val_dir val_dir/
    
Option val_dir specifies validation data folder. The program will automatically find *.gold_conll files in all the subfolders of training folder.
By default, it will run 400 "epochs". However, here *an "epoch" means 50 training files*, not really the whole training set.
During training, a very small subset of validation set will be used to do quick validation. This is just to make sure things go well.
After every 10 epochs, a larger subset of validation files will be used to do validation (but still not the whole set).

If you need to run evaluation on the whole validation set, use the predict.py script, to be explained in below.
    
**GPU is highly recommended.** It may take a few hours to run 400 epochs with GPU. 
    
To predict and evaluate run:
    
    $python predict.py model_destination/ test_folder/ results/ --triad --max_distance 60
    
In order to do evaluation, you need to combine all the truth file in one file "key.tmp" and save it under "results/".
You just need to append all the test files one after another, including the lines starting with "#begin" and "#end" (do not delete them!).
The program will automatically create a combined result file "response.tmp" at the end, and run the evaluation afterwards.
    
With GPU, the prediction and evaluation may take 15~30 minutes. Without GPU it may take longer.
    
If you need to modify paths for the evaluation script or key/response files, change them in the score() function in predict.py
    
    
    
    
    
    

