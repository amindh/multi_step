# Learning from mistakes: An Interpretable and Coherent Multi-step Ahead Time Series Forecasting Framework

This project contains our multi-step ahead forecasting algorithm.

###Packages

The main code is 'Dagger.py' contained in the folder 'src'. The class "Dagger"
have a method ".learn" that learn a multi-step ahead predictor by taking as 
input a training dataset of shape [T,dim,lag,num_traj]. 'src' contains two 
folders: 'predict_methods' containing the methods of our paper 
(DaD_sub,DaD_agg,_DaD_R) and DaD and a folder 'utils'.
 
### Prerequisites

Requirements to run the code:
'''python
pip install -r requirements.txt
'''


### Run an example

The 'main.py' file contains an example with random data. You can use your own
dataset to run the code.