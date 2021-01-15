# Instruction
Below is the instruction to replicate the training evaluation, and test processes.
*******************************Installation*******************************
1) please use requirements.txt file with below command. recommend to create a virenv before installation.

step 1: conda create -n tasks python==3.7.3
step 2: conda install jupyter==1.0.0
step 2: conda install spyder==4.0.1
step 3: pip install -r requirement.txt

Note spyder and jupyter are included. But feel free to remove them if you plan not to create a blank virenv.

*****************************Model training*******************************
1) If you want to use the provided trainset for retraining, please use file trainSet_enriched.csv.
2) change cwd to the folder raw data is saved.

python scripts/train_pipeline.py outputs/data/trainSet_enriched.csv

*****************************Model Evaluation*****************************
1) If you want to use the provided evalset for evaluation, please use file trainSet_eval.csv.
2) change cwd to the folder raw data is saved.

python scripts/evaluation.py outputs/data/trainSet_eval.csv

*****************************Model Prediction*****************************
1) change cwd to the folder raw data is saved.

python scripts/prediction.py candidateTestSet.txt

****************************Model Hyperparameter***************************
If in need of changing some configuration of the model, visit config.py file. 
Please bear in mind that not all hyperparameters are exposed in config.py.

**************************Data storage*************************
The raw data files are assumed and recommended to be saved in one level above the folder of the scripts. 

