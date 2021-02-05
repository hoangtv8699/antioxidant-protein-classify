## install library
pip install -r requirement.txt

## data
* folder data/csv for train data encode PSSM
* folder data/test for test data with independent_1 and independant_2 encode PSSM
* other is data in format csv and fasta

## python file
* train_pssm.py for training model
* helpers.py for help function like plot, read_data, padding, sensitivity, specitivity, mcc, auc ...
* test.py for testing model with test data
* other file no need for training
