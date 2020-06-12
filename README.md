# Language Classifier for English and Dutch
### Introduction
The project is an easy program to train and test **Decision Trees** and **Adaboost**.  
The program already has models trained using train.dat,
however the models can be trained on different data using the train option.

### Technologies
Python3  
pickle - for serialization and deserialization  
numpy - to add the smallest positive number to the denominator

### Launch
#### Training the Models
python3 train.py training_data_file model_output_file type_of_model  
The 'type of model' is limited to dt for decision tree and ada for adaboost

##### Example  
For training a decision tree on the available train.dat we would use:  
python3 train.py train.dat output.txt dt

#### Using the Models
python3 predict.py model_file data_file  
The program automatically detects the type of model if it is generated using the train module

##### Example  
For getting predictions using adaboost you would run:  
python3 predict.py adaoutput.txt test.txt

