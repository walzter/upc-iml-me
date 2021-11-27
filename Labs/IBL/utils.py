import os 

def get_working_files(NAME):
    '''
    Input --> Name of the folder to use 
    Output --> list of training and testing file names 
    '''
    training_list = []
    testing_list = []
    for file in os.listdir(f"./data/{NAME}"):
        if 'train' in file: 
            training_list.append(file)
        else: 
            testing_list.append(file)
    assert len(testing_list) == len(training_list)
    print('Training == Testing')
    return sorted(training_list), sorted(testing_list)