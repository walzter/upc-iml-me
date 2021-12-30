import time
import tqdm
from tqdm import tqdm

def cross_val(ds_list, algorithm, train_method, pred_method):
    perms = 0
    train = getattr(algorithm, train_method)
    pred = getattr(algorithm, pred_method)

    accuracy_matrix = []
    execution_time_m = []
    for i, ds_i in enumerate(ds_list):
        # Assert data gets splitted
        x_train = ds_i['x_train']
        y_train = ds_i['y_train']
        x_pred = ds_i['x_pred']
        y_pred = ds_i['y_pred']

        accuracy_per_test = []
        execution_time = []

        print(f'Training using index: {i}')
        start = time.time()
        train(x_train, y_train)
        end = time.time()
        execution_time.append(end - start)

        print(f'  Test index: {i}: started')
        correct_predictions = 0
        start = time.time()
        for predict_index in tqdm(range(len(x_pred))):
            predicted_label = pred(x_pred.iloc[predict_index])

            if y_pred[predict_index] == predicted_label:
                correct_predictions += 1

        end = time.time()
        execution_time.append(end - start)
        accuracy_per_test.append(correct_predictions/len(x_pred)*100)
        print(f'  Test index: {i} accuracy: {correct_predictions/len(x_pred)*100}')
        print(f'  Test index: {i}: ended')

        execution_time_m.append(execution_time)
        accuracy_matrix.append(accuracy_per_test)
        print(f'Evaluations ended: {perms}')

    return accuracy_matrix, execution_time_m
