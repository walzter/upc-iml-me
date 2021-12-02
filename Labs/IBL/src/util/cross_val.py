
def cross_val(ds_list, algorithm, pred_method, train_method):
    perms = 0
    train = getattr(algorithm, train_method)
    pred = getattr(algorithm, pred_method)

    for i, ds_i in enumerate(ds_list):
        # Assert data gets splitted
        x_train = ds_i['x_train']
        y_train = ds_i['y_train']
        x_test = ds_i['x_test']
        y_test = ds_i['y_test']
        train(x_train, y_train)
        for j in range(0, 10):
            perms += 1
            if i == j:
                continue

            print(pred(x_test))

    return ds_list
