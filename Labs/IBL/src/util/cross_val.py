
def cross_val(ds_list, algorithm, train_method, pred_method):
    perms = 0
    train = getattr(algorithm, train_method)
    pred = getattr(algorithm, pred_method)

    for i, ds_i in enumerate(ds_list):
        # Assert data gets splitted
        x_train = ds_i['x_train']
        y_train = ds_i['y_train']
        x_pred = ds_i['x_pred']
        y_pred = ds_i['y_pred']

        train(x_train, y_train)
        for j in range(0, 10):
            perms += 1
            if i == j:
                continue

            pred(x_pred.iloc[0])

    return ds_list
