


def train_test (args, model, data):
    
    ######################## setting data
    train_data, test_data = data[0], data[1]
    X_train_data, y_train_data = train_data.drop(['user_id', 'isbn', 'rating','book_author'], axis=1), train_data['rating']
    X_test_data, y_test_data = test_data.drop(['user_id', 'isbn', 'rating','book_author'], axis=1), test_data['rating']
    
    ######################## train
    model.fit(X_train_data, y_train_data)
    
    ######################## saving using parameters
    saving_param = model.get_params()

    f = "best_{}".format
    for param_name, param_value in saving_param:
        wandb.run.summary[f(param_name)] = param_value
        
    y_hat = catboost_cl.predict(X_test_data)

    
    return y_hat.tolist()