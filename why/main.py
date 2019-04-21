from quick_model import quick_model

# instantiate class
model = quick_model(test_size=.2)
parameters = {
            'application': 'binary',
            'objective': 'binary',
            'metric': 'auc',
            'is_unbalance': 'true',
            'boosting': 'gbdt',
            'num_leaves': 2,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 20,
            'learning_rate': 0.05,
            'max_depth': 2,
            'verbose': 0,
            'verbosity': 0,
            'min_data_in_leaf': 20
        }
num_leaves_valid_values = [2, 4, 6, 8]
max_depth_valid_values = [2, 3, 4, 5]
min_data_in_leaf_valid_values = [3, 5, 8, 13, 21]

for min_data_in_leaf_value in min_data_in_leaf_valid_values:
    for max_depth_value in max_depth_valid_values:
        for num_leaves_value in num_leaves_valid_values:
            parameters["num_leaves"] = num_leaves_value
            parameters["max_depth"] = max_depth_value
            parameters["min_data_in_leaf"] = min_data_in_leaf_value
            # run model (set save_graph to false if you don't have dot)
            _ = model.run_model(num_boost_round=5000, early_stopping_rounds=100, parameters=parameters, save_graph=True)
