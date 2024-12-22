import pandas as pd

def evaluate_model(model, X_test, y_test, metric):

    y_predict = model.predict(X_test)
    results_df = X_test
    results_df['y_real'] = y_test
    results_df['y_predict'] = y_predict
    
    metric_value = metric(y_test, y_predict)
    
    return results_df, metric_value