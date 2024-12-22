from probatus.feature_elimination import ShapRFECV 
from sklearn.linear_model import LogisticRegression

def select_features_rfecv(X, y, model=None, n_features=5):
   
    if model is None:
        model = LogisticRegression()

    shap_elimination = ShapRFECV(model, step=1, cv=5, n_jobs=-1)

    shap_elimination.fit_compute(X, y)

    selected_features = shap_elimination.get_reduced_features_set(num_features=n_features)

    return selected_features
