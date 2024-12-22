from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import optuna
import numpy as np

class Blend:
    def __init__(self, df):
        self.lgbm = LGBMClassifier()
        self.xgboost = XGBClassifier()
        self.catboost = CatBoostClassifier()
        self.df = df
        self.tuned_models = {}

    def tune_model_with_optuna(self, model, model_name, X_train, y_train):
        def objective(trial):
            params = {}
            if model_name == 'lgbm':
                params = {
                    "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
                    "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0)
                }
                model = LGBMClassifier(**params)
            elif model_name == 'xgboost':
                params = {
                    "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 15),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                }
                model = XGBClassifier(**params)
            elif model_name == 'catboost':
                params = {
                    "iterations": trial.suggest_int("iterations", 50, 500),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                    "depth": trial.suggest_int("depth", 3, 15),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                    "border_count": trial.suggest_int("border_count", 5, 255)
                }
                model = CatBoostClassifier(**params, silent=True)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            score = f1_score(y_train, y_pred)
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial), n_trials=50)
        best_params = study.best_params

        if model_name == 'lgbm':
            model = LGBMClassifier(**best_params)
        elif model_name == 'xgboost':
            model = XGBClassifier(**best_params)
        elif model_name == 'catboost':
            model = CatBoostClassifier(**best_params, silent=True)

        model.fit(X_train, y_train)
        self.tuned_models[model_name] = model
        return model

    def fit_and_blend(self):
        X = self.df.drop('Claim', axis=1)
        y = self.df['Claim']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        lgbm_tuned = self.tune_model_with_optuna(self.lgbm, 'lgbm', X_train, y_train)
        xgboost_tuned = self.tune_model_with_optuna(self.xgboost, 'xgboost', X_train, y_train)
        catboost_tuned = self.tune_model_with_optuna(self.catboost, 'catboost', X_train, y_train)
        lgbm_preds = lgbm_tuned.predict_proba(X_test)[:, 1]
        xgboost_preds = xgboost_tuned.predict_proba(X_test)[:, 1]
        catboost_preds = catboost_tuned.predict_proba(X_test)[:, 1]
        blended_preds = (lgbm_preds + xgboost_preds + catboost_preds) / 3
        final_preds = (blended_preds >= 0.5).astype(int)
        blended_f1 = f1_score(y_test, final_preds)
        print(f"\nБлендинг complete! F1-score: {blended_f1}")
        return blended_f1, final_preds
