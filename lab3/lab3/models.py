from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from  xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import optuna
class ml_models:
    def __init__(self, df):
        self.lgbm = LGBMClassifier()
        self.xgboost = XGBClassifier()
        self.catboost = CatBoostClassifier()
        self.df = df
        
    def fit(self):
        
        X = self.df.drop('Claim', axis=1)
        y = self.df['Claim']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        self.lgbm.fit(X_train, y_train)
        self.catboost.fit(X_train, y_train, silent=True)
        self.xgboost.fit(X_train, y_train)
        return X_test, y_test
    
    def get_best_model_in_test(self):
        
        X_test, y_test = self.fit()
        
        lgbm_f1 = f1_score(y_test, self.lgbm.predict(X_test))
        catboost_f1 = f1_score(y_test, self.catboost.predict(X_test))
        xgboost_f1 = f1_score(y_test, self.xgboost.predict(X_test))
        
        metrics_and_models = [
            (lgbm_f1, self.lgbm), 
            (catboost_f1, self.catboost), 
            (xgboost_f1, self.xgboost)
        ]
        
        best_f1_score, best_model = max(metrics_and_models, key=lambda x: x[0])
  
        self.best_model = best_model
        return best_model, best_f1_score, X_test, y_test
        
        
    def tune_best_model_with_optuna(self):
          
            if not hasattr(self, "best_model"):
                raise ValueError("Сначала нужно выполнить get_best_model_in_test()!")

            X = self.df.drop('Claim', axis=1)
            y = self.df['Claim']
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3)
            
            def objective(trial):

                if isinstance(self.best_model, LGBMClassifier):
                    params = {
                        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                        "max_depth": trial.suggest_int("max_depth", 3, 15),
                        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
                        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0)
                    }
                    model = LGBMClassifier(**params)
                
                elif isinstance(self.best_model, XGBClassifier):
                    params = {
                        "eta": trial.suggest_float("eta", 1e-3, 0.3, log=True),
                        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                        "max_depth": trial.suggest_int("max_depth", 3, 15),
                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "gamma": trial.suggest_float("gamma", 0, 5),
                    }
                    model = XGBClassifier(**params)
                
                elif isinstance(self.best_model, CatBoostClassifier):
                    params = {
                        "iterations": trial.suggest_int("iterations", 50, 500),
                        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                        "depth": trial.suggest_int("depth", 3, 15),"l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                        "border_count": trial.suggest_int("border_count", 5, 255)
                    }
                    model = CatBoostClassifier(**params, silent=True)
                
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_train)
                score = f1_score(y_train, y_pred)
                
                return score
            
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=30)  
            
            best_params = study.best_params
            if isinstance(self.best_model, LGBMClassifier):
                self.best_model = LGBMClassifier(**best_params)
            elif isinstance(self.best_model, XGBClassifier):
                self.best_model = XGBClassifier(**best_params)
            elif isinstance(self.best_model, CatBoostClassifier):
                self.best_model = CatBoostClassifier(**best_params, silent=True)
            
            self.best_model.fit(X_train, y_train)
            return self.best_model, best_params
        