import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

class RF_LGB_Classifier:
    def __init__(self, n_estimators=100, n_trials=100, random_state=2024, use_optuna=False):
        self.n_estimators = n_estimators
        self.n_trials = n_trials
        self.random_state = random_state
        self.use_optuna = use_optuna
        self.models = []

    def objective(self, trial, train_x, val_x, train_y, val_y, sub_params):
        
        param = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "num_leaves": trial.suggest_int("num_leaves", 2 ** 3, 2 ** 6),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1000.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1000.0, log=True),
        }
        param.update(sub_params)

        model = lgb.LGBMClassifier(**param)
        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(100,verbose=False)], eval_metric='accuracy')
        y_pred_train = model.predict(train_x)
        y_pred_val = model.predict(val_x)
        train_acc = accuracy_score(train_y, y_pred_train)
        val_acc = accuracy_score(val_y, y_pred_val)
        balance_acc = train_acc * 0.3 + val_acc * 0.7

        return balance_acc

    def base_estimator(self, x, y):
        if self.use_optuna == True:
            sub_params = {
                "boosting_type": "gbdt",
                "objective": "binary",
                "n_jobs": -1,
                "verbose": -1,
                "colsample_bytree": np.random.uniform(0.4, 1.0),
                "subsample": np.random.uniform(0.4, 1.0)
            }
            train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2)
            study = optuna.create_study(direction='maximize', sampler=TPESampler())
            study.optimize(lambda trial: self.objective(trial, train_x, val_x, train_y, val_y, sub_params), n_trials=self.n_trials)
            print("最佳参数:", study.best_params)
            sub_params.update(study.best_params)
        else:
            sub_params = {
                "boosting_type": "gbdt",
                "objective": "binary",
                "n_jobs": -1,
                "verbose": -1,
                "colsample_bytree": np.random.uniform(0.4, 1.0),
                "subsample": np.random.uniform(0.4, 1.0),
                "learning_rate": np.random.uniform(0.01, 0.3),
                "n_estimators": np.random.randint(50, 500),
                "max_depth": np.random.randint(2, 10),
                "num_leaves": np.random.randint(2 ** 3, 2 ** 6),
                "reg_lambda": np.random.uniform(1e-8, 1000.0),
                "reg_alpha": np.random.uniform(1e-8, 1000.0)
            }
        train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2)
        model = lgb.LGBMClassifier(**sub_params)
        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], callbacks=[lgb.early_stopping(100,verbose=False)], eval_metric='accuracy')
        return model
    
    def fit(self, x, y):
        for i in range(self.n_estimators):
            print(f"正在训练子树: {i+1}/{self.n_estimators}")
            sub_idx = np.random.choice(x.index, len(x), replace=True)
            sub_x, sub_y = x.loc[sub_idx], y.loc[sub_idx]
            model = self.base_estimator(sub_x, sub_y)
            self.models.append(model)

    def predict_proba(self, x):
        preds = np.zeros((len(x), self.n_estimators))
        for i, model in enumerate(self.models):
            preds[:, i] = model.predict_proba(x)[:, 1]
        return np.mean(preds, axis=1)
    
    def find_optimal_threshold(self, y_true, y_probs):
        # 计算 ROC AUC 分数
        thresholds = np.arange(0.0, 1.0, 0.0001)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            preds = (y_probs > threshold).astype(int)
            score = roc_auc_score(y_true, preds)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold
    
    def predict(self, x, y_true=None):
        preds = np.zeros((len(x), self.n_estimators))
        for i, model in enumerate(self.models):
            preds[:, i] = model.predict(x)
            
        preds = np.mean(preds, axis=1)

        # 如果提供了真实标签，则寻找最佳切分点
        if y_true is not None:
            optimal_threshold = self.find_optimal_threshold(y_true, preds)
            return (preds > optimal_threshold).astype(int)

        return (preds > 0.5).astype(int)  # 默认使用0.5作为切分点