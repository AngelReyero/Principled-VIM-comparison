from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import r2_score, accuracy_score
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold

from hidimstat import PFICV
from sklearn.linear_model import ( LinearRegression, Ridge, Lasso, LogisticRegression) 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    AdaBoostRegressor, BaggingRegressor, StackingRegressor
)
from sklearn.neural_network import MLPRegressor 
from sklearn.svm import SVR 
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import KFold 
from sklearn.base import clone 
from utils import generate_dataset
from tabicl import TabICLClassifier, TabICLRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, StackingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--setting", type=str)
    parser.add_argument("--corr", type=float)
    return parser.parse_args()


def main(args):

    n_samples_list = [100, 250, 500, 1000, 2000, 5000]
    #cor_list = [0.0, 0.3, 0.6, 0.9]
    corr = args.corr
    results = []
    setting = args.setting
    for seed in args.seeds:
         # base models
        base_models = [
            ("lr", LinearRegression()),
            ("lasso", Lasso()),
            ("dt", DecisionTreeRegressor(random_state=seed)),
            ("rf", RandomForestRegressor(random_state=seed)),
            ("et", ExtraTreesRegressor(random_state=seed)),
            ("gb", GradientBoostingRegressor(random_state=seed)),
            ("hgb", HistGradientBoostingRegressor(random_state=seed)),
            ("ab", AdaBoostRegressor(random_state=seed)),
            ("bag", BaggingRegressor(random_state=seed)),
            ("mlp", MLPRegressor(random_state=seed, max_iter=1000)),
            ("svr", SVR()),
            ("knn", KNeighborsRegressor()),
        ]

        models = {
            **dict(base_models),

            # Super Learner (Stacking ensemble)
            "SuperLearner": StackingRegressor(
                estimators=base_models,
                final_estimator=LinearRegression(),
                passthrough=False
            ),

            # TabPFN (deep prior model for tabular data)
            #"TabICL": TabICLRegressor()
        }
        if setting not in {"linear_sparse", "interaction_sparse"}:
            models["TabICL"] = TabICLRegressor()



        base_models_clf = [
            ("lr", LogisticRegression(max_iter=1000)),
            ("dt", DecisionTreeClassifier(random_state=seed)),
            ("rf", RandomForestClassifier(random_state=seed)),
            ("et", ExtraTreesClassifier(random_state=seed)),
            ("gb", GradientBoostingClassifier(random_state=seed)),
            ("hgb", HistGradientBoostingClassifier(random_state=seed)),
            ("ab", AdaBoostClassifier(random_state=seed)),
            ("bag", BaggingClassifier(random_state=seed)),
            ("mlp", MLPClassifier(max_iter=1000, random_state=seed)),
            ("svm", SVC(probability=True)),
            ("knn", KNeighborsClassifier()),
        ]

        models_clf = {
            **dict(base_models_clf),

            # Super Learner (Stacking ensemble)
            "SuperLearner": StackingClassifier(
                estimators=base_models_clf,
                final_estimator=LogisticRegression(max_iter=1000),
                passthrough=False
            ),

            # TabPFN classifier
            "TabICL": TabICLClassifier()
        }

        print(f"Seed {seed}")

        for n in n_samples_list:

            print(f"cor={corr}, n={n}")

            X, y, true_imp, task = generate_dataset(
                setting, n, cor=corr, seed=seed
            )
            if task == "classification":
                models = models_clf
                loss = log_loss
                metric = accuracy_score
            else:
                metric = r2_score
                loss = mean_squared_error
            
            n_folds = 5

            cv = KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=seed
            )

            for model_name, model in models.items():

                print(model_name)

                regressor_list = []
                scores = []

                for train_idx, test_idx in cv.split(X):

                    est = clone(model)

                    est.fit(
                        X[train_idx],
                        y[train_idx]
                    )

                    regressor_list.append(est)

                    scores.append(
                        metric(
                            y[test_idx],
                            est.predict(X[test_idx])
                        )
                    )

                r2_mean = np.mean(scores)
                r2_std = np.std(scores)

                pfi_cv = PFICV(
                    estimators=regressor_list,
                    cv=cv,
                    n_jobs=5,
                    loss=loss,
                    random_state=seed,
                )

                pfi_res = pfi_cv.fit_importance(X, y)

                imp_mean = (
                    pfi_res.mean(axis=1)
                )

                imp_std = (
                    pfi_res.std(axis=1)
                )

                row = {
                    "seed": seed,
                    "n_samples": n,
                    "cor": corr,
                    "model": model_name,
                    "r2_mean": r2_mean,
                    "r2_std": r2_std,
                }

                for j in range(len(imp_mean)):

                    row[f"imp_V{j}"] = imp_mean[j]
                    row[f"imp_std_V{j}"] = imp_std[j]

                    if true_imp is not None:
                        row[f"tr_V{j}"] = true_imp[j]

                results.append(row)

    results = pd.DataFrame(results)

    results.to_csv(
        f"../results/csv/asymp_relevance_setting{setting}_corr{corr}_seed{seed}.csv",
        index=False,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)

