import os
import os.path as op
import shutil

# standard third party imports
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

# impute missing values
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder


import warnings

warnings.filterwarnings(
    "ignore",
    message="pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="pandas.Float64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.",
    category=FutureWarning,
)


# standard code-template imports
from ta_lib.core.api import (
    create_context,
    get_dataframe,
    get_feature_names_from_column_transformer,
    string_cleaning,
    get_package_path,
    display_as_tabs,
    save_pipeline,
    load_pipeline,
    initialize_environment,
    load_dataset,
    save_dataset,
    DEFAULT_ARTIFACTS_PATH,
    list_datasets,
)

import ta_lib.eda.api as eda
from xgboost import XGBRegressor
from ta_lib.regression.api import SKLStatsmodelOLS
from ta_lib.regression.api import RegressionComparison, RegressionReport
import ta_lib.reports.api as reports
from ta_lib.data_processing.api import Outlier

initialize_environment(debug=False, hide_warnings=True)

artifacts_folder = DEFAULT_ARTIFACTS_PATH

config_path = op.join("conf", "config.yml")
context = create_context(config_path)

ground_truth_for_modelling = load_dataset(
    context, "/ground_truth_for_modelling/ground_truth"
)

# Creating variables for price of client and competitor
ground_truth_for_modelling["client_unit_price"] = (
    ground_truth_for_modelling["client_dollars_value"]
    / ground_truth_for_modelling["client_units_value"]
)
ground_truth_for_modelling["competitor_unit_price"] = (
    ground_truth_for_modelling["competitor_dollars_value"]
    / ground_truth_for_modelling["competitor_units_value"]
)

ground_truth_for_modelling.drop(columns=["Unnamed: 0"], inplace=True)


numerical_features = ground_truth_for_modelling.select_dtypes(include="number")

features = X = ground_truth_for_modelling.drop(columns=["client_lbs_value"])
target = y = ground_truth_for_modelling["client_lbs_value"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


save_dataset(context, X_train, "/train/ground_truth_for_modelling/features")
save_dataset(context, y_train, "/train/ground_truth_for_modelling/target")

save_dataset(context, X_test, "/test/ground_truth_for_modelling/features")
save_dataset(context, y_test, "/test/ground_truth_for_modelling/target")

# collecting different types of columns for transformations
cat_columns = X_train.select_dtypes("object").columns
num_columns = X_train.select_dtypes("number").columns


def one_hot_encode_column(dataframe, cat_variable):
    one_hot = pd.get_dummies(dataframe[cat_variable], drop_first=True)
    # Drop column B as it is now encoded
    dataframe = dataframe.drop("theme_name", axis=1)  # Join the encoded df
    dataframe = dataframe.join(one_hot)
    return dataframe


def engineer_date_related_features(dataframe):
    temp_dataframe = dataframe.copy()
    temp_dataframe["date"] = pd.to_datetime(temp_dataframe["date"])
    temp_dataframe["year"] = temp_dataframe["date"].dt.year
    temp_dataframe["month"] = temp_dataframe.date.dt.month
    temp_dataframe["quarter"] = temp_dataframe.date.dt.quarter
    temp_dataframe.drop(columns=["date"], inplace=True)
    return temp_dataframe


train_one_hot_encoder_transformer = FunctionTransformer(
    one_hot_encode_column, kw_args={"cat_variable": "theme_name"}
)
test_one_hot_encoder_transformer = FunctionTransformer(
    one_hot_encode_column, kw_args={"cat_variable": "theme_name"}
)
date_feature_generator = FunctionTransformer(engineer_date_related_features)

feature_engineering_pipeline_train = Pipeline(
    [
        ("one_hot_encoding", train_one_hot_encoder_transformer),
        ("date_feature_generator", date_feature_generator),
    ]
)
feature_engineering_pipeline_test = Pipeline(
    [
        ("one_hot_encoding", test_one_hot_encoder_transformer),
        ("date_feature_generator", date_feature_generator),
    ]
)

X_train = feature_engineering_pipeline_train.fit_transform(X_train)
X_test = feature_engineering_pipeline_train.fit_transform(X_test)


columns_to_drop = ["chewy_searchVolume", "total_searchVolume"]
X_train.drop(columns=columns_to_drop, inplace=True)
X_test.drop(columns=columns_to_drop, inplace=True)

reg_vars = X_train.columns.tolist()
X_train = X_train[reg_vars]
X_test = X_test[reg_vars]


# Custom Transformations like these can be utilised
def _custom_data_transform(df, cols2keep=None):
    """Transformation to drop some columns in the data

    Parameters
    ----------
        df - pd.DataFrame
        cols2keep - columns to keep in the dataframe
    """
    cols2keep = cols2keep or []
    if len(cols2keep):
        return df.select_columns(cols2keep)
    else:
        return df


imp_features = [
    "client_unit_price",
    "competitor_unit_price",
    "total_post",
    "google_searchVolume",
]

reg_ppln_ols = Pipeline(
    [
        (
            "",
            FunctionTransformer(
                _custom_data_transform, kw_args={"cols2keep": reg_vars}
            ),
        ),
        ("estimator", SKLStatsmodelOLS()),
    ]
)
reg_ppln_ols.fit(X_train, y_train.values.ravel())

reg_ppln_ols["estimator"].summary()

reg_ppln = Pipeline(
    [
        (
            "",
            FunctionTransformer(
                _custom_data_transform, kw_args={"cols2keep": reg_vars}
            ),
        ),
        ("Linear Regression", SKLStatsmodelOLS()),
    ]
)

reg_linear_report = RegressionReport(
    model=reg_ppln,
    x_train=X_train,
    y_train=y_train,
    x_test=X_test,
    y_test=y_test,
    refit=True,
)
reg_linear_report.get_report(
    include_shap=False, file_path="regression_linear_model_report"
)


parameters = {
    "max_features": [5],
    "min_samples_split": [2, 5],
    "max_depth": [3, 5],
    "ccp_alpha": [0.3, 0.5],
}
est = DecisionTreeRegressor()
dtreg_grid = GridSearchCV(est, parameters, cv=2, n_jobs=-1, verbose=True)

dtreg_grid.fit(X_train, y_train)

print(dtreg_grid.best_score_)
print(dtreg_grid.best_params_)

decision_tree = Pipeline(
    [
        (
            "",
            FunctionTransformer(
                _custom_data_transform, kw_args={"cols2keep": imp_features}
            ),
        ),
        ("Decision Tree", dtreg_grid.best_estimator_),
    ]
)
decision_tree.fit(X_train, y_train)

decision_tree_report = RegressionReport(
    model=decision_tree,
    x_train=X_train,
    y_train=y_train,
    x_test=X_test,
    y_test=y_test,
    refit=True,
)
decision_tree_report.get_report(include_shap=False, file_path="decision_tree_report")


parameters = {
    "n_estimators": [100, 200, 300],
    "min_samples_split": [2, 5],
    "max_depth": [3, 5],
    "ccp_alpha": [0.3, 0.5],
    "bootstrap": [True, False],
}
est = RandomForestRegressor()
rf_grid = GridSearchCV(est, parameters, cv=2, n_jobs=-1, verbose=True)

rf_grid.fit(X_train, y_train)

print(rf_grid.best_score_)
print(rf_grid.best_params_)

random_forest = Pipeline(
    [
        (
            "",
            FunctionTransformer(
                _custom_data_transform, kw_args={"cols2keep": imp_features}
            ),
        ),
        ("Random Forest", rf_grid.best_estimator_),
    ]
)
random_forest.fit(X_train, y_train)

rf_report = RegressionReport(
    model=random_forest,
    x_train=X_train,
    y_train=y_train,
    x_test=X_test,
    y_test=y_test,
    refit=True,
)
rf_report.get_report(include_shap=False, file_path="random_forest_report")

parameters = {
    "n_estimators": [100],
    "min_samples_split": [5],
    "bootstrap": [True, False],
}
est = XGBRegressor()
xgb_grid = GridSearchCV(est, parameters, cv=2, n_jobs=-1, verbose=True)

xgb_grid.fit(X_train, y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

xgboost = Pipeline(
    [
        (
            "",
            FunctionTransformer(
                _custom_data_transform, kw_args={"cols2keep": imp_features}
            ),
        ),
        ("XGBoost", xgb_grid.best_estimator_),
    ]
)
xgboost.fit(X_train, y_train)

xgb_report = RegressionReport(
    model=xgboost,
    x_train=X_train,
    y_train=y_train,
    x_test=X_test,
    y_test=y_test,
    refit=True,
)
xgb_report.get_report(include_shap=False, file_path="xgb_report")

