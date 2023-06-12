from ta_lib.regression.api import (
    RegressionReport,
    SKLStatsmodelOLS,
)
# standard code-template imports
from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    create_context,
    initialize_environment,
    load_dataset,
    save_dataset,
)
from xgboost import XGBRegressor
import os.path as op
# standard third party imports
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor
# impute missing values
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeRegressor

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

initialize_environment(debug=False, hide_warnings=True)

artifacts_folder = DEFAULT_ARTIFACTS_PATH

config_path = op.join('/home/rassel/capstone_code_templates_springboard/production/conf', 'config.yml')
context = create_context(config_path)

ground_truth_for_modelling = load_dataset(
    context, "/ground_truth_for_modelling/ground_truth"
)


def create_price_variables(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create price related variables

    Arguments
    ---------
    dataframe : pd.DataFrame
        The dataframe to use for creating the price related variables.

    Returns
    -------
    dataframe
        The processed dataframe
    """
    temp_dataframe = dataframe.copy()
    temp_dataframe["client_unit_price"] = (
        temp_dataframe["client_dollars_value"]
        / temp_dataframe["client_units_value"]
    )
    temp_dataframe["competitor_unit_price"] = (
        temp_dataframe["competitor_dollars_value"]
        / temp_dataframe["competitor_units_value"]
    )
    return temp_dataframe


ground_truth_for_modelling = create_price_variables(ground_truth_for_modelling)
ground_truth_for_modelling.drop(columns=["Unnamed: 0"], inplace=True)


numerical_features = ground_truth_for_modelling.select_dtypes(include="number")


def create_train_and_test_split(dataframe: pd.DataFrame) -> tuple:
    """
    Creates train and test related dataframes.

    Arguments
    ---------
    dataframe : pd.DataFrame
        The base dataframe to use for getting the train and test splits.

    Returns
    -------
    tuple
        The tuple of dataframes after the train and test split has been done.
    """
    X = dataframe.drop(columns=["client_lbs_value"])
    y = dataframe["client_lbs_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = create_train_and_test_split(ground_truth_for_modelling)

save_dataset(context, X_train, "/train/ground_truth_for_modelling/features")
save_dataset(context, y_train, "/train/ground_truth_for_modelling/target")
save_dataset(context, X_test, "/test/ground_truth_for_modelling/features")
save_dataset(context, y_test, "/test/ground_truth_for_modelling/target")

# collecting different types of columns for transformations
cat_columns = X_train.select_dtypes("object").columns
num_columns = X_train.select_dtypes("number").columns


def one_hot_encode_column(dataframe: pd.DataFrame, cat_variable: str) -> pd.DataFrame:
    """
    One hot encodes the categorical column specified.

    Arguments
    ---------
    dataframe : pd.DataFrame
        The dataframe to use.
    cat_variable: str
        The categorical variable to one hot encode.

    Returns
    -------
    dataframe
        The one hot encoded dataframe
    """
    one_hot = pd.get_dummies(dataframe[cat_variable], drop_first=True)
    # Drop column B as it is now encoded
    dataframe = dataframe.drop("theme_name", axis=1)  # Join the encoded df
    dataframe = dataframe.join(one_hot)
    return dataframe


def engineer_date_related_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers date related features.

    Arguments
    ---------
    dataframe : pd.DataFrame
        The dataframe to use for engineering the date related features.

    Returns
    -------
    dataframe
        The dataframe with the engineered features.
    """
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
X_test = feature_engineering_pipeline_test.fit_transform(X_test)


def create_final_set_of_training_and_test_datasets(train_df : pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Create the final training and test datasets.

    Arguments
    ---------
    train_df : pd.DataFrame
        The train dataframe
    test_df : pd.DataFrame
        The test dataframe

    Returns
    -------
    tuple
        The tuple of the final training and test datasets.

    """
    columns_to_drop = ["chewy_searchVolume", "total_searchVolume"]
    train_df.drop(columns=columns_to_drop, inplace=True)
    test_df.drop(columns=columns_to_drop, inplace=True)

    reg_vars = train_df.columns.tolist()
    train_df = train_df[reg_vars]
    test_df = test_df[reg_vars]

    return train_df, test_df, reg_vars


X_train, X_test, reg_vars = create_final_set_of_training_and_test_datasets(X_train, X_test)


# Custom Transformations like these can be utilised
def _custom_data_transform(df: pd.DataFrame, cols2keep=None):
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
