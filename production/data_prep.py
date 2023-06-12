# Project imports
from ta_lib.core.api import initialize_environment, save_dataset
from ta_lib.core.api import create_context, load_dataset
import os.path as op
import pandas as pd

pd.set_option("display.max_rows", 500)
# Initialization
initialize_environment(debug=False, hide_warnings=True)
config_path = op.join('/home/rassel/capstone_code_templates_springboard/production/conf', 'config.yml')
context = create_context(config_path)

google_search_data = load_dataset(context, "/raw/google_search")
product_manufacturer_list = load_dataset(context, "raw/product_manufacturer_list")
sales_data = load_dataset(context, "raw/sales")
social_media_data = load_dataset(context, "raw/social_media")
Theme_product_list = load_dataset(context, "raw/Theme_product_list")
Theme_list = load_dataset(context, "raw/Theme_list")


# sales_data
sales_data["system_calendar_key_N"] = pd.to_datetime(
    sales_data["system_calendar_key_N"].astype(str), format="%Y%m%d"
)
sales_data.rename(columns={"system_calendar_key_N": "date"}, inplace=True)

# social_media_data
social_media_data["published_date"] = pd.to_datetime(
    social_media_data["published_date"], errors="coerce"
)
social_media_data.rename(
    columns={"published_date": "date", "Theme Id": "theme_id"}, inplace=True
)

# google_search_data
google_search_data["date"] = pd.to_datetime(
    google_search_data["date"], format="%d-%m-%Y"
)
google_search_data.rename(columns={"Claim_ID": "theme_id"}, inplace=True)
google_search_data.drop(columns=["week_number", "year_new"], inplace=True)


# Theme_product_list
Theme_product_list.rename(
    columns={"PRODUCT_ID": "product_id", "CLAIM_ID": "theme_id"}, inplace=True
)

# product_manufacturer_list
product_manufacturer_list.drop(
    columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4", "Unnamed: 5", "Unnamed: 6"],
    inplace=True,
)
product_manufacturer_list.rename(
    columns={"PRODUCT_ID": "product_id", "Vendor": "vendor"}, inplace=True
)

# Theme_list
Theme_list.rename(
    columns={"CLAIM_ID": "theme_id", "Claim Name": "theme_name"}, inplace=True
)

# Step 1: Identify rows with zero sales_lbs_value
zero_sales_rows = sales_data[sales_data["sales_lbs_value"] == 0]

# Step 2 and 3: Calculate average value for nonzero sales_lbs_value rows
grouped = sales_data[sales_data["sales_lbs_value"] != 0].groupby("product_id")
get_imputed_value = (
    lambda row: grouped.get_group(row["product_id"])["sales_lbs_value"].sum()
    / grouped.get_group(row["product_id"])["sales_units_value"].sum()
    if row["product_id"] in grouped.groups
    else 0
)
imputed_values = zero_sales_rows.apply(get_imputed_value, axis=1)

# Step 4: Replace zero sales_lbs_value with the imputed values
sales_data.loc[sales_data["sales_lbs_value"] == 0, "sales_lbs_value"] = (
    imputed_values * zero_sales_rows["sales_units_value"]
)

# Print the updated DataFrame
sales_data[sales_data.product_id == 34090]


# Step 1: Identify rows with zero sales_dollars_value
zero_sales_rows = sales_data[sales_data["sales_dollars_value"] == 0]

# Step 2 and 3: Calculate average value for nonzero sales_dollars_value rows
grouped = sales_data[sales_data["sales_dollars_value"] != 0].groupby("product_id")
get_imputed_value = (
    lambda row: grouped.get_group(row["product_id"])["sales_dollars_value"].sum()
    / grouped.get_group(row["product_id"])["sales_units_value"].sum()
    if row["product_id"] in grouped.groups
    else 0
)
imputed_values = zero_sales_rows.apply(get_imputed_value, axis=1)

# Step 4: Replace zero sales_lbs_value with the imputed values
sales_data.loc[sales_data["sales_dollars_value"] == 0, "sales_dollars_value"] = (
    imputed_values * zero_sales_rows["sales_units_value"]
)

# Print the updated DataFrame
sales_data[sales_data.product_id == 11443]

pro_sales_data = sales_data[
    (sales_data["sales_dollars_value"] != 0) & (sales_data["sales_lbs_value"] != 0)
]
len(pro_sales_data)

save_dataset(context, pro_sales_data, "processed_data/sales")

google_search_data.drop_duplicates(inplace=True)

unique_theme_count = google_search_data.groupby("platform")["theme_id"].nunique()

google_search_data["year"] = google_search_data["date"].dt.year

# Count the unique theme_id for each year and platform
unique_theme_count = (
    google_search_data.groupby(["year", "platform"])["theme_id"].nunique().reset_index()
)

# Group by 'theme_id', 'platform', and weekly periods starting from Sunday
weekly_agg = (
    google_search_data.groupby(
        [pd.Grouper(key="date", freq="W-SAT"), "theme_id", "platform"]
    )["searchVolume"]
    .sum()
    .reset_index()
)

# Create a new DataFrame with four different platforms
platforms = ["google", "chewy", "amazon", "walmart"]
new_df = weekly_agg.pivot(
    index=["date", "theme_id"], columns="platform", values="searchVolume"
).reindex(columns=platforms)

pro_google_search_data = new_df.reset_index()

pro_google_search_data.rename(
    columns={
        "google": "google_searchVolume",
        "chewy": "chewy_searchVolume",
        "amazon": "amazon_searchVolume",
        "walmart": "walmart_searchVolume",
    },
    inplace=True,
)
pro_google_search_data = pro_google_search_data.fillna(0)
pro_google_search_data["total_searchVolume"] = (
    pro_google_search_data.google_searchVolume
    + pro_google_search_data.walmart_searchVolume
    + pro_google_search_data.chewy_searchVolume
    + pro_google_search_data.amazon_searchVolume
)
pro_google_search_data.head()

save_dataset(context, pro_google_search_data, "processed_data/google_search")

social_media_data.dropna(inplace=True)
social_media_data.drop_duplicates(inplace=True)

pro_social_media_data = (
    social_media_data.groupby([pd.Grouper(key="date", freq="W-SAT"), "theme_id"])[
        "total_post"
    ]
    .sum()
    .reset_index()
)
save_dataset(context, pro_social_media_data, "processed_data/social_media")

gs_theme_id = pro_google_search_data.theme_id.unique()
sm_theme_id = pro_social_media_data.theme_id.unique()
product_theme_id = Theme_product_list.theme_id.unique()
all_theme_id = Theme_list.theme_id.unique()

p = [
    "ethical - not specific",
    "ethical",
    "ethical - animal/fish & bird",
    "ethical - environment",
    "ethical - human",
    "fish",
]
a = Theme_list[Theme_list.theme_name.isin(p)].theme_id
for i in a:
    if i in product_theme_id:
        print(i)
        print(Theme_list[Theme_list.theme_id == i].theme_name)


save_dataset(
    context, product_manufacturer_list, "processed_data/product_manufacturer_list"
)
save_dataset(context, Theme_product_list, "processed_data/Theme_product_list")
save_dataset(context, Theme_list, "processed_data/Theme_list")


# pro_social_media_data=pd.read_csv('processed_data/pro_social_media_data.csv', index_col=0)
pro_social_media_data = load_dataset(context, "processed_data/social_media")
pro_social_media_data.drop("Unnamed: 0", axis=1, inplace=True)


# Convert the 'date' column to datetime format
pro_social_media_data["date"] = pd.to_datetime(pro_social_media_data["date"])

# Set the 'date' column as the DataFrame's index
pro_social_media_data.set_index("date", inplace=True)

# Resample the DataFrame to get the desired time intervals
df_weekly = pro_social_media_data.resample("W").sum()
df_monthly = pro_social_media_data.resample("M").sum()
df_quarterly = pro_social_media_data.resample("Q").sum()
df_yearly = pro_social_media_data.resample("Y").sum()

pro_google_search_data = load_dataset(context, "processed_data/google_search")
pro_google_search_data.drop("Unnamed: 0", axis=1, inplace=True)

pro_google_search_data = load_dataset(context, "processed_data/google_search")
pro_google_search_data.drop("Unnamed: 0", axis=1, inplace=True)


# Convert the 'date' column to datetime format
pro_google_search_data["date"] = pd.to_datetime(pro_google_search_data["date"])

# Set the 'date' column as the DataFrame's index
pro_google_search_data.set_index("date", inplace=True)

# Resample the DataFrame to get the desired time intervals
df_weekly = pro_google_search_data.resample("W").sum()
df_monthly = pro_google_search_data.resample("M").sum()
df_quarterly = pro_google_search_data.resample("Q").sum()
df_yearly = pro_google_search_data.resample("Y").sum()


pro_sales_data = load_dataset(context, "processed_data/sales")
# pro_sales_data=pd.read_csv('processed_data/pro_sales_data.csv', index_col=0)
pro_sales_data = load_dataset(context, "processed_data/sales")
pro_sales_data.drop("Unnamed: 0", axis=1, inplace=True)


# Convert the 'date' column to datetime format
pro_sales_data["date"] = pd.to_datetime(pro_sales_data["date"])

# Set the 'date' column as the DataFrame's index
pro_sales_data.set_index("date", inplace=True)

# Resample the DataFrame to get the desired time intervals
df_weekly = pro_sales_data.resample("W").sum()
df_monthly = pro_sales_data.resample("M").sum()
df_quarterly = pro_sales_data.resample("Q").sum()
df_yearly = pro_sales_data.resample("Y").sum()


pro_sales_data = load_dataset(context, "processed_data/sales")
pro_google_search_data = load_dataset(context, "processed_data/google_search")
pro_social_media_data = load_dataset(context, "processed_data/social_media")

pro_product_manufacturer_list = load_dataset(
    context, "processed_data/product_manufacturer_list"
)
pro_Theme_product_list = load_dataset(context, "processed_data/Theme_product_list")
pro_Theme_list = load_dataset(context, "processed_data/Theme_list")


pro_sales_data.drop("Unnamed: 0", axis=1, inplace=True)
pro_google_search_data.drop("Unnamed: 0", axis=1, inplace=True)
pro_social_media_data.drop("Unnamed: 0", axis=1, inplace=True)

pro_product_manufacturer_list.drop("Unnamed: 0", axis=1, inplace=True)
pro_Theme_product_list.drop("Unnamed: 0", axis=1, inplace=True)
pro_Theme_list.drop("Unnamed: 0", axis=1, inplace=True)

# Merge sales_data with product_manufacturer_list
merge1 = pd.merge(
    pro_sales_data, pro_product_manufacturer_list, how="inner", on="product_id"
)

# Merge the result with Theme_product_list
merge_sales_data = pd.merge(
    merge1, pro_Theme_product_list, how="inner", on="product_id"
)
merge_sales_data = pd.merge(
    merge_sales_data, pro_Theme_list, how="inner", on="theme_id"
)

merge_sales_data.head()

unique_theme_count = merge_sales_data.groupby("vendor")["theme_id"].nunique()

save_dataset(context, merge_sales_data, "merged_data/sales")
merge_google_search_data = pd.merge(
    pro_google_search_data, pro_Theme_list, how="inner", on="theme_id"
)

save_dataset(context, merge_google_search_data, "merged_data/google_search")

merge_social_media_data = pd.merge(
    pro_social_media_data, pro_Theme_list, how="inner", on="theme_id"
)
save_dataset(context, merge_social_media_data, "merged_data/social_media")

merge_sales_data["date"] = pd.to_datetime(merge_sales_data["date"])
merge_google_search_data["date"] = pd.to_datetime(merge_google_search_data["date"])
merge_social_media_data["date"] = pd.to_datetime(merge_social_media_data["date"])

common_theme_name = list(
    set(merge_sales_data.theme_name)
    & set(merge_google_search_data.theme_name)
    & set(merge_social_media_data.theme_name)
)
len(common_theme_name)

merge_sales_data = merge_sales_data[merge_sales_data.theme_name.isin(common_theme_name)]
merge_google_search_data = merge_google_search_data[
    merge_google_search_data.theme_name.isin(common_theme_name)
]
merge_social_media_data = merge_social_media_data[
    merge_social_media_data.theme_name.isin(common_theme_name)
]

save_dataset(context, merge_sales_data, "/merged_data/sales")
save_dataset(context, merge_google_search_data, "/merged_data/google_search")
save_dataset(context, merge_social_media_data, "/merged_data/social_media")
