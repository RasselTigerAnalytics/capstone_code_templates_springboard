"""
Tests if the data is aggregated at Saturday level.
"""

import pandas as pd

file_path = ('/home/rassel/capstone_code_templates_springboard/data/processed/merge_google_search_data.csv')


def test_if_data_is_aggregated_at_the_granularity_level_of_saturdays():
    """
    Tests if the data is aggregated at the granularity level of saturdays.
    """
    social_media_data = pd.read_csv(file_path)
    sample_row = social_media_data.sample(1)
    sample_row['date'] = pd.to_datetime(sample_row['date'])
    date_value = sample_row['date'].dt.day_name()
    date_value = date_value.iloc[0]
    assert date_value == 'Saturday'




