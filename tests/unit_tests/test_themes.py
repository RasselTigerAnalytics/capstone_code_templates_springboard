"""
Script to test the number of themes
"""
import os.path as op
import pandas as pd

file_path = ('/home/rassel/capstone_code_templates_springboard/data/merged_data/google_search_data.csv')


def test_if_common_themes_are_present():
    """
    Checks if the correct number of common themes are present
    """
    social_media_data = pd.read_csv(file_path)
    print(len(social_media_data['theme_name'].unique.tolist()))
    assert len(social_media_data['theme_name'].unique.tolist()) == 30
