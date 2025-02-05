import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import random
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

print("Starting the script...")  # Ensures the script starts running

def define_files():
    files = {
        "air_pollution_death": "data/air_pollution_death.csv",
        "transportation": 'data/road-transportation_country_emissions.csv',
        "coal": 'data/coal-mining_country_emissions.csv',
        "cropland": 'data/cropland-fires_country_emissions.csv',
        "residential_commercial": 'data/residential-and-commercial-onsite-fuel-usage_country_emissions.csv',
        "forest_clearing": 'data/forest-land-clearing_country_emissions.csv',
        "petrochemicals": 'data/petrochemicals_country_emissions.csv',
        "electricity_generation": 'data/electricity-generation_country_emissions.csv',
        "incineration_open_burning": 'data/incineration-and-open-burning-of-waste_country_emissions.csv',
        "health_expenditure": 'data/health-expenditure.csv',
        "urban_population": 'data/urban-population.csv'
    }
    return files

def clean_column_names(df):
    """Removes quotes and trims spaces from column names."""
    df.columns = df.columns.str.replace('"', '', regex=False).str.replace("'", "", regex=False).str.strip()
    return df

def get_common_country_codes(files, env_factors, socio_factors, air_pollution_file):
    env_country_codes = set()
    socio_country_codes = set()
    air_pollution_country_codes = set()

    # Extract unique country codes from environmental factors (using 'iso3_country')
    for factor in env_factors:
        try:
            df = pd.read_csv(files[factor], on_bad_lines='skip')

            # Clean column names
            df = clean_column_names(df)

            if "iso3_country" in df.columns:  # Using "iso3_country" for country codes
                env_country_codes.update(df["iso3_country"].dropna().unique())
            else:
                print(f"WARNING: 'iso3_country' column not found in {factor} dataset!")

        except Exception as e:
            print(f"Error processing {factor}: {e}")

    # Extract unique country codes from socio-economic factors (using 'Country Code')
    for factor in socio_factors:
        try:
            skip_rows = 3  # Skip metadata rows
            df = pd.read_csv(files[factor], skiprows=skip_rows, on_bad_lines='skip')

            # Clean column names
            df = clean_column_names(df)

            if "Country Code" in df.columns:  # Using "Country Code" for country codes
                socio_country_codes.update(df["Country Code"].dropna().unique())
            else:
                print(f"WARNING: 'Country Code' column not found in {factor} dataset!")

        except Exception as e:
            print(f"Error processing {factor}: {e}")

    # Extract unique country codes from air pollution data (using 'SpatialDimValueCode')
    try:
        df = pd.read_csv(files[air_pollution_file], on_bad_lines='skip')

        # Clean column names
        df = clean_column_names(df)

        if "SpatialDimValueCode" in df.columns:  # Using "SpatialDimValueCode" for country codes
            air_pollution_country_codes.update(df["SpatialDimValueCode"].dropna().unique())
        else:
            print(f"WARNING: 'SpatialDimValueCode' column not found in {air_pollution_file} dataset!")

    except Exception as e:
        print(f"Error processing {air_pollution_file}: {e}")

    # Find the common country codes across all datasets
    common_country_codes = sorted(
        env_country_codes.intersection(socio_country_codes).intersection(air_pollution_country_codes)
    )

    print(f"Common Country Codes Found: {common_country_codes}")
    return common_country_codes

def load_and_filter_data(files, common_country_codes):
    air_pollution_df = pd.read_csv(files["air_pollution_death"], on_bad_lines='skip')
    air_pollution_df = clean_column_names(air_pollution_df)

    if "SpatialDimValueCode" not in air_pollution_df.columns:
        print("ERROR: 'SpatialDimValueCode' column missing in air_pollution_death dataset!")
        return None, None, None

    air_pollution_df = air_pollution_df[
        (air_pollution_df["SpatialDimValueCode"].isin(common_country_codes)) & 
        (air_pollution_df["Period"] == 2018) & 
        (air_pollution_df["Dim1"] == "Both sexes")
    ]
    
    env_data = {}
    environment_factor_files_list = [
        'transportation', 'coal', 'cropland', 'residential_commercial', 
        'forest_clearing', 'petrochemicals', 'electricity_generation', 
        'incineration_open_burning'
    ]
    
    for factor in environment_factor_files_list:
        df = pd.read_csv(files[factor], on_bad_lines='skip')
        df = clean_column_names(df)
        if "iso3_country" in df.columns:
            df = df[df["iso3_country"].isin(common_country_codes)]
        else:
            print(f"WARNING: 'iso3_country' not found in {factor} dataset!")
        env_data[factor] = df
    
    socio_data = {}
    socioeconomic_files_list = ['health_expenditure', 'urban_population']
    
    for factor in socioeconomic_files_list:
        skip_rows = 3  # Skip metadata rows
        df = pd.read_csv(files[factor], skiprows=skip_rows, on_bad_lines='skip')
        df = clean_column_names(df)
        if "Country Code" in df.columns:
            df = df[df["Country Code"].isin(common_country_codes)]
        else:
            print(f"WARNING: 'Country Code' not found in {factor} dataset!")
        socio_data[factor] = df
    
    return air_pollution_df, env_data, socio_data

def merge_data(air_pollution_df, env_data, socio_data):
    merged_df = air_pollution_df.copy()

    for factor, df in env_data.items():
        if "iso3_country" in df.columns and "Value" in df.columns:
            merged_df = pd.merge(merged_df, df[['iso3_country', 'Value']], on='iso3_country', how='left', suffixes=('', f'_{factor}'))

    for factor, df in socio_data.items():
        if "Country Code" in df.columns and "Value" in df.columns:
            merged_df = pd.merge(merged_df, df[['Country Code', 'Value']], on='Country Code', how='left', suffixes=('', f'_{factor}'))

    return merged_df

def start_predict_xgboost():
    try:
        random_seed = 42
        np.random.seed(random_seed)
        random.seed(random_seed)

        print("Defining files...")
        files = define_files()
        print("Files defined successfully!")

        print("Getting common country codes...")
        environment_factor_files_list = [
            'transportation', 'coal', 'cropland', 'residential_commercial', 
            'forest_clearing', 'petrochemicals', 'electricity_generation', 
            'incineration_open_burning'
        ]
        socioeconomic_files_list = ['health_expenditure', 'urban_population']
        common_country_codes = get_common_country_codes(files, environment_factor_files_list, socioeconomic_files_list, "air_pollution_death")

        print("Loading and filtering data...")
        air_pollution_df, env_data, socio_data = load_and_filter_data(files, common_country_codes)

        if air_pollution_df is None:
            print("ERROR: Data loading failed.")
            return
        print("Merging data...")
        merged_df = merge_data(air_pollution_df, env_data, socio_data)
        print(f"Merged data has {merged_df.shape[0]} rows and {merged_df.shape[1]} columns.")
        print("air_pollution->",merged_df)

    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    print("Calling start_predict_xgboost()...")
    start_predict_xgboost()
    print("Script execution complete!")
