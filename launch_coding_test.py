import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import plotly.express as px
import xgboost as xgb
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

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
            df = clean_column_names(df)

            if "iso3_country" in df.columns:
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
            df = clean_column_names(df)

            if "Country Code" in df.columns:
                socio_country_codes.update(df["Country Code"].dropna().unique())
            else:
                print(f"WARNING: 'Country Code' column not found in {factor} dataset!")
        except Exception as e:
            print(f"Error processing {factor}: {e}")

    # Extract unique country codes from air pollution data (using 'SpatialDimValueCode')
    try:
        df = pd.read_csv(files[air_pollution_file], on_bad_lines='skip')
        df = clean_column_names(df)

        if "SpatialDimValueCode" in df.columns:
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

    # Filter data for common country codes and 2018
    air_pollution_df = air_pollution_df[(
        air_pollution_df["SpatialDimValueCode"].isin(common_country_codes)) & 
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

    for df in env_data.items():
        if "iso3_country" in df.columns and "Value" in df.columns:
            merged_df = pd.merge(merged_df, df[['iso3_country', 'Value']], on='iso3_country', how='outer')

    for df in socio_data.items():
        if "Country Code" in df.columns and "Value" in df.columns:
            merged_df = pd.merge(merged_df, df[['Country Code', 'Value']], on='Country Code', how='outer')
    return merged_df

def process_environment_and_socioeconomic_data(env_data, socio_data, common_country_codes):
    environment_results = []
    socioeconomic_results = []

    # **(B) Processing Environment Data**
    for factor, df in env_data.items():
        if "iso3_country" in df.columns and "start_time" in df.columns and "emissions_quantity" in df.columns:
            try:
                df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")  # Ensure proper date format
                df_filtered = df[(
                    df["iso3_country"].isin(common_country_codes)) & 
                    (df["start_time"] >= "2018-01-01")
                ]
                df_grouped = df_filtered.groupby("iso3_country", as_index=False)["emissions_quantity"].sum()
                df_grouped.rename(columns={"iso3_country": "Country Code", "emissions_quantity": factor}, inplace=True)
                environment_results.append(df_grouped)
            except Exception as e:
                print(f"Error processing {factor}: {e}")
        else:
            print(f"WARNING: Missing required columns in {factor} dataset!")

    # **(C) Processing Socioeconomic Data**
    for factor, df in socio_data.items():
        if "Country Code" in df.columns and "2018" in df.columns:
            try:
                df_filtered = df[df["Country Code"].isin(common_country_codes)][["Country Code", "2018"]]
                df_filtered.rename(columns={"2018": factor}, inplace=True)
                socioeconomic_results.append(df_filtered)
            except Exception as e:
                print(f"Error processing {factor}: {e}")
        else:
            print(f"WARNING: Missing required columns in {factor} dataset!")

    return environment_results, socioeconomic_results

def merge_environment_socioeconomic_air_pollution_data(environment_results, socioeconomic_results, air_pollution_df):
    # **(A) Merge environment_data and socioeconomic_data on common country codes**
    # environment_data_combined = pd.concat(environment_results, ignore_index=True)
    # socioeconomic_data_combined = pd.concat(socioeconomic_results, ignore_index=True)

    environment_data_combined = environment_results[0]
    for i in environment_results[1:]:
        environment_data_combined = pd.merge(environment_data_combined,i,on="Country Code", how="outer")
    
    merge_final = environment_data_combined
    
    for i in socioeconomic_results:
        socioeconomic_data_combined = pd.merge(merge_final,i,on="Country Code", how="outer")


    # Merge environment and socioeconomic data on 'Country Code'
    # merged_env_socio_data = pd.merge(environment_data_combined, socioeconomic_data_combined, on="Country Code", how="outer")
    
    # **(B) Aggregate air pollution deaths per country**
    air_pollution_agg = air_pollution_df.groupby("SpatialDimValueCode", as_index=False)["FactValueNumeric"].sum()
    air_pollution_agg.rename(columns={"SpatialDimValueCode": "Country Code"}, inplace=True)

    # **(C) Merge aggregated air pollution data**
    merged_data_with_deaths = pd.merge(merge_final, air_pollution_agg, 
                                       on='Country Code', how='outer')

    # Rename columns to meaningful names
    merged_data_with_deaths.rename(columns={
        'name_x': 'environmental_value',
        'name_y': 'socioeconomic_value',
        'Value': 'air_pollution_deaths'
    }, inplace=True)

    return merged_data_with_deaths

def start_predict_xgboost():
    # try:
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

        print("Processing environment and socioeconomic data...")
        environment_results, socioeconomic_results = process_environment_and_socioeconomic_data(env_data, socio_data, common_country_codes)

        # **Merge environment, socioeconomic, and air pollution data**
        print("Merging environment, socioeconomic, and air pollution data...")
        merged_data = merge_environment_socioeconomic_air_pollution_data(environment_results, socioeconomic_results, air_pollution_df)
        merged_data = merged_data[merged_data['Country Code'] != 'CHN']
        merged_data = merged_data[merged_data['Country Code'] != 'IND']
        # Rename the columns to more meaningful names
        # merged_data.rename(columns={
        #     'name_x': 'environmental_value',
        #     'name_y': 'socioeconomic_value',
        #     'SpatialDimValueCode': 'Country Code',
        #     'Value': 'air_pollution_deaths'  # Assuming 'Value' represents the air pollution deaths
        # }, inplace=True)

        print(f"Merged data has {merged_data.shape[0]} rows and {merged_data.shape[1]} columns.")
        print(merged_data)  # Optional: Check the first few rows of the merged dataframe

        model, X_train_val, y_train_val = train_model(merged_data)

        generate_beeswarm_plot(model, X_train_val)

        if merged_data is None or merged_data.empty:
            print("ERROR: merged_data is empty or not defined.")
        else:
            merged_data.to_csv("merged_data.csv", index=False)
            print("merged_data.csv saved successfully!")

    # except Exception as e:
    #     print("An error occurred:", e)

def train_model(df):
    X = df.drop(['SpatialDimValueCode', 'Air Pollution Deaths','FactValueNumeric'], axis=1, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    # y = df['Air Pollution Deaths'].astype(float)
    y = df['FactValueNumeric'].astype(float)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    scaler = StandardScaler()
    X_train_val = pd.DataFrame(scaler.fit_transform(X_train_val), columns=X_train_val.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    xgb_model = xgb.XGBRegressor(
        subsample=0.2,
        colsample_bytree=0.7,
        reg_alpha=10,
        reg_lambda=10,
        random_state=random_seed,
        n_estimators=199,
        learning_rate=0.066,
        max_depth=4,
        predictor='cpu_predictor'
    )

    xgb_model.fit(X_train_val, y_train_val)
    print(f'Train R^2 Score: {r2_score(y_train_val, xgb_model.predict(X_train_val))}')
    print(f'Test R^2 Score: {r2_score(y_test, xgb_model.predict(X_test))}')

    return xgb_model, X_train_val, y_train_val

def generate_beeswarm_plot(model, X_train_val):
    explainer = shap.Explainer(model, X_train_val)
    # shap_values = explainer.shap_values(X_train_val, check_additivity=False)
    shap_values = explainer(X_train_val)
    shap.summary_plot(shap_values, X_train_val, plot_type="dot")


if __name__ == "__main__":
    print("Calling start_predict_xgboost()...")
    start_predict_xgboost()
    print("Script execution complete!")
