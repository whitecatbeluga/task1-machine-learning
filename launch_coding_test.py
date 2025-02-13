import optuna
import plotly.express as px

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import plotly.express as px
import xgboost as xgb
import shap
import plotly.express as px

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
    # **(A) Merge environmental data**
    environment_data_combined = environment_results[0]
    for env_df in environment_results[1:]:
        environment_data_combined = pd.merge(environment_data_combined, env_df, on="Country Code", how="outer")

    # **(B) Merge socioeconomic data**
    socioeconomic_data_combined = socioeconomic_results[0]
    for socio_df in socioeconomic_results[1:]:
        socioeconomic_data_combined = pd.merge(socioeconomic_data_combined, socio_df, on="Country Code", how="outer")

    # **(C) Merge environmental and socioeconomic data**
    merged_env_socio_data = pd.merge(environment_data_combined, socioeconomic_data_combined, on="Country Code", how="outer")

    # **(D) Aggregate air pollution deaths per country**
    air_pollution_agg = air_pollution_df.groupby("SpatialDimValueCode", as_index=False)["FactValueNumeric"].sum()
    air_pollution_agg.rename(columns={"SpatialDimValueCode": "Country Code"}, inplace=True)

    # **(E) Merge with aggregated air pollution data**
    merged_data_with_deaths = pd.merge(merged_env_socio_data, air_pollution_agg, on='Country Code', how='outer')

    # **(F) Rename columns to meaningful names**
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
        
        # print("socioeconomic_results->>>>>>>>>>>>>>>\n",socioeconomic_results)

        # **Merge environment, socioeconomic, and air pollution data**
        print("Merging environment, socioeconomic, and air pollution data...")
        merged_data = merge_environment_socioeconomic_air_pollution_data(environment_results, socioeconomic_results, air_pollution_df)
        merged_data_with_outliers = merge_environment_socioeconomic_air_pollution_data(environment_results, socioeconomic_results, air_pollution_df)
        # CHN and IND should be optional it can be toogled from include and exclude
        merged_data_with_outliers = merged_data_with_outliers[merged_data_with_outliers['Country Code'] != 'CHN']
        merged_data_with_outliers = merged_data_with_outliers[merged_data_with_outliers['Country Code'] != 'IND']

        # if exclude_countries:
            # merged_data = merged_data[~merged_data['Country Code'].isin(['CHN', 'IND'])]


        print(f"Merged data has {merged_data.shape[0]} rows and {merged_data.shape[1]} columns.")
        print(merged_data)  # Optional: Check the first few rows of the merged dataframe

        model, X_train_val, y_train_val, train_r2, test_r2 = train_model(merged_data)
        model_with, X_train_val_with, y_train_val_with, train_r2_with, test_r2_with = train_model(merged_data_with_outliers)

        generate_beeswarm_plot(model_with, X_train_val_with)
        
        generate_html_file(merged_data, train_r2_with, test_r2_with)

        # Generate merge data
        # if merged_data is None or merged_data.empty:
        #     print("ERROR: merged_data is empty or not defined.")
        # else:
        #     merged_data.to_csv("merged_data.csv", index=False)
        #     print("merged_data.csv saved successfully!")

    # except Exception as e:
    #     print("An error occurred:", e)

def train_model(df):
    # Prepare dataset
    X = df.drop(['SpatialDimValueCode', 'Air Pollution Deaths', 'FactValueNumeric'], axis=1, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df['FactValueNumeric'].astype(float)

    # Train-Test Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # Standardize features
    scaler = StandardScaler()
    X_train_val = pd.DataFrame(scaler.fit_transform(X_train_val), columns=X_train_val.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Optuna optimization function
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 1500, 2200, step=100),  # More boosting rounds
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.009, log=True),  # More precise learning
            'max_depth': trial.suggest_int('max_depth', 6, 7),  # Balanced complexity
            'min_child_weight': trial.suggest_int('min_child_weight', 6, 10),  # Prevent overfitting
            'subsample': trial.suggest_float('subsample', 0.95, 1.0),  # Use more training data
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.95, 1.0),  # More features per tree
            'reg_alpha': trial.suggest_float('reg_alpha', 2, 8),  # Reduce restriction
            'reg_lambda': trial.suggest_float('reg_lambda', 2, 8),  # Reduce restriction
            'gamma': trial.suggest_float('gamma', 1, 4),  # Allow better splits
            'random_state': random_seed,
            'early_stopping_rounds': 50
        }

        # Train XGBoost model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_val, y_train_val, eval_set=[(X_test, y_test)], verbose=False)

        # Evaluate R^2 Score
        test_pred = model.predict(X_test)
        return r2_score(y_test, test_pred)

    # Run Optuna Optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # Run 50 trialsc

    # Get best parameters
    best_params = study.best_params
    print("Best parameters found: ", best_params)

    # Train final model with best parameters
    xgb_model = xgb.XGBRegressor(**best_params)
    xgb_model.fit(X_train_val, y_train_val, eval_set=[(X_test, y_test)], verbose=True)

    # Final model evaluation
    train_r2 = r2_score(y_train_val, xgb_model.predict(X_train_val))
    test_r2 = r2_score(y_test, xgb_model.predict(X_test))

    print(f'Final Train R^2 Score: {train_r2}')
    print(f'Final Test R^2 Score: {test_r2}')

    # results = xgb_model.evals_result()
 
    # # Convert to DataFrame
    # iterations = range(len(results['validation_0']['rmse']))  # XGBoost logs RMSE by default
    # train_r2_list = [r2_score(y_train_val, xgb_model.predict(X_train_val, iteration_range=(0, i+1))) for i in iterations]
    # test_r2_list = [r2_score(y_test, xgb_model.predict(X_test, iteration_range=(0, i+1))) for i in iterations]
    
    # # Plot
    # plt.figure(figsize=(10, 5))
    # plt.plot(iterations, train_r2_list, label="Train R²")
    # plt.plot(iterations, test_r2_list, label="Test R²")
    
    # plt.xlabel("Iteration")
    # plt.ylabel("R² Score")
    # plt.title("Train vs Test R² Score Over Iterations")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return xgb_model, X_train_val, y_train_val, train_r2, test_r2

def generate_beeswarm_plot(model, X_train_val):
    # Rename columns: replace underscores with spaces and capitalize
    X_train_val = X_train_val.rename(columns=lambda x: x.replace('_', ' ').title())

    # Generate SHAP values
    explainer = shap.Explainer(model, X_train_val)
    shap_values = explainer(X_train_val)

    # Generate SHAP plot without displaying it
    shap.summary_plot(shap_values, X_train_val, plot_type="dot", show=False)

    # Save the current figure generated by SHAP
    fig = plt.gcf()  # Get the current figure
    fig.savefig("beeswarm_plot.png", bbox_inches='tight')  

    plt.close(fig)

def generate_html_file(merged_data, train_r2, test_r2):
    merged_data['Country Code'] = merged_data['Country Code'].str.strip()  # Ensure clean data
    
    factors = [col for col in merged_data.columns if col not in ['Country Code', 'FactValueNumeric']]
    data_without_chn_ind = merged_data[~merged_data['Country Code'].isin(['CHN', 'IND'])]

    scatter_html_blocks = []                      # For all countries
    scatter_html_blocks_without_chn_ind = []      # Excluding CHN and IND

    for factor in factors:
        # Scatter plots with CHN & IND
        correlation_all = merged_data['FactValueNumeric'].corr(merged_data[factor])
        fig_all = px.scatter(
            merged_data,
            x=factor,
            y='FactValueNumeric',
            text="Country Code",
            title=f"Air Pollution Deaths vs {factor.replace('_', ' ').title()} (Corr: {round(correlation_all, 2)})",
            labels={'FactValueNumeric': 'Air Pollution Deaths'}
        )
        scatter_html_blocks.append((factor, fig_all.to_html(full_html=False)))

        # Scatter plots without CHN & IND
        correlation_without = data_without_chn_ind['FactValueNumeric'].corr(data_without_chn_ind[factor])
        fig_without = px.scatter(
            data_without_chn_ind,
            x=factor,
            y='FactValueNumeric',
            text="Country Code",
            title=f"Air Pollution Deaths vs {factor.replace('_', ' ').title()} (Excl. CHN & IND, Corr: {round(correlation_without, 2)})",
            labels={'FactValueNumeric': 'Air Pollution Deaths'}
        )
        scatter_html_blocks_without_chn_ind.append((factor, fig_without.to_html(full_html=False)))

        choropleth_map = px.choropleth(
            merged_data,
            locations="Country Code",
            color="FactValueNumeric",
            hover_data={"Country Code": True, "FactValueNumeric": True},
            color_continuous_scale=px.colors.sequential.Reds,
            labels={"FactValueNumeric": "Air Pollution Deaths"}  # Add label here
        ).to_html(full_html=False)

        with open("coding_test_output.html", "w", encoding="utf-8") as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Air Pollution and Emissions Analysis</title>
                    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
                    <style>
                        body {{
                            font-family: 'Montserrat', sans-serif;
                        }}
                        .container {{
                            display: flex;
                            align-items: center;
                            justify-content: space-around;
                            gap: 20px;
                        }}
                        .radio-buttons {{
                            display: flex;
                            flex-direction: column;
                            min-width: 150px;
                        }}
                        select {{
                            padding: 3px;
                        }}
                        .option-style {{
                            padding: 2px;
                        }}
                        .scatter-plot {{
                            display: none;
                        }}
                    </style>
                    <script>
                        let selectedFactorIndex = 0; // Track the selected factor

                        function showPlot(factorIndex) {{
                            selectedFactorIndex = factorIndex;
                            const withCHN = document.getElementsByClassName('with-china-india');
                            const withoutCHN = document.getElementsByClassName('without-china-india');
                            const selection = document.getElementById("filter-select").value;

                            // Hide all plots
                            for (let plot of withCHN) plot.style.display = 'none';
                            for (let plot of withoutCHN) plot.style.display = 'none';

                            // Determine active dataset
                            const activeClass = selection === 'with_china_india' ? withCHN : withoutCHN;
                            // Ensure selectedFactorIndex is within bounds
                            if (selectedFactorIndex >= activeClass.length) {{
                                selectedFactorIndex = 0; // Reset if out of bounds
                            }}
                            // Display the corresponding plot
                            if(activeClass === withoutCHN){{
                                if(selectedFactorIndex === 0){{
                                    selectedFactorIndex = "plot_without_"+selectedFactorIndex;
                                }}
                                let new_id = selectedFactorIndex.split("_")
                                let active_plot = "plot_without_"+new_id[new_id.length - 1];
                                activeClass[active_plot].style.display = 'block';
                            }}else if(activeClass === withCHN){{
                                if(selectedFactorIndex === 0){{
                                    selectedFactorIndex = "plot_with_"+selectedFactorIndex;
                                }}
                                let new_id = selectedFactorIndex.split("_")
                                let active_plot = "plot_with_"+new_id[new_id.length - 1];
                                activeClass[active_plot].style.display = 'block';
                            }}
                            if (activeClass[selectedFactorIndex]) {{
                                activeClass[selectedFactorIndex].style.display = 'block';
                            }}
                        }}

                        function toggleCountries() {{
                            showPlot(selectedFactorIndex); // Maintain the same factor when toggling
                        }}
                    </script>
                </head>
                <body>
                    <h2 style="text-align:center; margin:20px;">Air Pollution Deaths by Country</h2>
                    {choropleth_map}
                    <hr style="width:80%"/>
                    <h2 style="text-align:center;">Scatter Plots</h2>
                    <div class="container">
                        <div class="radio-buttons">
            """)

            # Add radio buttons for each scatter plot
            for i, (factor, scatter_html) in enumerate(scatter_html_blocks):
                plot_id_with = f"plot_with_{i}"
                # plot_id_without = f"plot_without_{i}"
                checked_attr = 'checked' if i == 0 else ''
                f.write(f"""
                    <label>
                        <input type="radio" name="scatter" onclick="showPlot('{plot_id_with}')" {checked_attr}>
                        {factor.replace("_", " ").title()}
                    </label>
                """)

            f.write("""
                <div class="option-container">
                        <br />
                        <label for="filter-select">Outlier Removal Option:</label>
                        <select id="filter-select" onchange="toggleCountries()">
                            <option class="option-style" value="with_china_india" selected>Include CHN & IND</option>
                            <option class="option-style" value="without_china_india">Exclude CHN & IND</option>
                        </select>
                    </div>
                </div> <!-- End of radio-buttons -->
            """)

            # Display scatter plots WITH CHN & IND
            for i, (factor, scatter_html) in enumerate(scatter_html_blocks):
                plot_id = f"plot_with_{i}"
                display_style = "block" if i == 0 else "none"
                f.write(f"""
                    <div id="{plot_id}" class="scatter-plot with-china-india" style="display: {display_style};">
                        {scatter_html}
                    </div>
                """)

            # Display scatter plots WITHOUT CHN & IND (hidden initially)
            for i, (factor, scatter_html) in enumerate(scatter_html_blocks_without_chn_ind):
                plot_id = f"plot_without_{i}"
                f.write(f"""
                    <div id="{plot_id}" class="scatter-plot without-china-india" style="display: none;">
                        {scatter_html}
                    </div>
                """)

            f.write(f"""
                    </div>
                    <hr style="width:80%"/>
                    <div style="text-align: center; margin:50px;">
                        <h2>Adjusted R² Score</h2>
                        <p>
                            Train R²: <strong>{train_r2:.4f}</strong><br />
                            Test R²: <strong>{test_r2:.4f}</strong>
                        </p>
                    </div>
                    <hr style="width:80%"/>
                    <div>
                        <h2 style="text-align:center;">SHAP Beeswarm Plot</h2>
                        <img src="./beeswarm_plot.png" alt="SHAP Beeswarm Plot" style="width:80%; height:auto;">
                    </div>
                </body>
            </html>
            """)

    
if __name__ == "__main__":
    print("Calling start_predict_xgboost()...")
    start_predict_xgboost()
    print("Script execution complete!")
