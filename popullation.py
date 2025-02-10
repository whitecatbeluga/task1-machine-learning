import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import random
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def define_files():
    files = {
        "air_pollution_death": "data/air_pollution_death.csv",
        "transportation": "data/road-transportation_country_emissions.csv",
        "coal": "data/coal-mining_country_emissions.csv",
        "cropland": "data/cropland-fires_country_emissions.csv",
        "residential_commercial": "data/residential-and-commercial-onsite-fuel-usage_country_emissions.csv",
        "forest_clearing": "data/forest-land-clearing_country_emissions.csv",
        "petrochemicals": "data/petrochemicals_country_emissions.csv",
        "electricity_generation": "data/electricity-generation_country_emissions.csv",
        "incineration_open_burning": "data/incineration-and-open-burning-of-waste_country_emissions.csv",
        "health_expenditure": "data/health-expenditure.csv",
        "urban_population": "data/urban-population.csv"
    }
    return files

# Set seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

# Load datasets with adjusted parameters for health_expenditure
def load_data():
    files = define_files()
    air_pollution_death = pd.read_csv(files['air_pollution_death'])
    transportation = pd.read_csv(files['transportation'])
    
    # Use skiprows=3 to properly load the header row from the health_expenditure file.
    health_expenditure = pd.read_csv(
        files['health_expenditure'], 
        skiprows=3, 
        engine='python',
        sep=',',
        quotechar='"',
        skipinitialspace=True,
        on_bad_lines='skip'
    )
    # Clean column names: remove extra quotes and spaces.
    health_expenditure.columns = health_expenditure.columns.str.replace('"','').str.strip()
    return air_pollution_death, transportation, health_expenditure

# Data preprocessing
def preprocess_data(air_pollution_death, transportation, health_expenditure):
    # Clean column names for all datasets.
    air_pollution_death.columns = air_pollution_death.columns.str.strip()
    transportation.columns = transportation.columns.str.strip()
    health_expenditure.columns = health_expenditure.columns.str.strip()
    
    # Debug: print columns from health_expenditure for inspection.
    print("Health Expenditure Columns:", health_expenditure.columns.tolist())
    
    # Search for the column that exactly equals "Country Code".
    country_code_cols = [col for col in health_expenditure.columns if col == "Country Code"]
    if not country_code_cols:
        print("Available columns:", health_expenditure.columns.tolist())
        raise ValueError("Column 'Country Code' not found in health_expenditure.")
    country_code_col = country_code_cols[0]
    
    # Determine common countries using:
    # - air_pollution_death: "SpatialDimValueCode"
    # - transportation: "iso3_country"
    # - health_expenditure: the found country_code_col
    common_countries = set(air_pollution_death['SpatialDimValueCode']).intersection(
        transportation['iso3_country'], health_expenditure[country_code_col]
    )
    
    # For air_pollution_death, filter using "Period" for target year.
    # We'll try 2018 first; if no rows, then use 2019.
    air_pollution_death['Period'] = air_pollution_death['Period'].astype(str).str.strip()
    target_year = "2018"
    if air_pollution_death[air_pollution_death['Period'] == target_year].empty:
        target_year = "2019"
        print("No data for 2018 found, using 2019 instead.")
    
    # Rename FactValueNumeric to "Air Pollution Deaths"
    air_pollution_death = air_pollution_death[
        air_pollution_death['SpatialDimValueCode'].isin(common_countries) & 
        (air_pollution_death['Period'] == target_year)
    ].rename(columns={"FactValueNumeric": "Air Pollution Deaths"})
    
    # For transportation, filter by target_year if a "Year" column exists.
    if 'Year' in transportation.columns:
        transportation = transportation[
            transportation['iso3_country'].isin(common_countries) & 
            (transportation['Year'] == int(target_year))
        ]
    else:
        transportation = transportation[transportation['iso3_country'].isin(common_countries)]
    
    # For health_expenditure, select only the country code and the "2018" column,
    # renaming "2018" to "Health Expenditure"
    if '2018' not in health_expenditure.columns:
        print("Available columns in health_expenditure:", health_expenditure.columns.tolist())
        raise ValueError("Column '2018' not found in health_expenditure.")
    health_expenditure = health_expenditure[[country_code_col, '2018']].rename(columns={'2018': 'Health Expenditure'})
    health_expenditure = health_expenditure[health_expenditure[country_code_col].isin(common_countries)]
    
    # Merge the datasets on the common country code columns.
    df = air_pollution_death.merge(transportation, left_on='SpatialDimValueCode', right_on='iso3_country') \
                             .merge(health_expenditure, left_on='SpatialDimValueCode', right_on=country_code_col)
    
    df = df.dropna(axis=1, thresh=int(0.5 * len(df)))
    return df

# Model training and evaluation
def train_model(df):
    # Drop the key columns and the target column, then select only numeric features.
    X = df.drop(['SpatialDimValueCode', 'Air Pollution Deaths'], axis=1)
    X = X.select_dtypes(include=[np.number])
    y = df['Air Pollution Deaths'].astype(float)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val)
    X_test = scaler.transform(X_test)

    xgb_model = xgb.XGBRegressor(
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_seed,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )
   
    xgb_model.fit(X_train_val, y_train_val)
    y_pred_train = xgb_model.predict(X_train_val)
    y_pred_test = xgb_model.predict(X_test)

    train_r2 = r2_score(y_train_val, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    print(f'Train R^2 Score: {train_r2}')
    print(f'Test R^2 Score: {test_r2}')

    return xgb_model, X_train_val, X_test, y_train_val, y_test

# SHAP beeswarm plot
def generate_beeswarm_plot(model, X_train_val):
    explainer = shap.Explainer(model, feature_perturbation="interventional")
    shap_values = explainer(X_train_val)
    shap.summary_plot(shap_values, X_train_val, plot_type="bee swarm")
    plt.savefig("beeswarm_plot.png", bbox_inches='tight')

# HTML generation
# def generate_html(df):
#     fig = px.scatter(df, x='Transportation Emissions', y='Air Pollution Deaths', color='SpatialDimValueCode')
#     fig.write_html('scatter_plot.html')
#     fig_choropleth = px.choropleth(df, locations='SpatialDimValueCode', color='Air Pollution Deaths',
#                                    hover_name='SpatialDimValueCode', hover_data=['Air Pollution Deaths'])
#     fig_choropleth.write_html('choropleth_map.html')

# Main execution
if __name__ == "__main__":
    air_pollution_death, transportation, health_expenditure = load_data()
    df = preprocess_data(air_pollution_death, transportation, health_expenditure)
    model, X_train_val, X_test, y_train_val, y_test = train_model(df)
    generate_beeswarm_plot(model, X_train_val)
    # generate_html(df)
