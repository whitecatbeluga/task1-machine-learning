import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import random
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# General Overview:
# Air pollution has always been prominent until to this day. Throughout the countries, every year, it has been
# rising and health of the people are compromised. In this exam, we will be taking a look on the factors
# that causes air pollution deaths around the world. We will be using the capabilities of machine learning,
# specifically the use of Extreme Gradient Boosting to discover and predict the future data and statistics
# regarding air pollutions deaths. In this exam, you will be tested by, first, to get the
# correlation of air pollution deaths vs the different environment and socio-economic factors. Second, is to
# predict, using XGBoost (Extreme Gradient Boosting), the number of air pollution deaths in the year 2018 and
# get the best possible R^2 Score. Third, you are tasked to create an HTML/CSS Application that shows the results
# of the prediction and correlation. And finally, explain the results through graphs (SHAP, and Scatter Plot).
# There will be specific instructions inside the functions to guide you throughout the exam.

# NOTE: There will be an HTML file in this exam where you can see the expected results of the exam.
#       You can use that as your reference.

# NOTE: You are free to create multiple functions for readability and organization of your code.
# NOTE: You are also free to look up different documentations for the libraries you will be using.


# This function defines all the files as a dictionary that will be used throughout the exam.
# This csv files contains your data in the different environment and socioeconomic factors
# These files are located at `data/...`
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


# This function should contain your code in getting the R^2 Score using xgboost.
def start_predict_xgboost():

    # Set seed to 42 for consistent results
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Define Files
    files = define_files()


    # These are the the list of environment factors.
    environment_factor_files_list = [
        'transportation',
        'coal',
        'cropland',
        'residential_commercial',
        'forest_clearing',
        'petrochemicals',
        'electricity_generation',
        'incineration_open_burning'
    ]
    # These are the list of socio-economic factors
    socioeconomic_files_list = [
        'health_expenditure',
        'urban_population'
    ]

    # 1. Construct a code that will retrieve the common country codes inside the both csv factor categories.
    # Since there are two factor files categories, combine them and the results should only return a list
    # of common country codes and should be sorted.

    #Your code below:

    #Your code above

    # 2. For our air pollution data, group the data by country code and
    #    filter the data by only retrieving the data that are:
    #   A. Present in the common country codes.
    #   B. AND in the period 2018
    #   C. AND Both sexes
    air_pollution_df = pd.read_csv(files["air_pollution_death"])
    #Your code below:

    #Your code above:

    # 3. With the use of common_country_codes
    #   A. We would like to create a separate list that consists environment_data and socioeconomc_data
    #   B. (For environment_data only) Iterate to each data and filter them by common country codes and
    #       select data starting from 2018-01-01 00:00:00. Aggregate this data by grouping it by country code and
    #       getting their respective total emissions_quantity. Rename the common country code column
    #       to Country Code and emissions_quantity to 'name'.
    #   C. (For socioeconomic_data) Iterate to each data and filter them by common country codes and
    #       select data in 2018 column. Rename the common country code column
    #       to Country Code and the column '2018' to 'name'.

    # Your code below:

    # You code above:


    # 4. With the use of the environment_data and socioeconomic_data
    #   A. Merge the environment_data and socioeconomic_data on common country codes
    #   B. Do another merge with the air pollution deaths data on common country codes.
    #   C. Create a dataframe using this merged data of A and B.

    # Your code below:

    # You code above:


    # 5. For our final dataframe
    #   A. Select the columns that have > 0.5 missing values
    #   B. Drop the drops missing values
    #   C. Based from the example output, there should be an option where CHN and IND are excluded

    # Your code below:

    # You code above:

    # 6. Determine our X and y values
    #   A. X should be columns other than common country codes and air pollution death
    #   B. y should be the 'air_pollution_death' values
    # Your code below:

    X = ""
    y = ""

    # In this exam, we shall use 80% as our train data and 20% as our test data
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    # 7. For standardization of dataset, a common requirement is to scale the data.
    #   A. Scale the X_train_val and X_test
    scaler = StandardScaler()
    # Your code below:

    # You code above:

    # 8. Defining our XGB Model with hyperparameters
    #   A. Try to get the best R^2 score as much as possible by tweaking the hyperparameters
    # NOTE: There are pre defined hyperparameters, try not to remove them in tuning for
    #       the best possible R^2 Score.

    xgb_model = xgb.XGBRegressor(
        # Your additional hyperparameters below:

        # Your additional hyperparameters above:
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_seed,
    )

    # 9. With the xgb_model.
    #   A. Generate the code to train the model
    #   B. Generate the code to evaluate and predict the Test and Train data
    #   C. Print the results to the HTML File

    test_r2 = ""
    train_r2 = ""


# 10. With the use of the results of your prediction, we should be able to explain the data that we got.
# Using SHAP's bee swarm plot, show the value or contribution of each feature to the prediction.
def generate_beeswarm_plot():
    shap_fig = plt.figure()
    shap_fig.savefig("beeswarm_plot.png", bbox_inches='tight')


# This function should contain the generation of your HTML File. Use the coding_test.html as your reference
# This function should also contain the generation of scatter plot and choropleth map
def generate_html_file():

    # 11. For the scatter plot:
    #   A. Show the scatter plot of the different countries' Air Pollution Deaths vs different factors (Refer to the HTML File)
    #   B. Show the correlation between the Air Pollution Deaths vs different factors (Refer to the HTML file)
    #   C. There should be an option to remove outliers (CHN and IND) or not in showing the graph.
    # NOTE: You can visit plotly for the scatter plot documentation
    scatter_html_blocks = []

    # Your code below:

    # You code above:

    # 12. For the choropleth map
    #   A. Show the map with the Air pollution deaths in every country (Refer to the HTML File)
    #   B. When a country is hovered in the map, show the Country Code and Air Pollution Deaths only.
    # NOTE: You can visit plotly for the choropleth map documentation
    choropleth_map = ""

    # Your code below:

    # Your code above:

    # HTML Creation starts here:
    with open("coding_test_output.html", "w", encoding="utf-8") as f:
        f.write("""
        <html>
            <head>
                <title>Air Pollution and Emissions Analysis</title>
                </script>
            </head>
            <body>
                <h1>Air Pollution Deaths by Country</h1>
            </body>
        </html>
        """)

