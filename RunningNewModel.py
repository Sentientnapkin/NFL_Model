import os
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from NewModelTesting import (prepare_prediction_data, predict_with_confidence, calculate_fantasy_ppr,
                             evaluate_and_plot_shap, prepare_training_data)
import joblib

data = pd.read_csv('NFL Final Data.csv')

models = {}
positions = ['WR', 'RB', 'TE']
for position in positions:
    model_path = os.path.join('models', f'{position}_model.pkl')
    if os.path.exists(model_path):
        print(f"Loading existing model for {position} from disk")
        models[position] = joblib.load(model_path)
    else:
        print(f"Model for {position} not found. Please train the model first.")
        exit()

# x, y = prepare_training_data(data, 'RB')
# evaluate_and_plot_shap(models['RB'], x, y, 'RB')

data_dict = {
    'Year': list(range(2020, 2024)),
    'Name': [''] * 4,
    'Position': [''] * 4,
    'Team (Abbr)': [''] * 4,
    'Age': [0] * 4,
    'Height (in)': [0] * 4,
    'Weight (lbs)': [0] * 4,
    'G': [0] * 4,
    'GS': [0] * 4,
    'Tgts': [0] * 4,
    'YAC': [0] * 4,
    'Rec': [0] * 4,
    'RecYds': [0] * 4,
    'RecTDs': [0] * 4,
    'RushAtt': [0] * 4,
    'RushYds': [0] * 4,
    'RushTD': [0] * 4,
    'RZ Completions': [0] * 4,
    'RZ Targets': [0] * 4,
    'RZ CompPercent': [0] * 4,
    'RZ Rec TDs': [0] * 4,
    'RZ Rushes': [0] * 4,
    'RZ Rush TDs': [0] * 4
}

# URL of the webpage
url = 'https://www.pro-football-reference.com/players/T/TaylJo02.htm'

player_name = ""

# Send a request to fetch the content of the webpage
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Example: Find the player's name
    player_name = soup.find('div', id='meta').find('h1').text

    player_data = soup.find('div', id='meta').findAll('p')
    player_data = [data.text for data in player_data]
    player_heightweight = player_data[2].split('\xa0')

    player_weight = player_heightweight[1].split('lb')[0]
    player_height = int(int(player_heightweight[2].removeprefix('(').removesuffix('cm,')) / 2.54)

    # Example: Find the player's stats table
    stats_table = soup.find('table', {'id': 'rushing_and_receiving'})

    # Extract table headers
    headers = [th.text for th in stats_table.find_all('th')[1:]]
    headers = headers[headers.index('Year'):headers.index('Awards') + 1]

    headers = ['Year', 'Age', 'Tm', 'Pos', 'No.', 'G', 'GS', 'Att', 'RushYds', 'RushTD', '1D', 'Succ%', 'Lng',
               'Y/A', 'Y/G', 'A/G', 'Tgt', 'Rec', 'RecYds', 'Y/R', 'RecTD', '1D', 'Succ%', 'Lng', 'R/G', 'Y/G',
               'Ctch%', 'Y/Tgt', 'Touch', 'Y/Tch', 'YScm', 'RRTD', 'Fmb', 'AV', 'Awards']

    # Extract table rows
    rows = stats_table.find_all('tr')
    stats = []
    for row in rows[2:len(rows) - 1]:  # Skip the header row
        year = row.find('th')
        year = ''.join(filter(str.isdigit, year.text))
        cols = row.find_all('td')
        cols_text = [col.text for col in cols]
        cols_text.insert(0, year)
        stats.append(cols_text)

    df1 = pd.DataFrame(stats, columns=headers)

    RZ_Data = data[['Year', 'Name', 'Position', 'RZ Completions', 'RZ Targets',
                    'RZ CompPercent', 'RZ Rec TDs', 'RZ Rushes', 'RZ Rush TDs', 'YAC']]

    # Update the dictionary with extracted stats
    for index, row in df1.iterrows():
        year = int(row['Year'])
        if year in data_dict['Year']:
            idx = data_dict['Year'].index(year)
            data_dict['Tgts'][idx] = int(row.get('Tgt', np.nan))
            data_dict['Rec'][idx] = int(row.get('Rec', np.nan))
            data_dict['RecYds'][idx] = int(row.get('RecYds', np.nan))
            data_dict['RecTDs'][idx] = int(row.get('RecTD', np.nan))
            data_dict['RushAtt'][idx] = int(row.get('Att', np.nan))
            data_dict['RushYds'][idx] = int(row.get('RushYds', np.nan))
            data_dict['RushTD'][idx] = int(row.get('RushTD', np.nan))
            data_dict['Height (in)'][idx] = player_height
            data_dict['Weight (lbs)'][idx] = int(player_weight)
            data_dict['Team (Abbr)'][idx] = row.get('Tm', np.nan)
            data_dict['Age'][idx] = int(row.get('Age', np.nan))
            data_dict['G'][idx] = int(row.get('G', np.nan))
            data_dict['GS'][idx] = int(row.get('GS', np.nan))

    for index, row in RZ_Data.iterrows():
        year = row['Year']
        if year in data_dict['Year']:
            idx = data_dict['Year'].index(year)
            data_dict['RZ Completions'][idx] = int(row['RZ Completions'])
            data_dict['RZ Targets'][idx] = int(row['RZ Targets'])
            data_dict['RZ CompPercent'][idx] = float(row['RZ CompPercent'])
            data_dict['RZ Rec TDs'][idx] = int(row['RZ Rec TDs'])
            data_dict['RZ Rushes'][idx] = int(row['RZ Rushes'])
            data_dict['RZ Rush TDs'][idx] = int(row['RZ Rush TDs'])
            data_dict['YAC'][idx] = int(row['YAC'])

    new_player_data = pd.DataFrame(data_dict)

    position = 'RB'
    X_pred = prepare_prediction_data(new_player_data, position)
    mean_pred, lower_bound, upper_bound = predict_with_confidence(models[position], X_pred, position)

    fantasy_points_mean = calculate_fantasy_ppr(mean_pred, position)
    mean_pred['Fantasy Points'] = fantasy_points_mean

    fantasy_points_lower = calculate_fantasy_ppr(lower_bound, position)
    lower_bound['Fantasy Points'] = fantasy_points_lower

    fantasy_points_upper = calculate_fantasy_ppr(upper_bound, position)
    upper_bound['Fantasy Points'] = fantasy_points_upper

    print(f"\nPredicted stats for {player_name}'s next season (with 95% confidence intervals):")
    for i, target in enumerate(['Tgts', 'Rec', 'RecYds', 'RecTDs', 'RushAtt', 'RushYds', 'RushTD']):
        print(
            f"{target}: {mean_pred[target].values[0]:.4f} ({lower_bound[target].values[0]:.4f}-{upper_bound[target].values[0]:.4f})")

    print(f"Fantasy Points: {fantasy_points_mean:.2f} ({fantasy_points_lower:.4f}-{fantasy_points_upper:.4f})")

else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")















'''
# Path to your chromedriver
DRIVER_PATH = 'C:\\Users\\Sebastian\\Downloads\\chromedriver-win64\\chromedriver.exe'

# Set up Chrome options if needed
chrome_options = webdriver.ChromeOptions()

# Set up the ChromeDriver service
service = Service(DRIVER_PATH)

# Initialize WebDriver with the driver path specified correctly
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL of the webpage with dropdown
url = 'https://www.pro-football-reference.com/years/2023/redzone-rushing.htm'
driver.get(url)

# Wait until the dropdown is present
wait = WebDriverWait(driver, 10)

# Close the WebDriver
driver.quit()



new_player_data = pd.DataFrame({
    'Year': [np.nan, np.nan, 2022, 2023],
    'Team (Abbr)': [np.nan, np.nan, 'TB', 'TB'],
    'Age': [np.nan, np.nan, 23, 24],
    'Height (in)': [np.nan, np.nan, 72, 72],
    'Weight (lbs)': [np.nan, np.nan, 214, 214],
    'Tgts': [np.nan, np.nan, 58, 70],
    'YAC': [np.nan, np.nan, 309, 611],
    'Rec': [np.nan, np.nan, 50, 64],
    'RecYds': [np.nan, np.nan, 290, 549],
    'RecTDs': [np.nan, np.nan, 2, 3],
    'RushAtt': [np.nan, np.nan, 129, 272],
    'RushYds': [np.nan, np.nan, 481, 990],
    'RushTD': [np.nan, np.nan, 1, 6],
    'RZ Completions': [np.nan, np.nan, 4, 7],
    'RZ Targets': [np.nan, np.nan, 6, 8],
    'RZ CompPercent': [np.nan, np.nan, 0.667, 0.8750],
    'RZ Rec TDs': [np.nan, np.nan, 1, 0],
    'RZ Rushes': [np.nan, np.nan, 11, 40],
    'RZ Rush TDs': [np.nan, np.nan, 1, 6]
})
'''
