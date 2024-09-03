import os
import joblib
import pandas as pd
import numpy as np

data = pd.read_csv('NFL Final Data.csv')

all_features = [
    "Age", "Height (in)", "Weight (lbs)", "G", "GS",
    "RushYds", "RushAtt", "RushTD", "Rec", "RecYds", "RecTDs", "Tgts",
    "YAC", "TotalYards", "TotalTDs", "RZ Completions", "RZ Targets",
    "RZ CompPercent", "RZ Rec TDs", "RZ Rushes", "RZ Rush TDs",
    "Team Points Scored", "Team Points Allowed", "Team Total Yards",
    "Team Pass Attempts", "Team Passing Yards", "Team PassingTDs",
    "Team Rush Attempts", "Team Rushing Yards", "Team Rushing TDs",
    "Team First Downs", "Team Turnovers", "Team RZSP",
    "Team Offensive Efficiency", "Team Passing Efficiency",
    "Team Rushing Efficiency", "Team Turnover Rate", "Catch%",
    "YardsPerReception", "YardsPerTarget", "YACPerReception",
    "TDperReception", "TDperTarget", "YardsPerRush", "TDperRush",
    "TotalYardsPerTouch", "TotalTDsPerTouch", "TeamPassingYardsPerAttempt",
    "TeamRushingYardsPerAttempt", "TeamPassingTDPercentage",
    "TeamRushingTDPercentage", "TeamYardsPerPlay", "TeamPointsPerYard",
    "PPR", "PosRank"
]

base_features = [
    "Name", "Position",
    "Age", "Height (in)", "Weight (lbs)", "G", "GS",
    "RushYds", "RushAtt", "RushTD", "Rec", "RecYds", "RecTDs", "Tgts",
    "YAC", "RZ Completions", "RZ Targets",
    "RZ CompPercent", "RZ Rec TDs", "RZ Rushes", "RZ Rush TDs",
    "Team Pass Attempts", "Team Passing Yards", "Team PassingTDs",
    "Team Rush Attempts", "Team Rushing Yards", "Team Rushing TDs",
    "Team First Downs", "Team RZSP",
    "Team Passing Efficiency",
    "Team Rushing Efficiency", "Catch%",
    "YardsPerReception", "YardsPerTarget", "YACPerReception",
    "TDperReception", "TDperTarget", "YardsPerRush", "TDperRush",
    "TotalYardsPerTouch", "TotalTDsPerTouch"
]


def calculate_fantasy_ppr(predicted_player_stats, pos):
    # If input is a numpy array, convert to DataFrame
    if isinstance(predicted_player_stats, np.ndarray):
        columns = ['Tgts', 'Rec', 'RecYds', 'RecTDs', 'RushAtt', 'RushYds', 'RushTD']
        predicted_player_stats = pd.DataFrame(predicted_player_stats, columns=columns)

    # Extract the relevant stats
    receptions = predicted_player_stats['Rec'].values[0]
    receiving_yards = predicted_player_stats['RecYds'].values[0]
    receiving_tds = predicted_player_stats['RecTDs'].values[0]
    if pos == 'RB':
        rushing_yards = predicted_player_stats['RushYds'].values[0]
        rushing_tds = predicted_player_stats['RushTD'].values[0]
    else:
        rushing_yards = 0
        rushing_tds = 0

    if pos == 'WR':
        points = (
                receptions * 1.0 +  # 1 point per reception (PPR)
                receiving_yards * 0.1 +  # 0.1 points per receiving yard
                receiving_tds * 6.0  # 6 points per receiving touchdown
        )

    elif pos == 'RB':
        points = (
                receptions * 1.0 +  # 1 point per reception (PPR)
                receiving_yards * 0.1 +  # 0.1 points per receiving yard
                receiving_tds * 6.0 +  # 6 points per receiving touchdown
                rushing_yards * 0.1 +  # 0.1 points per rushing yard
                rushing_tds * 6.0  # 6 points per rushing touchdown
        )

    elif pos == 'TE':
        points = (
                receptions * 1.0 +  # 1 point per reception (PPR)
                receiving_yards * 0.1 +  # 0.1 points per receiving yard
                receiving_tds * 6.0  # 6 points per receiving touchdown
        )

    else:
        raise ValueError(f"Invalid position: {pos}")

    # Calculate fantasy points

    return points



def prepare_prediction_data(prediction_data, position, window_size=4):
    X_list = []

    player_seasons = prediction_data.copy()
    num_seasons = len(player_seasons)

    bf = base_features.copy()

    if position == 'WR' or position == 'TE':
        bf.remove('RushAtt')
        bf.remove('RushYds')
        bf.remove('RushTD')
        bf.remove('Team Rush Attempts')
        bf.remove('Team Rushing Yards')
        bf.remove('Team Rushing TDs')
        # bf.remove('TeamRushingYardsPerAttempt')
        # bf.remove('TeamRushingTDPercentage')
        bf.remove('TDperRush')
        bf.remove('YardsPerRush')
        bf.remove('RZ Rushes')
        bf.remove('RZ Rush TDs')
        # bf.remove('Team Rushing Efficiency')

    player_seasons = player_seasons.groupby('Name')

    for name, seasons in player_seasons:
        seasons = seasons.sort_values('Year', ascending=True)  # Ensure chronological order
        num_seasons = len(seasons)

        # Check if the player has a season in 2023
        if 2023 in seasons['Year'].values:
            # Check if the player has any season in the age range 21 to 23
            age_21_to_23 = (data['Age'] >= 21) & (data['Age'] <= 23)
            if age_21_to_23.any():
                # Pad with -1 if there are fewer than window_size seasons
                if num_seasons < window_size:
                    padding_needed = window_size - num_seasons
                    padding = pd.DataFrame([[-1] * len(bf)] * padding_needed, columns=bf)
                    seasons = pd.concat([padding, seasons[bf]], ignore_index=True)
                else:
                    seasons = seasons[bf]

                # Grab the last 4 seasons of data
                window_data = seasons.tail(window_size)

                # Flatten the data for the current window
                X_season = []
                for i in range(window_size):
                    season_data = window_data.iloc[i][bf].values
                    X_season.extend(season_data)

                X_list.append(X_season)

    # Create DataFrame for features
    X = pd.DataFrame(X_list, columns=[f'{feat}_year_{i + 1}' for i in range(window_size) for feat in bf])

    return X


def predict_with_confidence(model, X, position, n_iterations=1000):
    predictions = []
    for _ in range(n_iterations):
        y_pred = model.predict(X)
        predictions.append(y_pred)

    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions, axis=0)
    lower_bound = np.percentile(predictions, 2.5, axis=0)
    upper_bound = np.percentile(predictions, 97.5, axis=0)

    if position == 'WR' or position == 'TE':
        columns = ['Tgts', 'Rec', 'RecYds', 'RecTDs']
    elif position == 'RB':
        columns = ['Tgts', 'Rec', 'RecYds', 'RecTDs', 'RushAtt', 'RushYds', 'RushTD']
    else:
        raise ValueError(f"Invalid position: {position}")

    return (pd.DataFrame(mean_prediction, columns=columns),
            pd.DataFrame(lower_bound, columns=columns),
            pd.DataFrame(upper_bound, columns=columns))


pred_data = prepare_prediction_data(data, 'RB')

predictions_df = pd.DataFrame(columns=['Player', 'Tgts', 'Rec', 'RecYds', 'RecTDs', 'RushAtt', 'RushYds',
                                       'RushTD', 'Fantasy Points'])
positions = ['RB', 'WR', 'TE']
model_dir = 'models'
models = {}

for position in positions:
    model_path = os.path.join(model_dir, f'{position}_model.pkl')
    if os.path.exists(model_path):
        print(f"Loading existing model for {position} from disk")
        models[position] = joblib.load(model_path)

# grab each chunk of 4 in pred_data
for i in range(0, len(pred_data)):
    player_data = pred_data.iloc[i]
    player_name = player_data['Name_year_1']
    player_position = player_data['Position_year_1']

    for year in range(1, 5):
        player_data = player_data.drop([f'Name_year_{year}', f'Position_year_{year}'])
        if player_position == 'WR' or player_position == 'TE':
            player_data = player_data.drop([f'RushAtt_year_{year}', f'RushYds_year_{year}', f'RushTD_year_{year}',
                                            f'Team Rush Attempts_year_{year}', f'Team Rushing Yards_year_{year}',
                                            f'Team Rushing TDs_year_{year}', f'TDperRush_year_{year}',
                                            f'YardsPerRush_year_{year}', f'RZ Rushes_year_{year}',
                                            f'RZ Rush TDs_year_{year}'])
    X_pred = player_data.to_frame().T

    mean_pred, lower_bound, upper_bound = predict_with_confidence(models[player_position], X_pred, player_position)

    fantasy_points_mean = calculate_fantasy_ppr(mean_pred, player_position)
    mean_pred['Fantasy Points'] = fantasy_points_mean

    fantasy_points_lower = calculate_fantasy_ppr(lower_bound, player_position)
    lower_bound['Fantasy Points'] = fantasy_points_lower

    fantasy_points_upper = calculate_fantasy_ppr(upper_bound, player_position)
    upper_bound['Fantasy Points'] = fantasy_points_upper

    if player_position == 'WR' or player_position == 'TE':
        new_row = pd.DataFrame({
            'Player': [player_name],
            'Tgts': [mean_pred['Tgts'].values[0]],
            'Rec': [mean_pred['Rec'].values[0]],
            'RecYds': [mean_pred['RecYds'].values[0]],
            'RecTDs': [mean_pred['RecTDs'].values[0]],
            'RushAtt': [0],
            'RushYds': [0],
            'RushTD': [0],
            'Fantasy Points': [fantasy_points_mean]
        })
    else:
        new_row = pd.DataFrame({
            'Player': [player_name],
            'Tgts': [mean_pred['Tgts'].values[0]],
            'Rec': [mean_pred['Rec'].values[0]],
            'RecYds': [mean_pred['RecYds'].values[0]],
            'RecTDs': [mean_pred['RecTDs'].values[0]],
            'RushAtt': [mean_pred['RushAtt'].values[0]],
            'RushYds': [mean_pred['RushYds'].values[0]],
            'RushTD': [mean_pred['RushTD'].values[0]],
            'Fantasy Points': [fantasy_points_mean]
        })

    predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)

    print("\nAdded data for player: ", player_name)

predictions_df.to_csv('2024_Predicted_Stats.csv', index=False)

