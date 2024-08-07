import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import joblib

# Define base features
base_features = [
    'Age', 'Height (in)', 'Weight (lbs)', 'Tgts', 'Rec', 'RecYds', 'RecTDs',
    'RushAtt', 'RushYds', 'RushTD', 'YAC', 'TotalYards', 'TotalTDs',
    'Team Points Scored', 'Team Total Yards', 'Team Pass Attempts',
    'Team Passing Yards', 'Team PassingTDs', 'Team Rush Attempts',
    'Team Rushing Yards', 'Team Rushing TDs', 'Team First Downs',
    'Team Turnovers', 'Team RZSP', 'Team Offensive Efficiency',
    'Team Passing Efficiency', 'Team Rushing Efficiency', 'Team Turnover Rate',
    'Catch%', 'YardsPerReception', 'YardsPerTarget', 'YACPerReception',
    'TDperReception', 'TDperTarget', 'YardsPerRush', 'TDperRush',
    'TotalYardsPerTouch', 'TotalTDsPerTouch', 'TeamPassingYardsPerAttempt',
    'TeamRushingYardsPerAttempt', 'TeamPassingTDPercentage',
    'TeamRushingTDPercentage', 'TeamYardsPerPlay', 'TeamPointsPerYard',
    'RZ Completions', 'RZ Targets', 'RZ CompPercent', 'RZ Rec TDs',
    'RZ Rushes', 'RZ Rush TDs'
]


def prepare_training_data(data, position, window_size=4):
    position_data = data[data['Position'] == position]
    targets = ['Tgts', 'Rec', 'RecYds', 'RecTDs', 'RushAtt', 'RushYds', 'RushTD']

    bf = base_features.copy()

    if position == 'WR' or position == 'TE':
        bf.remove('RushAtt')
        bf.remove('RushYds')
        bf.remove('RushTD')
        bf.remove('Team Rush Attempts')
        bf.remove('Team Rushing Yards')
        bf.remove('Team Rushing TDs')
        bf.remove('TeamRushingYardsPerAttempt')
        bf.remove('TeamRushingTDPercentage')
        bf.remove('TDperRush')
        bf.remove('YardsPerRush')
        bf.remove('RZ Rushes')
        bf.remove('RZ Rush TDs')
        bf.remove('Team Rushing Efficiency')
        targets = ['Tgts', 'Rec', 'RecYds', 'RecTDs']

    X_list, y_list = [], []
    player_seasons = position_data.groupby('Name')

    for name, seasons in player_seasons:
        seasons = seasons.sort_values('Year', ascending=True)
        num_seasons = len(seasons)

        age_21_to_23 = (seasons['Age'] >= 21) & (seasons['Age'] <= 23)
        if age_21_to_23.any():
            if num_seasons < window_size:
                padding_needed = window_size - num_seasons
                padding = pd.DataFrame([[np.nan] * len(bf)] * padding_needed, columns=bf)
                seasons = pd.concat([padding, seasons], ignore_index=True)

        for start in range(len(seasons) - window_size + 1):
            end = start + window_size

            window_data = seasons.iloc[start:end]

            X_season = []
            for i in range(window_size):
                season_data = window_data.iloc[i][bf].values
                X_season.extend(season_data)

            target_season = window_data.iloc[-1][targets].values

            X_list.append(X_season)
            y_list.append(target_season)

    X = pd.DataFrame(X_list, columns=[f'{feat}_year_{i + 1}' for i in range(window_size) for feat in bf])
    y = pd.DataFrame(y_list, columns=targets)

    return X, y


def create_and_train_model(X_train, y_train):

    model1 = LinearRegression()
    model2 = RandomForestRegressor(n_estimators=100, random_state=42)
    model3 = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, early_stopping=True, random_state=42)

    ensemble = VotingRegressor([
        ('lr', model1),
        ('rf', model2),
        ('nn', model3)
    ])

    # Use MultiOutputRegressor to handle multiple output targets
    multi_output_ensemble = MultiOutputRegressor(ensemble)
    multi_output_ensemble.fit(X_train, y_train)
    return multi_output_ensemble


def evaluate_model(model, X_test, y_test, targets):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')

    for i, target in enumerate(targets):
        print(f"{target}:")
        print(f"  Mean Squared Error: {mse[i]}")
        print(f"  R-squared Score: {r2[i]}")


def print_feature_importance(model, feature_names):
    rf_model = model.estimators_[0].named_estimators_['rf']
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 most important features:")
    print(feature_importance.head(10))


def train_models():
    # Load the data
    data = pd.read_csv('NFL Final Data.csv')

    positions = ['RB', 'WR', 'TE']
    models = {}

    for position in positions:
        print(f"\nTraining model for {position}")
        X, y = prepare_training_data(data, position=position, window_size=4)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        imputer = SimpleImputer(strategy='constant', fill_value=-1)
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)

        X_test_scaled = scaler.transform(X_test_imputed)

        model = create_and_train_model(X_train_scaled, y_train)
        models[position] = model

        print(f"\nEvaluation for {position} model:")
        evaluate_model(model, X_test_scaled, y_test, y.columns)
        print_feature_importance(model, X.columns)

        # Save model
        joblib.dump(model, 'ensemble_models_1/' + f'{position}_model.joblib')


def predict_player_production(new_player_data, position, models, window_size=4):
    data = pd.read_csv('NFL Final Data.csv')

    # Calculate additional features based on the provided data
    new_player_data['TotalYards'] = new_player_data['RecYds'] + new_player_data['RushYds']
    new_player_data['TotalTDs'] = new_player_data['RecTDs'] + new_player_data['RushTD']
    new_player_data['Catch%'] = new_player_data['Rec'] / new_player_data['Tgts']
    new_player_data['YardsPerReception'] = new_player_data['RecYds'] / new_player_data['Rec']
    new_player_data['YardsPerTarget'] = new_player_data['RecYds'] / new_player_data['Tgts']
    new_player_data['YACPerReception'] = new_player_data['YAC'] / new_player_data['Rec']
    new_player_data['TDperReception'] = new_player_data['RecTDs'] / new_player_data['Rec']
    new_player_data['TDperTarget'] = new_player_data['RecTDs'] / new_player_data['Tgts']
    new_player_data['YardsPerRush'] = new_player_data['RushYds'] / new_player_data['RushAtt']
    new_player_data['TDperRush'] = new_player_data['RushTD'] / new_player_data['RushAtt']
    new_player_data['TotalYardsPerTouch'] = new_player_data['TotalYards'] / (
            new_player_data['Rec'] + new_player_data['RushAtt'])
    new_player_data['TotalTDsPerTouch'] = new_player_data['TotalTDs'] / (
            new_player_data['Rec'] + new_player_data['RushAtt'])

    team_data = data[['Year', 'Team (Abbr)', 'Team Points Scored', 'Team Points Allowed', 'Team Total Yards',
                      'Team Pass Attempts', 'Team Passing Yards', 'Team PassingTDs', 'Team Rush Attempts',
                      'Team Rushing Yards', 'Team Rushing TDs', 'Team First Downs', 'Team Turnovers', 'Team RZSP',
                      'Team Offensive Efficiency', 'Team Passing Efficiency', 'Team Rushing Efficiency',
                      'Team Turnover Rate',
                      'TeamPassingYardsPerAttempt', 'TeamRushingYardsPerAttempt', 'TeamPassingTDPercentage',
                      'TeamRushingTDPercentage', 'TeamYardsPerPlay', 'TeamPointsPerYard']]

    player_seasons = pd.merge(new_player_data, team_data, on=['Year', 'Team (Abbr)'], how='left')

    # Select features based on position
    bf = base_features.copy()

    if position == 'WR' or position == 'TE':
        bf.remove('RushAtt')
        bf.remove('RushYds')
        bf.remove('RushTD')
        bf.remove('Team Rush Attempts')
        bf.remove('Team Rushing Yards')
        bf.remove('Team Rushing TDs')
        bf.remove('TeamRushingYardsPerAttempt')
        bf.remove('TeamRushingTDPercentage')
        bf.remove('TDperRush')
        bf.remove('YardsPerRush')
        bf.remove('RZ Rushes')
        bf.remove('RZ Rush TDs')
        bf.remove('Team Rushing Efficiency')

    # Prepare the input data
    X = []
    for i in range(window_size):
        season_data = player_seasons.iloc[i][bf].values
        X.extend(season_data)

    X = np.array(X).reshape(1, -1)

    # Impute missing values
    imputer = SimpleImputer(strategy='constant', fill_value=-1)
    X_imputed = imputer.fit_transform(X)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Make prediction
    model = models[position]

    prediction = model.predict(X_scaled)

    return prediction[0]


if __name__ == "__main__":
    # Uncomment the following line to train and save models
    train_models()

    # Load saved models
    models = {
        'RB': joblib.load('ensemble_models_1/RB_model.joblib'),
        'WR': joblib.load('ensemble_models_1/WR_model.joblib'),
        'TE': joblib.load('ensemble_models_1/TE_model.joblib')
    }

    # Example usage with new player data
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

    # Determine the position based on the data (in this case, it looks like an RB)
    position = 'RB'

    # Make prediction
    prediction = predict_player_production(new_player_data, position, models)

    # Print the prediction
    print(f"Predicted production for {position}:")
    if position == 'RB':
        print(f"RushAtt: {prediction[0]:.2f}")
        print(f"RushYds: {prediction[1]:.2f}")
        print(f"RushTD: {prediction[2]:.2f}")
        print(f"Rec: {prediction[3]:.2f}")
        print(f"RecYds: {prediction[4]:.2f}")
        print(f"RecTDs: {prediction[5]:.2f}")
    else:  # WR or TE
        print(f"Tgts: {prediction[0]:.2f}")
        print(f"Rec: {prediction[1]:.2f}")
        print(f"RecYds: {prediction[2]:.2f}")
        print(f"RecTDs: {prediction[3]:.2f}")
