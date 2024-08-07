import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import joblib

# Load and preprocess data
data = pd.read_csv('NFL Final Data.csv')

base_features = [
        "Age", "Height (in)", "Weight (lbs)",
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
        "TeamRushingTDPercentage", "TeamYardsPerPlay", "TeamPointsPerYard"
    ]

def prepare_training_data(data, position, window_size=4):
    position_data = data[data['Position'] == position]
    targets = ['Tgts', 'Rec', 'RecYds', 'RecTDs', 'RushAtt', 'RushYds', 'RushTD']

    X_list, y_list = [], []
    player_seasons = position_data.groupby('Name')

    for name, seasons in player_seasons:
        seasons = seasons.sort_values('Year', ascending=True)  # Ensure chronological order
        num_seasons = len(seasons)

        # Check if the player has any season in the age range 21 to 23
        age_21_to_23 = (seasons['Age'] >= 21) & (seasons['Age'] <= 23)
        if age_21_to_23.any():
            # Pad with -1 if there are fewer than window_size seasons
            if num_seasons < window_size:
                padding_needed = window_size - num_seasons
                padding = pd.DataFrame([[-1] * len(base_features)] * padding_needed, columns=base_features)
                seasons = pd.concat([padding, seasons], ignore_index=True)

        # Create features for each 4-year window
        for start in range(len(seasons) - window_size + 1):
            end = start + window_size

            # Extract the relevant window of seasons
            window_data = seasons.iloc[start:end]

            # Flatten the data for the current window
            X_season = []
            for i in range(window_size):
                season_data = window_data.iloc[i][base_features].values
                X_season.extend(season_data)

            # Use the last season in the window as the target
            target_season = window_data.iloc[-1][targets].values

            X_list.append(X_season)
            y_list.append(target_season)

    # Create DataFrames for features and targets
    X = pd.DataFrame(X_list, columns=[f'{feat}_year_{i + 1}' for i in range(window_size) for feat in base_features])
    y = pd.DataFrame(y_list, columns=targets)

    return X, y


def create_model():
    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42)),
        ('lgbm',
         LGBMRegressor(n_estimators=100, random_state=42, min_child_samples=10, min_split_gain=0.0, max_depth=10))
    ]

    final_estimator = xgb.XGBRegressor(n_estimators=100, random_state=42)

    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=final_estimator,
        cv=5
    )

    multi_output_regressor = MultiOutputRegressor(stacking_regressor)

    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectFromModel(xgb.XGBRegressor(n_estimators=100))),
        ('regressor', multi_output_regressor)
    ])


def optimize_model(model, X, y):
    param_space = {
        'regressor__estimator__rf__max_depth': Integer(3, 10),
        'regressor__estimator__rf__min_samples_split': Integer(2, 20),
        'regressor__estimator__xgb__max_depth': Integer(3, 10),
        'regressor__estimator__xgb__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'regressor__estimator__lgbm__num_leaves': Integer(20, 3000),
        'regressor__estimator__lgbm__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'regressor__estimator__final_estimator__max_depth': Integer(3, 10),
        'regressor__estimator__final_estimator__learning_rate': Real(0.01, 0.3, prior='log-uniform')
    }

    tscv = TimeSeriesSplit(n_splits=5)

    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=50,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42
    )

    bayes_search.fit(X, y)
    return bayes_search.best_estimator_


def train_and_save_models(data, positions, model_dir='models'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for position in positions:
        model_path = os.path.join(model_dir, f'{position}_model.pkl')
        if os.path.exists(model_path):
            print(f"Loading existing model for {position} from disk")
            models[position] = joblib.load(model_path)
        else:
            print(f"Training model for {position}")
            X, y = prepare_training_data(data, position)

            model = create_model()
            optimized_model = optimize_model(model, X, y)

            # Evaluate the model using time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            mae_scores, r2_scores = [], []

            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                optimized_model.fit(X_train, y_train)
                y_pred = optimized_model.predict(X_test)

                mae_scores.append(mean_absolute_error(y_test, y_pred, multioutput='raw_values'))
                r2_scores.append(r2_score(y_test, y_pred, multioutput='raw_values'))

            mae_avg = np.mean(mae_scores, axis=0)
            MAE_scores.append(mae_avg)
            r2_avg = np.mean(r2_scores, axis=0)
            R2_scores.append(r2_avg)

            for i, target in enumerate(y.columns):
                print(f"{target} - MAE: {mae_avg[i]:.2f}, R2: {r2_avg[i]:.2f}")

            models[position] = optimized_model
            joblib.dump(optimized_model, model_path)
            print(f"Model for {position} saved to disk")


def prepare_prediction_data(prediction_data, window_size=4):
    X_list = []

    df = prediction_data.copy()

    # Calculate derived metrics
    df['TotalYards'] = df['RushYds'] + df['RecYds']
    df['TotalTDs'] = df['RushTD'] + df['RecTDs']
    df['Catch%'] = df['Rec'] / df['Tgts']
    df['YardsPerReception'] = df['RecYds'] / df['Rec']
    df['YardsPerTarget'] = df['RecYds'] / df['Tgts']
    df['YACPerReception'] = df['YAC'] / df['Rec']
    df['TDperReception'] = df['RecTDs'] / df['Rec']
    df['TDperTarget'] = df['RecTDs'] / df['Tgts']
    df['YardsPerRush'] = df['RushYds'] / df['RushAtt']
    df['TDperRush'] = df['RushTD'] / df['RushAtt']
    df['TotalYardsPerTouch'] = (df['RushYds'] + df['RecYds']) / (df['RushAtt'] + df['Rec'])
    df['TotalTDsPerTouch'] = (df['RushTD'] + df['RecTDs']) / (df['RushAtt'] + df['Rec'])

    team_data = data[['Year', 'Team (Abbr)', 'Team Points Scored', 'Team Points Allowed', 'Team Total Yards',
                      'Team Pass Attempts', 'Team Passing Yards', 'Team PassingTDs', 'Team Rush Attempts',
                      'Team Rushing Yards', 'Team Rushing TDs', 'Team First Downs', 'Team Turnovers', 'Team RZSP',
                      'Team Offensive Efficiency', 'Team Passing Efficiency', 'Team Rushing Efficiency',
                      'Team Turnover Rate',
                      'TeamPassingYardsPerAttempt', 'TeamRushingYardsPerAttempt', 'TeamPassingTDPercentage',
                      'TeamRushingTDPercentage', 'TeamYardsPerPlay', 'TeamPointsPerYard']]

    player_seasons = pd.merge(df, team_data, on=['Year', 'Team (Abbr)'], how='left')
    num_seasons = len(player_seasons)

    # Check if the player has any season in the age range 21 to 23
    age_21_to_23 = (data['Age'] >= 21) & (data['Age'] <= 23)
    if age_21_to_23.any():
        # Pad with -1 if there are fewer than window_size seasons
        if num_seasons < window_size:
            padding_needed = window_size - num_seasons
            padding = pd.DataFrame([[-1] * len(base_features)] * padding_needed, columns=base_features)
            player_seasons = pd.concat([padding, player_seasons], ignore_index=True)

        # Grab the last 4 seasons of data
        window_data = player_seasons.iloc[-window_size:]

        # Flatten the data for the current window
        X_season = []
        for i in range(window_size):
            season_data = window_data.iloc[i][base_features].values
            X_season.extend(season_data)

        X_list.append(X_season)

    # Create DataFrame for features
    X = pd.DataFrame(X_list, columns=[f'{feat}_year_{i + 1}' for i in range(window_size) for feat in base_features])

    return X


def predict_production(player_data, position):
    model = models[position]
    prediction = model.predict(player_data)
    return pd.DataFrame(prediction, columns=['Tgts', 'Rec', 'RecYds', 'RecTDs', 'RushAtt', 'RushYds', 'RushTD'])


def calculate_fantasy_ppr(predicted_player_stats, pos):
    # Extract the relevant stats
    receptions = predicted_player_stats['Rec'].values[0]
    receiving_yards = predicted_player_stats['RecYds'].values[0]
    receiving_tds = predicted_player_stats['RecTDs'].values[0]
    rushing_yards = predicted_player_stats['RushYds'].values[0]
    rushing_tds = predicted_player_stats['RushTD'].values[0]

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
        raise ValueError(f"Invalid position: {position}")

    # Calculate fantasy points

    return points

# Main execution

MAE_scores = []
R2_scores = []

models = {}
positions = ['WR', 'RB', 'TE']
train_and_save_models(data, positions)

# print("Evaluation results per position:")
# for i, position in enumerate(positions):
#     print(f"\n{position}")
#     for j, target in enumerate(['Tgts', 'Rec', 'RecYds', 'RecTDs', 'RushAtt', 'RushYds', 'RushTD']):
#         print(f"{target} - MAE: {MAE_scores[i][j]:.2f}, R2: {R2_scores[i][j]:.2f}")

# Example usage
new_player_data = pd.DataFrame({
    'Year': [-1, -1, 2022, 2023],
    'Team (Abbr)': [-1, -1, 'TB', 'TB'],
    'Age': [-1, -1, 23, 24],
    'Height (in)': [-1, -1, 72, 72],
    'Weight (lbs)': [-1, -1, 214, 214],
    'Tgts': [-1, -1, 58, 70],
    'YAC': [-1, -1, 309, 611],
    'Rec': [-1, -1, 50, 64],
    'RecYds': [-1, -1, 290, 549],
    'RecTDs': [-1, -1, 2, 3],
    'RushAtt': [-1, -1, 129, 272],
    'RushYds': [-1, -1, 481, 990],
    'RushTD': [-1, -1, 1, 6],
    'RZ Completions': [-1, -1, 4, 7],
    'RZ Targets': [-1, -1, 6, 8],
    'RZ CompPercent': [-1, -1, 0.667, 0.8750],
    'RZ Rec TDs': [-1, -1, 1, 0],
    'RZ Rushes': [-1, -1, 11, 40],
    'RZ Rush TDs': [-1, -1, 1, 6]
})

position = 'RB'
predicted_stats = predict_production(prepare_prediction_data(new_player_data), position)

fantasy_points = calculate_fantasy_ppr(predicted_stats, position)
predicted_stats['Fantasy Points'] = fantasy_points

print(f"\nPredicted stats for the player's next season:")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(predicted_stats)
