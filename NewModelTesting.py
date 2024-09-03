from numba import jit, cuda
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, TimeSeriesSplit
from sklearn.inspection import permutation_importance
import shap
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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


def prepare_training_data(data, position, fantasy, window_size=4):
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
        # bf.remove('TeamRushingYardsPerAttempt')
        # bf.remove('TeamRushingTDPercentage')
        bf.remove('TDperRush')
        bf.remove('YardsPerRush')
        bf.remove('RZ Rushes')
        bf.remove('RZ Rush TDs')
        # bf.remove('Team Rushing Efficiency')
        targets = ['Tgts', 'Rec', 'RecYds', 'RecTDs']

    X_list, y_list = [], []
    player_seasons = position_data.groupby('Name')

    for name, seasons in player_seasons:
        seasons = seasons.sort_values('Year', ascending=True)  # Ensure chronological order
        num_seasons = len(seasons)

        # Check if the player has any season in the age range 21 to 23
        age_21_to_23 = (seasons['Age'] >= 21) & (seasons['Age'] <= 23)
        if age_21_to_23.any():
            # Pad with NaN if there are fewer than window_size seasons
            if num_seasons < window_size:
                padding_needed = window_size - num_seasons
                padding = pd.DataFrame([[np.nan] * len(bf)] * padding_needed, columns=bf)
                seasons = pd.concat([padding, seasons], ignore_index=True)

        # Create features for each 4-year window
        for start in range(len(seasons) - window_size + 1):
            end = start + window_size

            # Extract the relevant window of seasons
            window_data = seasons.iloc[start:end]

            # Flatten the data for the current window
            X_season = []
            for i in range(window_size):
                season_data = window_data.iloc[i][bf].values
                X_season.extend(season_data)

            if fantasy:
                target_season = window_data.iloc[-1][bf].values
            else:
                # Use the last season in the window as the target
                target_season = window_data.iloc[-1][targets].values

            X_list.append(X_season)
            y_list.append(target_season)


    # Create DataFrames for features and targets
    X = pd.DataFrame(X_list, columns=[f'{feat}_year_{i + 1}' for i in range(window_size) for feat in bf])

    if fantasy:
        y = pd.DataFrame(y_list, columns=bf)
    else:
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
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('scaler', StandardScaler()),
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

    tscv = TimeSeriesSplit(n_splits=4)

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


def save_metrics_to_file(position, mae_avg, r2_avg, targets, file_path='models/model_performance.txt'):
    with open(file_path, 'a') as f:
        f.write(f"\nPerformance metrics for {position}:\n")
        for i, target in enumerate(targets):
            f.write(f"{target} - MAE: {mae_avg[i]:.2f}, R2: {r2_avg[i]:.2f}\n")
        f.write("\n")


def plot_performance_metrics(positions, mae_scores, r2_scores, targets):
    target_lengths = {'WR': 4, 'TE': 4, 'RB': 7}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

    # make dictionary of position and their targets
    position_targets = {'WR': ['Tgts', 'Rec', 'RecYds', 'RecTDs'], 'TE': ['Tgts', 'Rec', 'RecYds', 'RecTDs'], 'RB': ['Tgts', 'Rec', 'RecYds', 'RecTDs', 'RushAtt', 'RushYds', 'RushTD']}

    width = 0.25

    for i, position in enumerate(positions):
        n_targets = target_lengths[position]
        x = np.arange(n_targets)

        ax1.bar(x + i * width, mae_scores[i], width, label=position)
        ax2.bar(x + i * width, r2_scores[i], width, label=position)

        # Set xticks and xticklabels
        ax1.set_xticks(x + (i - (len(positions) - 1) / 2) * width)
        ax1.set_xticklabels(position_targets[position], rotation=45, ha='right')

        ax2.set_xticks(x + (i - (len(positions) - 1) / 2) * width)
        ax2.set_xticklabels(position_targets[position], rotation=45, ha='right')

    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('MAE by Position and Target')
    ax1.legend()
    ax2.set_ylabel('R-squared')
    ax2.set_title('R-squared by Position and Target')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('models/performance_metrics.png')
    plt.close()


def plot_learning_curve(estimator, X, y, cv, scoring='neg_mean_absolute_error', position=''):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title(f"Learning Curve - {position}")
    plt.xlabel("Training examples")
    plt.ylabel("Mean Absolute Error")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig('models/' + f'learning_curve_{position}.png')
    plt.close()


def feature_importance(model, X, y, position=''):
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title(f"Top 20 Feature Importances - {position}")
    plt.tight_layout()
    plt.savefig('models/' + f'feature_importance_{position}.png')
    plt.close()


def evaluate_and_plot_shap(model, X, y, position):
    # Fit the model
    model.fit(X, y)

    # Extract the MultiOutputRegressor from the pipeline
    multi_output_regressor = model.named_steps['regressor']

    # Loop over each output model
    for output_index, output_model in enumerate(multi_output_regressor.estimators_):
        # Use the final estimator of the current StackingRegressor
        final_estimator = output_model.final_estimator_
        print(final_estimator)

        # Create a SHAP Explainer for the final estimator
        explainer = shap.Explainer(final_estimator, X)

        # Compute SHAP values for the current output
        shap_values = explainer(X)

        print(shap_values)

        # Plot SHAP summary for each target
        for target_index, target in enumerate(y.columns):
            print(f"Plotting SHAP summary for {target} ({position})")
            plt.figure(figsize=(12, 8))

            # Generate a SHAP summary plot
            shap.summary_plot(shap_values, X, plot_type="bar")

            plt.title(f'SHAP Values for {target} ({position})')
            plt.tight_layout()

            # Ensure the 'models' directory exists
            if not os.path.exists('models'):
                os.makedirs('models')

            # Save the plot
            plt.savefig(f'models/shap_summary_{position}_{target}.png')
            plt.close()

            # Optional: Plot SHAP summary for all targets combined
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X, plot_type="bar")
            plt.title(f'Overall SHAP Values ({position})')
            plt.tight_layout()
            plt.savefig('models/' + f'shap_summary_{position}_overall.png')
            plt.close()


def evaluate_model(model, X, y):
    tscv = TimeSeriesSplit(n_splits=4)
    mae_scores, r2_scores = [], []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae_scores.append(mean_absolute_error(y_test, y_pred, multioutput='raw_values'))
        r2_scores.append(r2_score(y_test, y_pred, multioutput='raw_values'))

    mae_avg = np.mean(mae_scores, axis=0)
    r2_avg = np.mean(r2_scores, axis=0)

    return mae_avg, r2_avg


def evaluate_and_plot_fantasy(model, X, y, position):
    tscv = TimeSeriesSplit(n_splits=4)
    ppr_true, ppr_pred = [], []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate the model based on PPR
        y_true_ppr = y_test['PPR'].values
        y_pred_ppr = calculate_fantasy_ppr(y_pred, position)

        ppr_true.extend(y_true_ppr)
        ppr_pred.extend(y_pred_ppr)

    ppr_mae = mean_absolute_error(ppr_true, ppr_pred)
    ppr_r2 = r2_score(ppr_true, ppr_pred)

    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot PPR
    ax.scatter(ppr_true, ppr_pred)
    ax.plot([min(ppr_true), max(ppr_true)], [min(ppr_true), max(ppr_true)], 'r--')
    ax.set_xlabel('Actual PPR')
    ax.set_ylabel('Predicted PPR')
    ax.set_title(f'{position} PPR: MAE={ppr_mae:.2f}, R2={ppr_r2:.2f}')

    plt.tight_layout()
    plt.show()

    # Save the plot
    plt.savefig(f'models/fantasy_evaluation_{position}.png')
    plt.close()

    return ppr_mae, ppr_r2


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



def train_and_save_models(data, positions, model_dir='models'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    all_mae_scores = []
    all_r2_scores = []
    target_columns = None  # Initialize target_columns

    for position in positions:
        model_path = os.path.join(model_dir, f'{position}_model.pkl')
        if os.path.exists(model_path):
            print(f"Loading existing model for {position} from disk")
            models[position] = joblib.load(model_path)
            # Load or compute target columns for this position
            _, y = prepare_training_data(data, position, False)
            if target_columns is None:
                target_columns = y.columns
        else:
            print(f"Training model for {position}")
            X, y = prepare_training_data(data, position, False)
            if target_columns is None:
                target_columns = y.columns

            model = create_model()

            optimized_model = optimize_model(model, X, y)

            mae_avg, r2_avg = evaluate_model(optimized_model, X, y)
            all_mae_scores.append(mae_avg)
            all_r2_scores.append(r2_avg)

            save_metrics_to_file(position, mae_avg, r2_avg, y.columns)

            # Plot learning curve
            plot_learning_curve(optimized_model, X, y, cv=TimeSeriesSplit(n_splits=4), position=position)

            # Feature importance
            feature_importance(optimized_model, X, y, position=position)

            # # Fantasy evaluation
            # x_fantasy, y_fantasy = prepare_training_data(data, position, True)
            # evaluate_and_plot_fantasy(optimized_model, x_fantasy, y_fantasy, position=position)

            # SHAP analysis
            # evaluate_and_plot_shap(optimized_model, X, y, position=position)

            models[position] = optimized_model
            joblib.dump(optimized_model, model_path)
            print(f"Model for {position} saved to disk")

    # Plot overall performance metrics
    if target_columns is not None:
        plot_performance_metrics(positions, all_mae_scores, all_r2_scores, target_columns)
    else:
        print("No target columns available. Unable to plot performance metrics.")


def prepare_prediction_data(prediction_data, position, window_size=4):
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


    # Check if the player has any season in the age range 21 to 23
    age_21_to_23 = (data['Age'] >= 21) & (data['Age'] <= 23)
    if age_21_to_23.any():
        # Pad with -1 if there are fewer than window_size seasons
        if num_seasons < window_size:
            padding_needed = window_size - num_seasons
            padding = pd.DataFrame([[-1] * len(bf)] * padding_needed, columns=bf)
            player_seasons = pd.concat([padding, player_seasons], ignore_index=True)

        # Grab the last 4 seasons of data
        window_data = player_seasons.iloc[-window_size:]

        # Flatten the data for the current window
        X_season = []
        for i in range(window_size):
            season_data = window_data.iloc[i][bf].values
            X_season.extend(season_data)

        X_list.append(X_season)

    # Create DataFrame for features
    X = pd.DataFrame(X_list, columns=[f'{feat}_year_{i + 1}' for i in range(window_size) for feat in bf])

    return X


def calculate_fantasy_ppr(predicted_player_stats, pos):
    # If input is a numpy array, convert to DataFrame
    if isinstance(predicted_player_stats, np.ndarray):
        columns = ['Tgts', 'Rec', 'RecYds', 'RecTDs', 'RushAtt', 'RushYds', 'RushTD']
        predicted_player_stats = pd.DataFrame(predicted_player_stats, columns=columns)

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
        raise ValueError(f"Invalid position: {pos}")

    # Calculate fantasy points

    return points


# Main execution
models = {}
positions = ['WR', 'RB', 'TE']
# train_and_save_models(data, positions)
