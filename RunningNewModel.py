import os
import pandas as pd
import numpy as np
from NewModelTesting import prepare_prediction_data, predict_with_confidence, calculate_fantasy_ppr
import joblib


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

# Example usage with confidence intervals
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

position = 'RB'
X_pred = prepare_prediction_data(new_player_data, position)
mean_pred, lower_bound, upper_bound = predict_with_confidence(models[position], X_pred, position)

fantasy_points_mean = calculate_fantasy_ppr(mean_pred, position)
mean_pred['Fantasy Points'] = fantasy_points_mean

fantasy_points_lower = calculate_fantasy_ppr(lower_bound, position)
lower_bound['Fantasy Points'] = fantasy_points_lower

fantasy_points_upper = calculate_fantasy_ppr(upper_bound, position)
upper_bound['Fantasy Points'] = fantasy_points_upper

print(f"\nPredicted stats for the player's next season (with 95% confidence intervals):")
for i, target in enumerate(['Tgts', 'Rec', 'RecYds', 'RecTDs', 'RushAtt', 'RushYds', 'RushTD']):
    print(f"{target}: {mean_pred[0][i]:.2f} ({lower_bound[0][i]:.2f} - {upper_bound[0][i]:.2f})")