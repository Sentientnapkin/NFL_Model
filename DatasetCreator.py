import pandas as pd

# Read the data
rushing_df = pd.read_csv('Data/RB Data.csv')
receiving_df = pd.read_csv('Data/ReceiverData.csv')

# Merge the dataframes
merged_df = pd.merge(rushing_df, receiving_df, on=['Year', 'Name'], how='outer', suffixes=('_rush', '_rec'))

# Fill NaN values with 0 for numerical columns
numeric_columns = merged_df.select_dtypes(include=['float64', 'int64']).columns
merged_df[numeric_columns] = merged_df[numeric_columns].fillna(0)

# Calculate total yards and touchdowns
merged_df['TotalYards'] = merged_df['RushYDs'] + merged_df['RecYds']
merged_df['TotalTDs'] = merged_df['RushTD'] + merged_df['RecTDs']

# Sort by total yards
merged_df = merged_df.sort_values('TotalYards', ascending=False)

# Output the data
merged_df.to_csv('CombinedData.csv')

NFL_Data = pd.read_csv('Data/NFL Player Stats(1922 - 2022).csv')

# Convert Year to integer type in both dataframes
merged_df['Year'] = merged_df['Year'].astype(int)
NFL_Data['Year'] = NFL_Data['Year'].astype(int)

# Filter both datasets for years 2015-2022
merged_df_2015_2022 = merged_df[(merged_df['Year'] >= 2015) & (merged_df['Year'] <= 2022)]
NFL_Data = NFL_Data[(NFL_Data['Year'] >= 2015) & (NFL_Data['Year'] <= 2022)]

# Merge the datasets on 'Name' and 'Year'
merged_df_2015_2022 = pd.merge(merged_df_2015_2022, NFL_Data[['Year', 'Name', 'Age', 'Pos']],
                               on=['Year', 'Name'],
                               how='left')

# Fill NaN values in Age and Pos columns (in case of unmatched players)
merged_df_2015_2022['Age'] = merged_df_2015_2022['Age'].fillna('Unknown')
merged_df_2015_2022['Pos'] = merged_df_2015_2022['Pos'].fillna('Unknown')

# Filter out unknown players and keep only TE, WR, and RB positions
valid_positions = ['TE', 'WR', 'RB']
known_positions = merged_df_2015_2022[merged_df_2015_2022['Pos'].isin(valid_positions)]
unknown_positions = merged_df_2015_2022[merged_df_2015_2022['Pos'] == 'Unknown']

# Check if players with unknown positions appear elsewhere with a known position
known_players = NFL_Data[NFL_Data['Pos'].isin(valid_positions)]
known_position_dict = known_players.set_index('Name')['Pos'].to_dict()

# Update positions for unknown players
unknown_positions.loc[:, 'Pos'] = unknown_positions['Name'].map(known_position_dict).fillna('Unknown')

# Combine known and updated unknown positions
merged_df_2015_2022 = pd.concat([known_positions, unknown_positions[unknown_positions['Pos'].isin(valid_positions)]], ignore_index=True)

# Function to find and adjust age
def adjust_age(row, player_age_dict):
    if row['Age'] == 'Unknown' and (row['Name'], row['Year']) in player_age_dict:
        known_age, known_year = player_age_dict[(row['Name'], row['Year'])]
        return known_age + (row['Year'] - known_year)
    return row['Age']

# Build a dictionary of known ages with the year
known_ages = NFL_Data[NFL_Data['Age'] != 'Unknown']
player_age_dict = {(name, year): (int(age), year) for name, year, age in known_ages[['Name', 'Year', 'Age']].values}

# Adjust ages for unknown players
merged_df_2015_2022['Age'] = merged_df_2015_2022.apply(adjust_age, axis=1, player_age_dict=player_age_dict)

# Get 2023 data
df_2023 = merged_df[merged_df['Year'] == 2023]

# Find players in 2023 who were also in 2022
df_2022 = merged_df_2015_2022[merged_df_2015_2022['Year'] == 2022]
players_2023 = df_2023[df_2023['Name'].isin(df_2022['Name'])]

# Add Age and Pos information for 2023 players
players_2023 = pd.merge(players_2023, df_2022[['Name', 'Age', 'Pos']], on='Name', how='left')

# Increment Age by 1 for 2023 players
players_2023['Age'] = players_2023['Age'].apply(lambda x: int(x) + 1 if x != 'Unknown' else x)

# Combine 2015-2022 data with 2023 data
final_df = pd.concat([merged_df_2015_2022, players_2023], ignore_index=True)

# Sort the dataframe by Year and Name
final_df = final_df.sort_values(['Year', 'Name'])

# Reset the index
final_df = final_df.reset_index(drop=True)

# Output the final data
final_df.to_csv('FinalData.csv', index=False)
