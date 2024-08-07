import pandas as pd
import numpy as np

# df = pd.read_csv('Data/NFL Player Master Data.csv')
#
# # Merge the DataFrames on 'Year' and 'Name'
# merged_df = pd.read_csv('RZ_Data.csv')
#
# merged_df = merged_df[merged_df['Year'] != 2014]
# merged_df.fillna(0, inplace=True)
# merged_df = merged_df.sort_values(by=['Name', 'Year'], ascending=True)
#
# # Remove duplicate rows based on 'Year' and 'Name' columns
# merged_df = merged_df.drop_duplicates(subset=['Year', 'Name'])
#
# # Save the merged DataFrame to a new CSV file
# merged_df.to_csv('RZ_Data.csv', index=False)
#
#
# better_merged_df = pd.merge(df, merged_df, on=['Name', 'Year'], how='inner')
#
#
# better_merged_df.to_csv('NFL Final Data RZ.csv', index=False)

# df1 = pd.read_csv('Data/NFL Final Data no RZ.csv')
# df2 = pd.read_csv('Data/RZ_Data.csv')
#
# merged_df = pd.merge(df1, df2, on=['Name', 'Year'])
#
# merged_df.to_csv('NFL Final Data.csv', index=False)
#
# df = pd.read_csv('NFL Final Data.csv')
# df['RZ_CompPercent'] = round(df['RZ_CompPercent']/100, 4)
#
# df.to_csv('NFL Final Data.csv', index=False)
