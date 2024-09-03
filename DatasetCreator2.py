import time
import pandas as pd
import numpy as np
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup

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


# df = pd.DataFrame(columns=['Year', 'Name', 'G'])
#
# # Path to your chromedriver
# DRIVER_PATH = 'C:\\Users\\Sebastian\\Downloads\\chromedriver-win64\\chromedriver.exe'
#
# # Set up Chrome options if needed
# chrome_options = webdriver.ChromeOptions()
#
# # Set up the ChromeDriver service
# service = Service(DRIVER_PATH)
#
# # Initialize WebDriver with the driver path specified correctly
# driver = webdriver.Chrome(service=service, options=chrome_options)
#
# # URL of the webpage with dropdown
# url = 'https://www.pro-football-reference.com/players/A/'
# driver.get(url)
#
# wait = WebDriverWait(driver, 0)
#
# urls = (driver.find_element(By.ID, 'div_alphabet')
#         .find_element(By.TAG_NAME, 'ul').find_elements(By.TAG_NAME, 'li'))
# for url in urls:
#     print(url.text)
#
#     # Click on the URL
#     url.click()
#     wait = WebDriverWait(driver, 10)
#
#     # Get all players
#     players = driver.find_element(By.ID, 'div_players').find_elements(By.TAG_NAME, 'p')
#     for p in players:
#         player = p.find_element(By.TAG_NAME, 'a')
#         player.click()
#         wait = WebDriverWait(driver, 0)
#
#         print(player)
#
#         response = requests.get(player.href)
#
#         soup = BeautifulSoup(response.content, 'html.parser')
#
#         # Example: Find the player's name
#         player_name = soup.find('div', id='meta').find('h1').text
#
#         # Example: Find the player's stats table
#         stats_table = soup.find('table', {'id': 'rushing_and_receiving'})
#
#         # Extract table headers
#         headers = [th.text for th in stats_table.find_all('th')[1:]]
#         headers = headers[headers.index('Year'):headers.index('Awards') + 1]
#
#         # Extract table rows
#         rows = stats_table.find_all('tr')
#         for row in rows[2:len(rows) - 1]:  # Skip the header row
#             year = row.find('th')
#             year = ''.join(filter(str.isdigit, year.text))
#             cols = row.find_all('td')
#
#             df = df.append({'Year': year, 'Name': player_name, 'G': cols[5].text, 'GS': cols[6].text}, ignore_index=True)
#
#         # Go back to the previous page
#         driver.back()
#
# # Close the WebDriver
# driver.quit()
#
# df.to_csv('Games_Played_Data', index=False)
#
# game_data = pd.read_csv('Data/Games Played Data.csv')
#
# Big_Data = pd.read_csv('Data/NFL Final Data.csv')
#
# merged_df = pd.merge(Big_Data, game_data, on=['Name', 'Year'], how='inner')
#
#
# # remove any duplicates of name and year
# merged_df = merged_df.drop_duplicates(subset=['Name', 'Year'])
#
# merged_df.to_csv('Merged_Data.csv', index=False)

all_data = []
for year in range(2015, 2024):
    url = f'https://www.pro-football-reference.com/years/{year}/fantasy.htm'

    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        headers = [th.text for th in soup.find('table', {'id': 'fantasy'}).find_all('th')]
        headers = headers[11:43]
        headers.insert(0, 'Year')
        headers[1] = 'Name'

        rows = soup.find('table', {'id': 'fantasy'}).find_all('tr')

        for row in rows[2:]:
            cols = row.find_all('td')
            cols_text = [col.text for col in cols]
            if len(cols_text) <= 1:
                continue

            cols_text[0] = cols_text[0].replace('*', '')
            cols_text[0] = cols_text[0].replace('+', '')
            cols_text.insert(0, year)
            all_data.append(cols_text)

df = pd.DataFrame(all_data, columns=headers)
fantasy_data = df[['Year', 'Name', 'PPR', 'PosRank']]

Big_Data = pd.read_csv('NFL Final Data.csv')

merged_df = pd.merge(Big_Data, fantasy_data, on=['Name', 'Year'], how='inner')

# remove any duplicates of name and year
merged_df = merged_df.drop_duplicates(subset=['Name', 'Year'])

merged_df.to_csv('Merged_Data.csv', index=False)


