import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime

from google.colab import files
uploaded_files = files.upload()

import io

# Read the fatalities data
fatalities_data = pd.read_csv(io.BytesIO(uploaded_files['drugwar.csv']), parse_dates=['event_date'])

# Read the misinformation tweet data
tweets_data = pd.read_csv(io.BytesIO(uploaded_files['data.csv']), parse_dates=['Date posted'])

# Convert 'event_date' to a common format (dd-mm-yyyy)
fatalities_data['event_date'] = pd.to_datetime(fatalities_data['event_date'], format='%d-%b-%y').dt.strftime('%d-%m-%Y')

# Convert 'Date posted' to a common format (dd-mm-yyyy)
tweets_data['Date posted'] = pd.to_datetime(tweets_data['Date posted'], format='%d-%m-%Y %H:%M').dt.strftime('%d-%m-%Y')

# Step 4: Calculate the counts

# Count the number of fatalities for each date
fatalities_count = fatalities_data.groupby('event_date').size().reset_index(name='number_of_fatalities')

# Count the number of misinformation tweets for each date
tweets_count = tweets_data.groupby('Date posted').size().reset_index(name='frequency_of_misinformation_tweets')

# Step 5: Merge the datasets
merged_data = pd.merge(fatalities_count, tweets_count, left_on='event_date', right_on='Date posted', how='inner')

# Step 6: Define variables
X = merged_data['number_of_fatalities']
y = merged_data['frequency_of_misinformation_tweets']

# Step 7: Fit the regression model
X = sm.add_constant(X)  # Add a constant term to the predictor variable
model = sm.OLS(y, X)
results = model.fit()

# Step 8: Interpret the results
print(results.summary())


import matplotlib.pyplot as plt

# Scatter plot of data points
plt.scatter(X['number_of_fatalities'], y, label='Data Points')

# Regression line
plt.plot(X['number_of_fatalities'], results.fittedvalues, color='red', label='Regression Line')

plt.xlabel('Number of Fatalities')
plt.ylabel('Frequency of Misinformation Tweets')
plt.title('Regression Analysis')
plt.legend()
plt.show()