import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('weather.csv')

# Initial inspection
print(df.head())
print(df.describe())
print(df.info())

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Detect and handle outliers using IQR
Q1 = df['temperature'].quantile(0.25)
Q3 = df['temperature'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['temperature'] < (Q1 - 1.5 * IQR)) | (df['temperature'] > (Q3 + 1.5 * IQR)))]

# Fix date format
df['date'] = pd.to_datetime(df['date'])

# Feature engineering
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['year'] = df['date'].dt.year
df['weekday'] = df['date'].dt.weekday
df['season'] = df['month'].apply(lambda x: 'winter' if x in [12, 1, 2] else ('spring' if x in [3, 4, 5] else ('summer' if x in [6, 7, 8] else 'autumn')))

# Normalization/Standardization
scaler = StandardScaler()
df[['temperature']] = scaler.fit_transform(df[['temperature']])

# Save the cleaned data
df.to_csv('cleaned_weather_data.csv', index=False)
