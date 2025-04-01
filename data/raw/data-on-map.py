import pandas as pd
import folium
'''
This script is part of course: "Case Study: Model Engineering" (DLBDSME01) from IU Internation University.

This code illustrates how to visualize datapoints from the dataset provided for use case 2.

Author: Prof. Dr. Bertram Taetz
Date: 10.01.2023
'''

# Load data from file
data = pd.read_csv('data-apr14.csv')
print(f'Number of trips: {data.shape[0]:,} \n\n' f'{data.head()}')

# Use part of the dataset (first 5 samples) to illustrate visualization!
dataSample = data[0:5]

# Create map and add locations
map = folium.Map(dataSample[['Lat', 'Lon']].mean().values.tolist())
for lat, lon in zip(dataSample['Lat'], dataSample['Lon']):
    folium.Marker([lat, lon]).add_to(map)

sw = dataSample[['Lat', 'Lon']].min().values.tolist()
ne = dataSample[['Lat', 'Lon']].max().values.tolist()
map.fit_bounds([sw, ne])

# Save map
map.save('map.html')

print('Saved map in current folder as map.html, to be viewed e.g. via web browser!')
