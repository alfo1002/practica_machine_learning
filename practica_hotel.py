import pandas as pd
from ydata_profiling import ProfileReport

hotel_bookings = pd.read_csv('hotel_bookings.csv')

hotel_bookings.info()