import pandas as pd
from ydata_profiling import ProfileReport

hotel_bookings = pd.read_csv('hotel_bookings.csv')

hotel_bookings.info()

profile = ProfileReport(hotel_bookings, title="Hotel Bookings Data Profiling Report", explorative=True)
#profile.to_file("hotel_bookings_profile_report.html")

# reservation_status: contiene información de si la reserva fue cancelada o no, lo cual puede introducir sesgo en el modelo.
hotel_bookings = hotel_bookings.drop(['reservation_status', 'reservation_status_date'], axis=1)

is_cancelled = hotel_bookings['is_canceled'].copy()
hotel_data = hotel_bookings.drop(['is_canceled'], axis=1) # Drop the target variable from the features

## Split the dataset into training and testing sets
original_count = len(hotel_bookings)
training_size = 0.60
test_size = (1 - training_size) / 2
training_count = int(original_count * training_size)
test_count = int(original_count * test_size)
validation_count = original_count - training_count - test_count

print(training_count, test_count, validation_count, original_count)

# Split the dataset into training, testing, and validation sets
from sklearn.model_selection import train_test_split

train_x, rest_x, train_y, rest_y = train_test_split(hotel_data, is_cancelled, train_size=training_count)

test_x, validate_x, test_y, validate_y = train_test_split(rest_x, rest_y, train_size=test_count)

print(len(train_x), len(test_x), len(validate_x))

# onehotencoder sirve para convertir variables categóricas en variables numéricas
# hotel es una variable categórica que indica el tipo de hotel, por lo que se puede aplicar one-hot encoding
# # sparse_output=False significa que se devuelve una matriz densa en lugar de una matriz dispersa
# handle_unknown='ignore' significa que si hay una categoría desconocida en los datos de prueba, se ignorará 
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
one_hot_encoder.fit(train_x[['hotel']])
one_hot_encoder.transform(train_x[['hotel']])

# Binarizer convierte variables categóricas en variables binarias
from sklearn.preprocessing import Binarizer
binarizer = Binarizer()
_ = train_x.copy()
binarizer.fit(_[['total_of_special_requests']])
_['has_made_special_requests'] = binarizer.transform(_[['total_of_special_requests']])
_[['total_of_special_requests', 'has_made_special_requests']].sample(10)


# RobustScaler es una técnica de escalado que es robusta a los valores atípicos
# adr es un valor que representa una ganancia por habitación, podría ser negativa 
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
_ = train_x.copy()
# sirve para ajustar el escalador a los datos de entrenamiento
robust_scaler.fit(_[['adr']]) 
# transform aplica la transformación a los datos de entrenamiento
_['adr_scaled'] = robust_scaler.transform(train_x[['adr']])
_[['adr', 'adr_scaled']].sample(10)



##################### PIPELINES ####################################
# Como aplicar onehotencoder a las variables categóricas
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline

one_hot_encoding = ColumnTransformer([
    (
            'one_hot_encode',
            OneHotEncoder(sparse_output=False, handle_unknown='ignore'),
            [
                'hotel',
                'meal',
                'distribution_channel',
                'reserved_room_type',
                'assigned_room_type',
                'customer_type'
            ]
    )
])

binarizer = ColumnTransformer([
    (
            'binarizer',
            Binarizer(),
            [
                'total_of_special_requests',
                'required_car_parking_spaces',
                'booking_changes',
                'previous_bookings_not_canceled',
                'previous_cancellations',
            ]
    )
])

# OneHotEncoder y Binarizer se pueden combinar en un pipeline
# el onehotencoder se aplica para romper jerarquías de variables categóricas en variables binarias
one_hot_binarized = Pipeline([
    ('binarizer', binarizer),
    ('one_hot_encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# RobustScaler se puede aplicar a la columna 'adr' para escalar los valores de ganancia por habitación
scaler = ColumnTransformer([
    ('scaler', RobustScaler(), ['adr'])
])

# Passthrough se utiliza para mantener las columnas que no se transforman, en este caso, las noches de estancia
passthrough = ColumnTransformer([
    (
        'cualquier_id',
        'passthrough',
        [
            'stays_in_week_nights',
            'stays_in_weekend_nights',
        ]
    )
])

# Finalmente, se combinan todas las transformaciones en un pipeline
feature_engineering_pipeline = Pipeline([
    (
        'features',
        FeatureUnion([
            ('one_hot_encoding', one_hot_encoding),
            ('binarizer', one_hot_binarized),
            ('scaler', scaler),
            ('passthrough', passthrough)
        ])
    )
])

# entrenar el pipeline con los datos de entrenamiento
feature_engineering_pipeline.fit(train_x)

# transformar los datos de entrenamiento, prueba y validación
featurized = feature_engineering_pipeline.transform(train_x)
featurized.shape

print(featurized)

# hasta el paso anterior se han transformado 
# las variables categóricas en variables numéricas, 
# se han escalado las variables numéricas y se han 
# mantenido las variables que no se transforman
# como resultado se obtiene una matriz de características
# que se puede utilizar para entrenar un modelo de machine learning

