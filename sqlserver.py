#First, make sure you have the following packages installed: `pyodbc`, `tensorflow`, and `keras`.


#import pyodbc
import sqlalchemy as db
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sqlalchemy import text

# Define database connection details
server = 'localhost'
database = 'AdventureWorksDW2019'
username = 'sa'
password = '$42023*'

# Create a database connection
#conn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password)
#from sqlalchemy import create_engine
 
 #server = 'localhost'
 #database = 'AdventureWorksDW2019'
 #username = 'sa'
 #password = '$42023*'
 
# Define database connection string
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=SQL Server'

# Create a database engine with options
#engine = create_engine(connection_string, connect_args={'options': '-c lock_timeout=180000'})

# Create a database engine
engine = db.create_engine(connection_string)

# Create a database connection
conn = engine.connect()

# Retrieve data from database
query = text("SELECT [ProductKey],[CustomerKey],[SalesTerritoryKey],CustomerPONumber  FROM [AdventureWorksDW2019].[dbo].[FactInternetSales]")
#df = pd.read_sql(query, conn) 
#with engine.connect() as conn:
#result_set= conn.execute(query)
result_set = conn.execute(query)

# Load the result set into a pandas dataframe
df = pd.read_sql_query(sql=query, con=engine.connect())

#df = pd.DataFrame(result_set.fetchall(), columns=result_set.keys())

# Close the result set
result_set.close()
conn.close()

# Print Dataframe
print(df.columns)
print(df.index)

# Split data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2)

# Define the input and output variables
X_train = train_df.drop(columns=['ProductKey'])
y_train = train_df[str('SalesTerritoryKey')]

X_test = test_df.drop(columns=['ProductKey'])
y_test = test_df[str('SalesTerritoryKey')]

# Define the neural network
model = Sequential()
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(X_train.shape)

x = np.asarray(X_train).astype('float32')
y = np.asarray(y_train).astype('float32')

# Train the model
model.fit(x, y, epochs=10, batch_size=16, validation_split=0.1)

xt = np.asarray(X_test).astype('float32')
yt = np.asarray(y_test).astype('float32')

# Evaluate the model
score = model.evaluate(xt, yt)
print('Test accuracy:', score[1])

# Close the database connection
#conn.close()