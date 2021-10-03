import pandas as pd
import numpy as np
import os
from collections import namedtuple
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

class Backtest:
  """
  Runs backtesting on the provided data for a supplied model and provides help er methods.
  """
  def __init__(self):
    self.data = pd.read_csv('../data/BC_COVID_CASES_Cases.csv')
    self.wastewater_data = pd.read_csv('../data/BC_COVID_CASES_Wastewater.csv')
    self.join_wastewater_data()
    self.clean_data()
    self.predictions_info = pd.read_csv('../data/predictions.csv')['Date:Delay'].to_list()
    self.predictions = pd.read_csv('../data/predictions.csv')
    self.predictions['Actual'] = 0 #For calculating RMSPE

  def __call__(self, output_predication_csv = False, **kwargs):
    """
    Backtests model for every prediction.
    No forward contamination
    Method: 
    1. Shift the target 'New cases' up T - D days
    2. Get the T - D day and create X_prime
    3. From the original data test against T
    """
    model = kwargs['model']
    for items in self.predictions_info:
      items = items.split(':')
      T = items[0]
      D = int(items[1])

      X_prime, idx = self.construct_x_prime(T, D)

      # Convert to numpy arrays
      X_train = X_prime.loc[:, 'Cumulative cases':'Active cases'].to_numpy()
      X_test = self.data.loc[idx+D, 'Cumulative cases':'Active cases'].to_numpy().reshape(1, -1)
      Y_train = X_prime['New cases'].to_numpy()
      Y_test = self.data.loc[idx+D, 'New cases']

      Y_hat = None
      if model == 'regression':
        Y_hat = self.run_regression(X_train, Y_train, X_test)
      else:
        print('Please pass a model to train and evaluate.')
      
      self.add_prediction(Y_hat, Y_test, T, D)


    RMSPE = self.RMSPE()
    if output_predication_csv: self.output_csv(model)
    return RMSPE

  def construct_x_prime(self, T, D):
      """
      Helper method. Returns X_prime and the index for the last row of X_prime.
      """
      T_prime = (datetime.strptime(T, '%Y-%m-%d') - timedelta(D)).strftime('%Y-%m-%d')

      # Shift the target 'New cases' up by D days
      X_prime = self.data.copy() 
      X_prime['New cases'] = X_prime['New cases'].shift(-D)

      # idx is the index of the last row of X'
      idx = X_prime.index[X_prime['Date'] == T_prime].tolist()[0]
      X_prime = X_prime[0:idx+1]
      return X_prime, idx

  def train_test_sets(self, X, num = 4) -> list:
    """
    Helper function. Returns a list of tuples containg the data frame X split num ways
    Output: [(X_train, X_test) ...]
    """
    TrainTest = namedtuple('TrainTest', 'train test')
    output = list()
    kf = KFold(n_splits = num)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
      X_train = X.iloc[train_index]
      X_test = X.iloc[test_index]
      output.append(TrainTest(X_train, X_test))
    return output

  def RMSPE(self) -> float:
    """
    Root Mean Squared Predication Error
    """
    # Count => Y_hat, Actual => Y
    return np.sqrt(np.mean((self.predictions['Count'] - self.predictions['Actual'])**2))
  
  def join_wastewater_data(self):
    self.wastewater_data['Date']= pd.to_datetime(self.wastewater_data['Date'], dayfirst=True)
    self.wastewater_data = self.wastewater_data.pivot(index='Date', columns='Plant')
    self.wastewater_data.columns = self.wastewater_data.columns.droplevel(0)
    self.wastewater_data['Date'] = self.wastewater_data.index.astype(str)
    self.wastewater_data = self.wastewater_data.rename_axis(None)
    self.wastewater_data = self.wastewater_data.fillna(value=0)
    self.data = self.data.merge(self.wastewater_data, on='Date', how='left')

  def clean_data(self):
    """
    Cleans the given data set.
    Methodology:
    1. Missing data on the first 17 days are zero'd.
    2. Forward fill is performed on the remaining missing days (weekends)
    3. Zero all missing data for watewater
    """
    self.data.loc[:, 'Annacis Island':'Northwest Langley'] = self.data.loc[:, 'Annacis Island':'Northwest Langley'].fillna(value=0)
    self.data = self.data.fillna(method='ffill')
    self.data = self.data.fillna(value=0)
    
  def add_prediction(self, Y_hat, Y_test, T, D):
    """
    Adds the predication of the model to the predications output.
    """
    # Get the index of the Date:Delay signature
    idx = self.predictions.index[self.predictions['Date:Delay'] == f'{T}:{D}'].tolist()[0]
    self.predictions.loc[idx,'Count'] = Y_hat
    self.predictions.loc[idx,'Actual'] = Y_test
  
  def get_sample_data(self, signature):
    """
    Returns the X' dataset based on the signature 'YYYY-MM-DD:PD'.
    Can be used for benchmarking, tuning, and implementing models.
    """
    items = signature.split(':')
    T = items[0]
    D = int(items[1])
    return self.construct_x_prime(T, D)[0]

  def output_csv(self, model):
    file = f'{model}_predictions.csv'
    folder = './output'

    if not os.path.exists(folder):
      os.mkdir(folder)
    full_path = os.path.join(folder, file)   

    output_to_upload = self.predictions[['Date:Delay', 'Count']]
    output_to_upload.to_csv(full_path, index=False)

  def run_regression(self, X_train, Y_train, X_test) -> float:
    """
    Returns a prediction using standard linear regression
    """
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_hat = model.predict(X_test)[0]
    return Y_hat

# model = Backtest()
# RMSPE = model(output_predication_csv = False, model='regression')
# print(f'RMSPE: {RMSPE}')

# X_prime = model.get_sample_data('2021-08-14:3')
# print(model.train_test_sets(X_prime)[0].train) # Prints the train set of the first fold

