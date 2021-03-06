import pandas as pd
import numpy as np
import os
from collections import namedtuple
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb

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

  def __call__(self, output_predication_csv=False, alpha=None, poly=2, **kwargs):
    """
    Backtests model for every prediction.
    No forward contamination
    Method: 
    1. Shift the target 'New cases' up T - D days
    2. Get the T - D day and create X_prime
    3. From the original data test against T
    """
    # Manage arguments
    try:
      model = kwargs['model']
      days = kwargs['days']
      features = kwargs['features']
    except KeyError:
      print('Required argument is missing')
      return None

    for items in self.predictions_info:
      items = items.split(':')
      T = items[0]
      D = int(items[1])

      X_prime, idx = self.construct_x_prime(T, D)

      # Convert to numpy arrays
      X_train = X_prime.loc[:, features]
      X_test = self.data.loc[idx+1, features]
      Y_train = X_prime['New cases']
      Y_test = self.data.loc[idx+D+1, 'New cases']

      
      ########################################
      # if T == '2020-04-01' and D == 1:
      #   print(X_train)
      #   print(Y_train)
      #   print(X_test)
      #   print(Y_test)
      #   break
      ########################################

      Y_hat = None
      if model == 'regression':
        Y_hat = self.run_regression(X_train, Y_train, X_test, days, model=LinearRegression())
      elif model == 'ridge_regression':
        Y_hat = self.run_regression(X_train, Y_train, X_test, days, model=Ridge(alpha=alpha))
      elif model == 'lasso_regression':
        Y_hat = self.run_regression(X_train, Y_train, X_test, days, model=Lasso(alpha=alpha, tol=0.01))
      elif model == 'polynomial_regression':
        Y_hat = self.run_polynomial_regression(X_train, Y_train, X_test, days, poly)
      elif model == 'knn':
        Y_hat = self.run_knn(X_train, Y_train, X_test, days, model=KNeighborsRegressor(n_neighbors=3, weights = "distance", metric = "manhattan"))
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
      T_prime = (datetime.strptime(T, '%Y-%m-%d') - timedelta(D+1)).strftime('%Y-%m-%d')

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
    1. For the wastewater data, 0 seems to be equivilent to NaN
    2. Forward fill all values
    3. Only NaN is left, so set them to 0
    4. Try changing cummulative features to daily, set adjustments to zero
    """
    # cummulative = [
    #   'Cumulative cases',
    #   'Cumulative Vancouver Coastal',
    #   'Cumulative Fraser Health',
    #   'Cumulative Island Health ',
    #   'Cumulative Interior Health',
    #   'Cumulative Northern Health',
    #   'Recovered'
    # ]

    # self.data.loc[:, 'Annacis Island':'Northwest Langley'] = self.data.loc[:, 'Annacis Island':'Northwest Langley'].replace(0, np.nan)
    # self.data[cummulative] = self.data[cummulative].diff()
    # self.data.loc[0,'Cumulative cases':'Cumulative Vancouver Coastal'] = 1
    # self.data = self.data.fillna(method='ffill')
    # self.data = self.data.fillna(value=0)

    # # Clear the last row from fill forward
    # self.data.loc[self.data.shape[0]-1,'New cases':] = np.nan
    
    # # There are small adjustments/corrections in the cummulative data.
    # numbers = self.data._get_numeric_data()
    # numbers[numbers < 0] = 0

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

    output_to_upload = self.predictions
    output_to_upload.to_csv(full_path, index=False)

  def run_regression(self, X_train, Y_train, X_test, days, model) -> float:
    """
    Returns a prediction using standard linear regression
    """
    if days: 
      X_train = X_train.tail(days)
      Y_train = Y_train.tail(days)
    
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy().reshape(1, -1)
    Y_train = Y_train.to_numpy()

    model.fit(X_train, Y_train)
    Y_hat = model.predict(X_test)[0]
    return Y_hat

  def run_polynomial_regression(self, X_train, Y_train, X_test, days, poly) -> float:
    if days: 
      X_train = X_train.tail(days)
      Y_train = Y_train.tail(days)

    # print(X_train)
    # print(Y_train)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy().reshape(1, -1)

    poly_reg = PolynomialFeatures(degree=poly)
    X_train_poly = poly_reg.fit_transform(X_train)
    X_test_poly = poly_reg.fit_transform(X_test)
    model = LinearRegression()

    model.fit(X_train_poly, Y_train)
    Y_hat = model.predict(X_test_poly)[0]
    return Y_hat

  def run_knn(self, X_train, Y_train, X_test, days, model) -> float:
    """
    Returns a prediction using KNN
    """
    if days: 
      X_train = X_train.tail(days)
      Y_train = Y_train.tail(days)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy().reshape(1, -1)
    Y_train = Y_train.to_numpy()#.reshape(1, -1)

    #model = KNeighborsRegressor()
    model.fit(X_train, Y_train)
    Y_hat = model.predict(X_test)[0]
    return Y_hat