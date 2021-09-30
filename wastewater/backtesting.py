import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer

class Backtest:
  """
  Runs backtesting on the provided data for a supplied model and provides heler methods.
  """
  def __init__(self):
    self.data = pd.read_csv('data/BC_COVID_CASES_Cases.csv')
    self.clean_data()
    self.predictions_info = pd.read_csv('data/predictions.csv')['Date:Delay'].to_list()
    self.predictions = pd.read_csv('data/predictions.csv')

  def __call__(self, model):
    """
    Backtests model for every prediction.
    """
    for items in self.predictions_info:
      items = items.split(':')
      T = items[0]
      D = int(items[1])

      T_prime = (datetime.strptime(T, '%Y-%m-%d') - timedelta(D)).strftime('%Y-%m-%d')

      # Shift the target 'New cases' up by D days
      X_prime = self.data.copy() 
      X_prime['New cases'] = X_prime['New cases'].shift(-D)


      idx = X_prime.index[X_prime['Date'] == T_prime].tolist()[0]
      X_prime = X_prime[0:idx+1]

      X = X_prime.loc[:, 'Cumulative cases':'Active cases']
      Y = X_prime['New cases']

      self.evaluate_model(X, Y, model)

      break

  def evaluate_model(self, X, Y, model):
    """
    Trains and tests model using k-fold cross validation.
    """
    kf = KFold(n_splits=4)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
      X_train = X.iloc[train_index]
      Y_train = Y.iloc[train_index]
      X_test = X.iloc[test_index]
      Y_test = Y.iloc[test_index].to_numpy()

      model.fit(X_train, Y_train)
      Y_hat = model.predict(X_test)

      # RMSE
      # print(np.sqrt(np.mean((Y_hat-Y_test)**2))) 


  def clean_data(self):
    """
    Missing data on the first 17 days are zero'd.
    Fill forward is performed on the remaining missing days (weekends)
    """
    self.data[0:18] = self.data[0:18].fillna(value=0)
    self.data = self.data.fillna(method='ffill')

  def get_sample_data(self):
    pass

  def output_prediction_score(self):
    pass

