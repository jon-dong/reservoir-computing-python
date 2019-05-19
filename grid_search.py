import numpy as np
import data1D
from reservoir import Reservoir
from sklearn.model_selection import GridSearchCV


b = Reservoir(n_res=2000, res_scale=0.1, res_encoding='phase', res_enc_param=1.5*np.pi,
              input_scale=.5, input_encoding='phase',
              random_projection='simulation', weights_type='complex gaussian',
              activation_fun='intensity', activation_param=10,
              parallel_runs=1,  bias_scale=0.1, leak_rate=0.15,
              pred_horizon=50, rec_pred_steps=1, forget = 100,
              train_method='ridge', train_param=1e1, verbose=1, refreshing=True,
              ref_horizon=500, parallel=100
             )
params = [
  # {'n_res': np.arange(1000, 5000, 500),
  {# 'res_scale': np.arange(.01, .1, .03),
  # 'input_scale': np.arange(.1, .8, .2),
  # 'bias_scale': np.arange(.01, .1, .03),
  'train_param': [5, 1e1, 1e2, 1e3]
  }
]
ctanh = GridSearchCV(estimator=b, param_grid=params, return_train_score=True, cv=3, verbose=2)
n_sequence=3
spatial_points = 30
lyap_exp = 0.0461
ks_data, xx, tt = data1D.kuramoto_sivashinsky_matlab(sequence_length=1000, n_sequence=n_sequence, spatial_points=spatial_points)
input_shape=ks_data.shape
input_data = ks_data
ref_horizon = 500
parallel = 100
ctanh.fit(input_data)

print(ctanh.best_params_)