from sklearn.externals import joblib
from flask import Flask
import os


app = Flask(__name__)

path = os.getcwd() + '/static/'
#path = app.root_path + '/models/'

p_vectorizer = joblib.load(path + 'vectorizer.pkl')
p_labeler = joblib.load(path + 'labeler.pkl')
p_model_catalog = joblib.load(path + 'model_catalog.pkl')
p_log_model_1 = joblib.load(path + 'log_model_1.pkl')
p_log_model_2 = joblib.load(path + 'log_model_2.pkl')
p_log_model_3 = joblib.load(path + 'log_model_3.pkl')
p_log_model_4 = joblib.load(path + 'log_model_4.pkl')
p_xgb = joblib.load(path + 'xgb.pkl')
