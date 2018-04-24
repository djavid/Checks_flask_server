from flask import Flask, request
from sklearn.externals import joblib
import numpy as np
import re
import pymorphy2

app = Flask(__name__)

p_vectorizer = None
p_labeler = None
p_model_catalog = None
p_log_model_1 = None
p_log_model_2 = None
p_log_model_3 = None
p_log_model_4 = None
p_xgb = None

labels = ['Алкоголь', 'Бакалея', 'Гастрономия', 'Дети', 'Для дома',
          'Животные', 'Здоровье', 'Кафе', 'Компьютер', 'Косметика',
          'Кулинария', 'Машина', 'Молочка', 'Мясо и птица', 'Напитки',
          'Не определена', 'Овощи и фрукты', 'Одежда и обувь', 'Рыба',
          'Снеки', 'Табак', 'Упаковка', 'Услуги', 'Хлеб', 'Чай и сладкое']


def predict_proba(s):
    s = p_vectorizer.transform(s)
    s_meta_catalog = p_model_catalog.predict_proba(s)

    s_meta = []
    p_log_models = [p_log_model_1, p_log_model_2, p_log_model_3, p_log_model_4]
    for model in p_log_models:
        s_meta.append(model.predict_proba(s))

    s_meta = np.stack(s_meta)
    s_meta_mean = np.mean(s_meta, axis=0)
    s_meta = np.hstack([s_meta_mean, s_meta_catalog])
    preds = p_xgb.predict_proba(s_meta)

    categories = []
    for i, arr in enumerate(preds):
        category = labels[np.where(arr == arr.max())[0][0]]
        categories.append(category)

        # print(preprocess(s[i]))
        print(category)

    return ' '.join(categories)


def preprocess(word):
    print(word)
    s = word
    s = re.sub(r'[0-9]{4,}', r' ', s)
    s = re.sub(r'^[0-9]+|[0-9]+$', r'', s)
    s = re.sub(r'([а-я|a-z]+)[.]', r'\1. ', s)
    s = re.sub(r'([0-9]+)(грамов|грам|гра|гр\.|гp|г\.|гр|г|gr|g)', r'\1гр.', s)
    s = re.sub(r'([0-9]+)(шту|шт\.|шт|ш\.)', r'\1шт.', s)
    s = re.sub(r'([0-9]+)(кило|кг\.|кг)', r'\1 кг.', s)
    s = re.sub(r'([0-9]+)(литр\.|литр\.|лт|л)', r'\1л', s)
    s = re.sub(r'([0-9]+)(( кило | кг\.| кг | кг\b))', r'кг. ', s)
    s = re.sub(r'[!|?|*|=|(|)|/|\\|:|+|#|$|&]', r' ', s)
    s = re.sub(r'%', r'% ', s)
    s = re.sub(r'([a-z|а-я]+)([0-9]+)', r'\1 \2', s)
    s = re.sub(r'([0-9]+)([a-z|а-я]+)', r'\1 \2', s)
    s = re.sub(r' , | ,', r', ', s)
    s = re.sub(r'([a-z|а-я]+),([a-z|а-я]+)', r'\1, \2', s)
    s = re.sub(r'\s+', r' ', s)
    s = s.strip()
    s = s.lower()
    s = s[:1].upper() + s[1:]

    return s


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict', methods=['GET'])
def predict():
    good_title = request.args.get('value')

    return predict_proba([good_title])


if __name__ == '__main__':

    # try:
        morph = pymorphy2.MorphAnalyzer()

        print(app.root_path)
        p_vectorizer = joblib.load('/models/vectorizer.pkl')
        print('model loaded')
        p_labeler = joblib.load('/models/labeler.pkl')
        print('model loaded')
        p_model_catalog = joblib.load('/models/model_catalog.pkl')
        print('model loaded')
        p_log_model_1 = joblib.load('/models/log_model_1.pkl')
        print('model loaded')
        p_log_model_2 = joblib.load('/models/log_model_2.pkl')
        print('model loaded')
        p_log_model_3 = joblib.load('/models/log_model_3.pkl')
        print('model loaded')
        p_log_model_4 = joblib.load('/models/log_model_4.pkl')
        print('model loaded')
        p_xgb = joblib.load('/models/xgb.pkl')
        print('model loaded')

    # except Exception as e:
    #     print('No model here')
    #     print(e.__str__())

        app.run()


def f_tokenizer(s):
    t = s.split(' ')
    f = []
    for j in t:
        m = morph.parse(j)
        if len(m) != 0:
            f.append(m[0].normal_form)
        else:
            f.append(j)
    return f
