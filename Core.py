import re
import pymorphy2
from flask import Flask, jsonify
import numpy as np
import Config


app = Flask(__name__)


def predict_proba(s):
    transformed = Config.p_vectorizer.transform(s)
    s_meta_catalog = Config.p_model_catalog.predict_proba(transformed)

    s_meta = []
    p_log_models = [Config.p_log_model_1, Config.p_log_model_2, Config.p_log_model_3, Config.p_log_model_4]
    for model in p_log_models:
        s_meta.append(model.predict_proba(transformed))

    s_meta = np.stack(s_meta)
    s_meta_mean = np.mean(s_meta, axis=0)
    s_meta = np.hstack([s_meta_mean, s_meta_catalog])
    preds = Config.p_xgb.predict_proba(s_meta)

    categories = []
    normalized = []
    for i, arr in enumerate(preds):
        category = Config.p_labeler.classes_[np.where(arr == arr.max())[0][0]]
        categories.append(category)
        normalized.append(preprocess(s[i]))

    return jsonify(
        categories=categories,
        normalized=normalized
    )


def preprocess(word):
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


def f_tokenizer(s):
    morph = pymorphy2.MorphAnalyzer()

    t = s.split(' ')
    f = []
    for j in t:
        m = morph.parse(j)
        if len(m) != 0:
            f.append(m[0].normal_form)
        else:
            f.append(j)
    return f

