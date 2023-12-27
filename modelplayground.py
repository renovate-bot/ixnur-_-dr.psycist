"""Have you ever been smacked by fate?"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer, plot_confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import random
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

load.dataset: pd.DataFrame = pd.read_csv("dataset.csv")
veri_df = pd.DataFrame(load.dataset)
veri_df.head()
veri_df['kelime_sayisi'] = veri_df['hasta'].apply(lambda x: len(x.split()))
veri_df['cumle_uzunlugu'] = veri_df['hasta'].apply(lambda x: len(x))

class dr_psycist:
    def __init__(self):
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    def eğit(self, veri):
        random.shuffle(veri)
        X_text = [item["hasta"] for item in veri]
        X_data = [item["hasta_verileri"] for item in veri]
        y = [item["rahatsizlik"] for item in veri]
        X_text = [self.clean_text(text) for text in X_text]

        parametreler = {
            'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
            'tfidfvectorizer__use_idf': (True, False),
            'multinomialnb__alpha': [1e-2, 1e-3, 1e-4]
        }

        grid_search = GridSearchCV(model_pipeline, param_grid, cv=10, scoring=make_scorer(mean_squared_error), n_jobs=-1)
        grid_search.fit(X_text, y)  
        self.model = grid_search.best_estimator_
        
        print(f"Eğitim tamamlandı. En iyi parametreler: {grid_search.best_params_}")
        print(f"En iyi çapraz doğrulama skoru: {grid_search.best_score_}")

        X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.6, random_state=422)
        self.visualize_model_performance(X_test, y_test)
        model = MLPRegressor(max_iter=10000)

        param_grid = {
            'mlpregressor__hidden_layer_sizes': [(500,), (1000,), (500, 500), (1000, 500)],
            'mlpregressor__activation': ['relu', 'tanh', 'logistic', 'identity'],
            'mlpregressor__solver': ['adam', 'sgd'],
            'mlpregressor__alpha': [0.0001, 0.001, 0.01, 0.1],
        }

        model_pipeline = Pipeline([
            ('countvectorizer', CountVectorizer(analyzer='char', ngram_range=(1, 3))),
            ('standard_scaler', StandardScaler(with_mean=False)),
            ('mlpregressor', model)
        ])

        grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring=make_scorer(mean_squared_error), n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"En iyi modelin test hatası (MSE): {mse}")
        print("En iyi modelin parametreleri:")
        print(grid_search.best_params_)

        model_filename = "trained_model.joblib"
        joblib.dump(best_model, model_filename)

        print("Yapay Zeka Psikolog oluşturuluyor...")
        veri_df = pd.read_csv("veri.csv")
        veri = veri_df.to_dict('records')
        print("Veri seti okundu.")
        print("Veri seti eğitim ve test olarak ayırılıyor...")

        print("Yapay Zeka Psikolog oluşturuldu.")
        psikolog = dr_psycist()
        psikolog.eğit(veri)

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = text.lower()
        return text

    def visualize_model_performance(self, X_test, y_test):
        disp = plot_confusion_matrix(self.model, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
        disp.ax_.set_xlabel('Predicted Label')
        disp.ax_.set_ylabel('True Label')
        disp.ax_.set_xticklabels(['Hayır', 'Evet'])
        disp.ax_.set_title('Confusion Matrix')
        plt.show()

        y_pred = self.model.predict(X_test)
        print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

    def modeli_guncelle(self, yeni_veri):
        X = [item["hasta_verileri"] for item in yeni_veri]
        y = [item["rahatsizlik"] for item in yeni_veri]
        self.model.fit(X, y)
        print("Model güncellendi.")

    def ozellik_secimi(self, X, y, k):
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        return X_new
        
    def trymodels(RandomForestRegressor,GradientBoostingRegressor, XGBRegressor):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_rf = RandomForestRegressor()
        model_gb = GradientBoostingRegressor()
        model_xgb = XGBRegressor()
        model_rf.fit(X_train, y_train)
        model_gb.fit(X_train, y_train)
        model_xgb.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        y_pred_gb = model_gb.predict(X_test)
        y_pred_xgb = model_xgb.predict(X_test)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        mse_gb = mean_squared_error(y_test, y_pred_gb)
        mse_xgb = mean_squared_error(y_test, y_pred_xgb)
        print("Random Forest MSE:", mse_rf)
        print("Gradient Boosting MSE:", mse_gb)
        print("XGBoost MSE:", mse_xgb)
        models = [model_rf, model_gb, model_xgb]
        mses = [mse_rf, mse_gb, mse_xgb]
        min_mse_index = mses.index(min(mses))
        best_model = models[min_mse_index]
        print("En iyi model:", best_model)
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)    
        model_rf = RandomForestRegressor()
        model_gb = GradientBoostingRegressor()
        model_xgb = XGBRegressor()
        model_rf.fit(X_train, y_train)
        model_gb.fit(X_train, y_train)
        model_xgb.fit(X_train, y_train)

veri = pd.read_csv("veri.csv")
veri = veri.to_dict('records')
psikolog = dr_psycist()
psikolog.eğit(veri)
yeni_veri = [.("NEWDATA").]  
print("Yeni veri eklendi.")
yeni_veri = [X["hasta"] for Y in yeni_veri]
X_data = [item["hasta_verileri"] for item in veri]
y = [item["rahatsizlik"] for item in veri]
print("Veri seti eğitildi.")
X = X_data + [item["hasta_verileri"] for item in yeni_veri]
y = y + [item["rahatsizlik"] for item in yeni_veri]
psikolog.model.fit(X, y)
print("Yeni veri ile modeli eğitildi.")
X_data = X
psikolog.modeli_guncelle(yeni_veri)
print("Yeni veri ile modeli güncellendi.")
X_selected = psikolog.ozellik_secimi(X_data, y, k=10)
