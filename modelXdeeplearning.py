"""terelellii"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer, plot_confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class dr_psycist:
    def __init__(self):
        self.tfidf_model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        self.deep_model = None

    def eğit(self, veri):
        random.shuffle(veri)
        X_text = [item["hasta"] for item in veri]
        y = [item["rahatsizlik"] for item in veri]
        X_text = [self.clean_text(text) for text in X_text]

        # TF-IDF ve MultinomialNB modeli
        parametreler = {
            'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
            'tfidfvectorizer__use_idf': (True, False),
            'multinomialnb__alpha': [1e-2, 1e-3, 1e-4]
        }

        grid_search = GridSearchCV(self.tfidf_model, parametreler, cv=10, scoring=make_scorer(mean_squared_error), n_jobs=-1)
        grid_search.fit(X_text, y)
        self.tfidf_model = grid_search.best_estimator_

        print(f"Eğitim tamamlandı. En iyi parametreler: {grid_search.best_params_}")
        print(f"En iyi çapraz doğrulama skoru: {grid_search.best_score_}")

        # Derin öğrenme modeli
        self.deep_model = self.build_deep_model(X_text, y)
        self.train_deep_model(X_text, y, epochs=5, batch_size=32)

    def build_deep_model(self, X, y):
        max_words = 10000  # Örneğin, en çok kullanılan 10,000 kelimeyi kullanın
        max_len = 100  # Örneğin, metinlerinizi en fazla 100 kelimeye sınırlayın

        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(X)
        X = pad_sequences(tokenizer.texts_to_sequences(X), maxlen=max_len)

        model = Sequential([
            Embedding(max_words, 32, input_length=max_len),
            LSTM(64),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_deep_model(self, X, y, epochs, batch_size):
        max_words = 10000  # Örneğin, en çok kullanılan 10,000 kelimeyi kullanın
        max_len = 100  # Örneğin, metinlerinizi en fazla 100 kelimeye sınırlayın

        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(X)
        X = pad_sequences(tokenizer.texts_to_sequences(X), maxlen=max_len)

        self.deep_model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        text = text.lower()
        return text

    def modeli_guncelle(self, yeni_veri):
        X_text = [item["hasta"] for item in yeni_veri]
        y = [item["rahatsizlik"] for item in yeni_veri]
        X_text = [self.clean_text(text) for text in X_text]

        # TF-IDF ve MultinomialNB modelini güncelle
        self.tfidf_model.fit(X_text, y)

        # Derin öğrenme modelini güncelle
        X = pad_sequences(tokenizer.texts_to_sequences(X_text), maxlen=max_len)
        self.deep_model.fit(X, y)

    def ozellik_secimi(self, X, y, k):
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        return X_new

    def trymodels(self, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor):
        # Burada modelleri denemek için kullanılan fonksiyonu ekleyin
        pass

    def predict(self, text):
        # Tahmin yaparken her iki modelin çıkışını birleştirin
        tfidf_pred = self.tfidf_model.predict([text])[0]
        deep_pred = self.deep_model.predict(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=max_len))[0][0]
        combined_pred = (tfidf_pred + deep_pred) / 2  # İki modelin çıkışını ortalamak
        return combined_pred

# Veri setinizi yükleyin ve ön işleme adımlarını uygulayın
veri_df = pd.read_csv("dataset.csv")
veri = veri_df.to_dict('records')

# Psikolog sınıfını oluşturun ve eğitimi başlatın
psikolog = dr_psycist()
psikolog.eğit(veri)

# Yeni veri ekleyin ve modeli güncelleyin
yeni_veri = [{"hasta": "Have you ever been smacked by fate?", "hasta_verileri": "NEWDATA", "rahatsizlik": 1}]
psikolog.modeli_guncelle(yeni_veri)
print("Yeni veri ile modeli güncellendi.")

# Örnek bir metin için tahmin yapın
ornek_metin = "How do you feel?"
tahmin = psikolog.predict(ornek_metin)
print(f"Model tahmini: {tahmin}")
