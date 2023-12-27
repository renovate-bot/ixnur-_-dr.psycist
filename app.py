"""bana 3 nas 3'de felak yolla cano"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class Psikolog:
    def __init__(self):
        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        self.load_model()

    def load_model(self):
        try:
            self.model = joblib.load("trained_model.joblib")
            print("Model başarıyla yüklendi.")
        except FileNotFoundError:
            print("Eğitilmiş model bulunamadı. Lütfen önce modeli yükleyin!")

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        return text

    def predict_emotion(self, text):
        text = self.clean_text(text)
        prediction = self.model.predict([text])
        return prediction[0]

class TerapiUygulamasi:
    def __init__(self):
        self.psikolog = Psikolog()
        self.questions = [
            "Nasıl hissediyorsunuz?",
            "Bu duyguyu neyin tetiklediğini düşünüyorsunuz?",
            "Bu duyguyu daha önce yaşadınız mı?",
            "Fiziksel bir rahatsızlığın varmı? varsa bu yaşantını etkiliyormu?"
            "Duygu durumunuz hakkında daha fazla detay verebilir misiniz?",    
            "Duygu durumunuzu etkileyen bir olay oldu mu? Eğer evetse, lütfen anlatın.",
            "Bu hissi daha önce yaşadınız mı? Eğer evetse, bu durumda size nasıl yardımcı oldunuz?",
            "Durumunuz hakkında başka paylaşmak istediğiniz bir şey var mı?",
            "Bu duyguyu daha önce yaşadınız mı?",
            "Durumunuz hakkında başka neler paylaşabilirsin?",
            "Bu hissi daha önce yaşadınız mı? Eğer evetse, bu durumda size nasıl yardımcı oldunuz?",
            "Bu durumun seni nasıl etkilediğini anlatırmısın?"
            "Genel olarak şu anki problemlerine, gelecektekilere nazaran daha fazla önem verirsin.",
            "Duygularını ve hislerini sık sık ve de kolayca ifade edebilirsin.",
            "Sosyal çevrende nasıl hissedersin.",
            "Aile yaşantında rahatmı hissediyorsun "
            "Fikrin anlaşılma sürecinin detaylarından çok fikrin geneliylemi ilgilenirsin.",
            "Alternatif çözümleri denemektense, kendi tecrübelerine güvenmeye daha mı yatkınsın.",
            "Sık sık insanlık ve geleceğimiz hakkında kafa patlatırmısın.",    
        ]
        self.max_question_attempts = 5

    def get_user_input(self, question):
        return input(question + " ")
    
    def analyzeresponse(self, response):
        try:
            memory_score = float(response)
            if 0 <= memory_score <= 10:
                print(f"Teşekkür ederiz! Geçmişteki olayları hatırlama konusundaki puanınız: {memory_score}")
            else:
                print("Lütfen 0 ile 10 arasında bir sayı girin.")
        except ValueError:
            print("Geçerli bir sayı girmediğinizden emin olun.")
        sentiment_score = self.calculate_sentiment_score(response)
        self.provide_specific_feedback(sentiment_score)
        print(f"Duygu durumunuz hakkında daha fazla detay verebilir misiniz? {sentiment_score}")

    def calculate_sentiment_score(self, response):
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(response)
        return sentiment_scores
    
    def provide_specific_feedback(self, sentiment_score):
        if sentiment_score >= 0.5:
            print("Harika! Pozitif bir duygu durumundasınız.")
        elif 0.2 <= sentiment_score < 0.5:
            print("Duygu durumunuz ılımlı. Belki bir hobiye zaman ayırmak iyi olabilir.")
        elif -0.2 <= sentiment_score < 0.2:
            print("Duygu durumunuz nötr. Her şey normal görünüyor.")
        elif -0.5 <= sentiment_score < -0.2:
            print("Olumsuz bir duygu durumundasınız. Rahatsız olduğunuz konuları düşünmek önemli olabilir.")
        else:
            print("Çok olumsuz bir duygu durumundasınız. Lütfen bir uzmana danışmayı düşünün.")

        emotion_intensity = len(re.findall(r'\b(stresli|üzgün|mutlu|sinirli)\b', response, flags=re.IGNORECASE))

        if emotion_intensity >= 2:
            print("Duygusal olarak yoğun bir durumdasınız. Bir uzmana danışmayı düşünebilirsiniz.")
        elif emotion_intensity == 1:
            print("Duygusal bir durumdasınız. Bu duyguyu anlamaya çalışmak önemlidir.")
        else:
            print("Duygu durumunuz normal görünüyor. Ancak, her zaman konuşmak iyidir.")

    def run_therapy_session(self):
        print("Hoş geldiniz! Terapi oturumumuza başlamadan önce birkaç soru sormak isteriz.")

        for question in self.questions:
            attempts = 0
            while attempts < self.max_question_attempts:
                user_input = self.get_user_input(question)
                if user_input.strip() != "":
                    self.analyze_user_response(user_input)
                    break
                else:
                    print("Lütfen bir cevap girin.")
                    attempts += 1

        print("Terapi oturumu tamamlandı. Teşekkür ederiz.")

if __name__ == "__main__":
    uygulama = TerapiUygulamasi()
    uygulama.run_therapy_session()
