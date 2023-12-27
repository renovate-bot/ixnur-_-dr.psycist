#                بِسْــــــــــــــــــمِ اﷲِالرَّحْمَنِ اارَّحِيم
# لِيَجْزِيَهُمُ اللّٰهُ اَحْسَنَ مَا عَمِلُوا وَيَزٖيدَهُمْ مِنْ فَضْلِهٖؕ وَاللّٰهُ يَرْزُقُ مَنْ يَشَٓاءُ بِغَيْرِ حِسَابٍ 

import re
import numpy as np
import joblib
from nltk.sentiment import SentimentIntensityAnalyzer

class AdvancedQLearningPsychologicalTherapist:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        """
        Advanced Q-learning tabanlı bir psikolog sınıfı.

        Parameters:
            learning_rate (float): Q-table güncelleme hızı.
            discount_factor (float): Gelecekteki ödüllerin şu anki ödüllerden ne kadar indirgeneceği.
            exploration_rate (float): Keşfetme / sömürme stratejisi için oran.
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.actions = ["action1", "action2"]  
        self.q_table = np.zeros((num_states, num_actions)) 

    def update_q_table(self, state, action, next_state, reward):
        """
        Q-tablosunu güncelleme işlemi.

        Parameters:
            state (int): Şu anki durum.
            action (int): Seçilen eylem.
            next_state (int): Gelecekteki durum.
            reward (float): Alınan ödül.
        """
        current_q_value = self.q_table[state, action]
        max_future_q_value = np.max(self.q_table[next_state, :])

        # Daha sofistike bir Q-table güncelleme stratejisi ekleyin
        new_q_value = self.learning_rate * (reward + self.discount_factor * max_future_q_value - current_q_value)
        self.q_table[state, action] += new_q_value

    def explore_exploit_strategy(self, state):
        """
        Keşfetme ve sömürme stratejisi.

        Parameters:
            state (int): Şu anki durum.

        Returns:
            str: Seçilen eylem.
        """
        # Softmax stratejisi ekleyin
        action_probabilities = np.exp(self.q_table[state, :] / self.exploration_rate)
        action_probabilities /= np.sum(action_probabilities)
        chosen_action = np.random.choice(self.actions, p=action_probabilities)
        return chosen_action

    def run_therapy_session(self):
        """
        Terapi oturumu yönetimi.
        """
        pass

    def analyze_user_response(self, response):
        """
        Kullanıcı cevabını analiz etme.

        Parameters:
            response (str): Kullanıcının cevabı.
        """
        pass

    def calculate_sentiment_score(self, response):
        """
        Duygu skoru hesaplama.

        Parameters:
            response (str): Kullanıcının cevabı.

        Returns:
            dict: Sentiment skorları.
        """
        pass

    def provide_specific_feedback(self, sentiment_score):
        """
        Duygu durumu analizine dayalı spesifik geri bildirim sağlama.

        Parameters:
            sentiment_score (dict): Sentiment skorları.
        """
        pass


class TerapiUygulamasi:
    def __init__(self):
        self.psikolog = AdvancedQLearningPsychologicalTherapist()
        self.questions = [
            #sorular ...
            "Genel olarak şu anki problemlerine, gelecektekilere nazaran daha fazla önem verirsin.",
            "Duygularını ve hislerini sık sık ve de kolayca ifade edebilirsin.",
            #sorular ...
        ]
        self.max_question_attempts = 5

    def get_user_input(self, question):
        """
        Kullanıcıdan giriş almak.

        Parameters:
            question (str): Soru metni.

        Returns:
            str: Kullanıcının girişi.
        """
        return input(question + " ")

    def analyze_user_response(self, response):
        """
        Kullanıcı cevabını analiz etme.

        Parameters:
            response (str): Kullanıcının cevabı.
        """
        self.psikolog.analyze_user_response(response)

    def run_therapy_session(self):
        """
        Terapi oturumu yönetimi.
        """
        pass


if __name__ == "__main__":
    uygulama = TerapiUygulamasi()
    uygulama.run_therapy_session()


""" def update_q_table(self, state, action, next_state, reward):
        current_q_value = self.q_table[state, action]
        max_future_q_value = np.max(self.q_table[next_state, :])
        new_q_value = self.learning_rate * (reward + self.discount_factor * max_future_q_value - current_q_value)
        self.q_table[state, action] += new_q_value
    def explore_exploit_strategy(self, state):
        action_probabilities = np.exp(self.q_table[state, :] / self.exploration_rate)
        action_probabilities /= np.sum(action_probabilities)
        chosen_action = np.random.choice(self.actions, p=action_probabilities)
        return chosen_action
    
  class AdvancedQLearningPsychologicalTherapist:
    def __init__(self, questions, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.questions = questions
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((len(questions), len(actions)))
        self.nlp = spacy.load("en_core_web_sm")

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.actions)
        else:
            return self.actions[np.argmax(self.q_table[state, :])]

    def update_q_table(self, state, action, next_state, reward):
        current_q_value = self.q_table[state, action]
        max_future_q_value = np.max(self.q_table[next_state, :])
        new_q_value = (1 - self.learning_rate) * current_q_value + \
                       self.learning_rate * (reward + self.discount_factor * max_future_q_value)
        self.q_table[state, action] = new_q_value

    def get_feedback(self, user_response):
        doc = self.nlp(user_response)

        positive_words = ["mutlu", "harika", "iyi"]
        negative_words = ["üzgün", "kötü", "sıkıntılı"]

        sentiment_score = doc.sentiment.polarity

        for token in doc:
            if token.text.lower() in positive_words:
                sentiment_score += 0.2
            elif token.text.lower() in negative_words:
                sentiment_score -= 0.2

        if sentiment_score >= 0.5:
            return "Terapist: Bu harika! Pozitif bir duygu durumundasınız."
        elif 0.2 <= sentiment_score < 0.5:
            return "Terapist: Duygu durumunuz ılımlı. Belki bir hobiye zaman ayırmak iyi olabilir."
        elif -0.2 <= sentiment_score < 0.2:
            return "Terapist: Duygu durumunuz nötr. Her şey normal görünüyor."
        elif -0.5 <= sentiment_score < -0.2:
            return "Terapist: Olumsuz bir duygu durumundasınız. Rahatsız olduğunuz konuları düşünmek önemli olabilir."
        else:
            return "Terapist: Çok olumsuz bir duygu durumundasınız. Lütfen bir uzmana danışmayı düşünün."

    def simulate_therapy(self, epochs):
        for epoch in range(epochs):
            current_state = np.random.randint(len(self.questions))
            action = self.choose_action(current_state)

            user_response = input(f"Terapist: {self.questions[current_state]} ")
            therapist_feedback = self.get_feedback(user_response)

            print(therapist_feedback)

            next_state = np.random.randint(len(self.questions))
            # Burada kullanıcının cevabına göre bir ödül belirleyebilirsiniz.
            reward = 1 if "harika" in therapist_feedback.lower() else -1

            self.update_q_table(current_state, action, next_state, reward)

questions = [
    "Nasıl hissediyorsunuz?",
    "Bu duyguyu neyin tetiklediğini düşünüyorsunuz?",
    "Bu duyguyu daha önce yaşadınız mı?",
    "Fiziksel bir rahatsızlığınız var mı? Varsa, bu durum yaşantınızı etkiliyor mu?",
    #diğer sorular ...
]

actions = ["Empathize", "Reflect", "Encourage", "Question"]

therapist = AdvancedQLearningPsychologicalTherapist(questions, actions)
therapist.simulate_therapy(1000)
        
           def update_q_table(self, state, action, next_state, reward):
        current_q_value = self.q_table[state, action]
        max_future_q_value = np.max(self.q_table[next_state, :])
        new_q_value = (1 - self.learning_rate) * current_q_value + \
                       self.learning_rate * (reward + self.discount_factor * max_future_q_value)
        self.q_table[state, action] = new_q_value
        
        
        
        
        
        """
