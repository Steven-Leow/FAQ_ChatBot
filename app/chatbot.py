# import nltk
# import difflib
# from random import choice
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# # Function to match keywords with NLTK processing
# def match_keyword(user_input):
#     # Tokenize input and remove stop words
#     stop_words = set(stopwords.words('english'))
#     tokens = word_tokenize(user_input)
#     filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
#     print(filtered_tokens)
    
#     # Flatten synonyms into one list
#     all_keywords = {word: key for key, words in synonyms.items() for word in words}

#     for token in filtered_tokens:
#         close_matches = difflib.get_close_matches(token, all_keywords.keys(), n=1, cutoff=0.8)
#         if close_matches:
#             return all_keywords[close_matches[0]]
    
#     return None

import json
import torch
from transformers import pipeline

class Chatbot:
    def __init__(self, faq_data_path: str):

        # Load the candidate labels
        with open(faq_data_path, 'r') as file:
            self.candidate_labels = json.load(file)

        # Extract the categories and topics
        self.categories = list(self.candidate_labels.keys())
        self.topics = {category: list(details.keys()) for category, details in self.candidate_labels.items()}

        # Load the intent classification pipeline
        self.device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=self.device)

    def classify_intent(self, user_question: str):
        # Classify category first
        category_result = self.classifier(user_question, self.categories)
        category = category_result['labels'][0]
        category_confidence = category_result['scores'][0]
        print(f"Category: {category}, Confidence: {category_confidence}")

        if category_confidence < 0.8:
            return None, None

        # Classify topic within identified category
        topic_result = self.classifier(user_question, self.topics[category])
        topic = topic_result['labels'][0]
        topic_confidence = topic_result['scores'][0]
        print(f"Topic: {topic}, Confidence: {topic_confidence}")

        if topic_confidence < 0.8:
            return None, None

        # Return category and topic
        return category, topic

    def get_answer(self, category: str, topic: str) -> str:
        return self.candidate_labels[category][topic]

    def run(self):
        print("Welcome to the FAQ Chatbot! Type 'exit' to quit.")

        while True:
            user_input = input("You: ").lower()

            if user_input == 'exit':
                print("Goodbye! Have a nice day.")
                break

            # Classify the intent of the user's input
            category, topic = self.classify_intent(user_input)

            if category is None or topic is None:
                print("Bot: Sorry, I couldn't understand your question. Please try rephrasing your question.")
            else:
                # Fetch the answer based on classified category and topic
                answer = self.get_answer(category, topic)
                print(f"Bot: {answer}")

# Initialize the chatbot with the path to the FAQ data
chatbot = Chatbot(faq_data_path='data/faq_data.json')

# Run the chatbot
if __name__ == "__main__":
    chatbot.run()
