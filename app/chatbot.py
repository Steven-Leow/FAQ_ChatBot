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
