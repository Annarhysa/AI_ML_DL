import nltk
from nltk import word_tokenize, pos_tag
import re #regular expressions

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def generate_smart_reply(user_input):
    tokens = word_tokenize(user_input)
    pos_tags = pos_tag(tokens)

    question_pattern = re.compile(r'\b(?:what|when|where|who|how)\b', re.IGNORECASE)
    greeting_pattern = re.compile(r'\b(?:hi|hello|hey)\b', re.IGNORECASE)

    if question_pattern.search(user_input):
        return "I'm not sure. Can you provide more details?"

    if greeting_pattern.search(user_input):
        return "Hello! How can I assist you today?"

    return "I'm not sure how to respond to that."

# User input
user_input = input("Enter a message: ")

# Generate and print the smart reply
smart_reply = generate_smart_reply(user_input)
print("\nGenerated Smart Reply:", smart_reply)