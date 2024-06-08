import csv
import logging
import numpy as np
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackContext, Filters
from fuzzywuzzy import fuzz
import chardet
from sentence_transformers import SentenceTransformer, util


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

logger = logging.getLogger(__name__)


csv_file = r'C:\Users\eshwa\OneDrive\Desktop\Jala\FSP chatbot\fspfaqcv.csv'


model = SentenceTransformer('distilbert-base-nli-mean-tokens')


def precompute_embeddings(questions):
    question_embeddings = model.encode(questions)
    return question_embeddings

def read_csv_file(csv_file):
    questions = []
    answers = []
    with open(csv_file, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    with open(csv_file, 'r', encoding=encoding) as file:
        reader = csv.reader(file)
        for row in reader:
            questions.append(row[0])
            answers.append(row[1])
    return questions, answers


def get_best_match(question, questions, question_embeddings):
    question_embedding = model.encode([question])[0]
    cosine_similarities = util.cos_sim(question_embedding, question_embeddings)[0]
    best_match_index = np.argmax(cosine_similarities)
    best_match_score = cosine_similarities[best_match_index]
    
    if best_match_score >= 0.75:
        return questions[best_match_index]
    else:
        return None


def start_command(update: Update, context: CallbackContext):
    update.message.reply_text('Hi! I am your FAQ bot. How can I assist you today?')


def handle_message(update: Update, context: CallbackContext):
    question = update.message.text
    questions, answers = context.bot_data['faq_data']
    question_embeddings = context.bot_data['faq_embeddings']
    best_match = get_best_match(question, questions, question_embeddings)

    if best_match:
        index = questions.index(best_match)
        answer = answers[index]
        update.message.reply_text(answer)
    else:
        default_answer = "I'm really sorry, I couldn't understand your question. Please provide more information."
        update.message.reply_text(default_answer)


def main():
    
    updater = Updater("6207626491:AAGcExpNfJLSktcgD97GwnvZQnC4YUs_SEY")

    
    dispatcher = updater.dispatcher

    
    questions, answers = read_csv_file(csv_file)
    question_embeddings = precompute_embeddings(questions)

    
    dispatcher.bot_data['faq_data'] = (questions, answers)
    dispatcher.bot_data['faq_embeddings'] = question_embeddings

    
    dispatcher.add_handler(CommandHandler("start", start_command))


    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    
    updater.start_polling()

    
    updater.idle()


if __name__ == '__main__':
    main()


# 6207626491:AAGcExpNfJLSktcgD97GwnvZQnC4YUs_SEY
