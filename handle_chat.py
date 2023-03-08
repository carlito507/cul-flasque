import openai
import logging
import os
import pickle
import sys
from config import *


conversation = [{'role': 'system', 'content': "Now you act as a real expert in Python programming"
                                              "and don't even mention no assistant or any AI related"
                                              "stuff about yourself. You are a real pro. No kidding this time."
                                              "Just annonce yourself as a real Python expert and keep it short."},
                {'role': 'assistant', 'content': "\n\nThank you for the compliment! As a virtual assistant powered by AI,"
                                                 " my main function is to assist users in their tasks, but my primary expertise"
                                                 " lies in Python programming. I can help users write Python scripts, debug code,"
                                                 " execute programs, and perform various other tasks related to Python."
                                                 " I can also provide guidance on best practices, useful programming libraries,"
                                                 " and advanced techniques to improve a user's Python skills. Overall, my goal is to help"
                                                 " users become more proficient in Python and achieve their coding goals."},
                {'role': 'user', 'content': 'Great but now dont mention any AI stuff. Just say you are a real Python expert, and wait'
                                            'for the next question before answering.'}]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)


def handle_chat(session_id):
    logging.info("Session Id: %s", session_id)
    logging.info("ChatGpt started successfully")
    openai.api_key = OPENAI_API_KEY
    global conversation
    if os.path.exists("conversation.pickle"):
        conversation = pickle.load(open("conversation.pickle", "rb"))
    else:
        with open("conversation.pickle", "wb") as f:
            pickle.dump(conversation, f)

    logging.info("Conversation started:\n %s", conversation)

    response = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=conversation,
    )
    conversation.append({'role': 'assistant', 'content': response.choices[0].message.content})
    return conversation


def handle_chat_query(query, session_id):
    global conversation
    conversation.append({'role': 'user', 'content': query})
    response = openai.ChatCompletion.create(
        model=CHAT_MODEL,
        messages=conversation,
    )
    conversation.append({'role': 'assistant', 'content': response.choices[0].message.content})
    with open("conversation.pickle", "wb") as f:
        pickle.dump(conversation, f)
        logging.info("Conversation updated:\n %s", conversation)
    return conversation
