# LLM as a group recommender system
from signal import signal, alarm, SIGALRM # TODO: Solo se puede importar en UNIX!
import time

class Timeout:
    def __init__(self, seconds=1, message="Timed out"):
        self._seconds = seconds
        self._message = message

    @property
    def seconds(self):
        return self._seconds

    @property
    def message(self):
        return self._message
    
    @property
    def handler(self):
        return self._handler

    @handler.setter
    def handler(self, handler):
        self._handler = handler

    def handle_timeout(self, *_):
        raise TimeoutError(self.message)

    def __enter__(self):
        self.handler = signal(SIGALRM, self.handle_timeout)
        alarm(self.seconds)
        return self

    def __exit__(self, *_):
        alarm(0)
        signal(SIGALRM, self.handler)    
        

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import FakeListLLM
from langchain_community.chat_models import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever

from langchain.prompts import ChatPromptTemplate

from langchain_community.callbacks import get_openai_callback

import pandas as pd
import tiktoken
import os

import pickle

import numpy as np
from tqdm import tqdm

import datetime

from io import StringIO
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatCohere

import sys
import time

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


def num_tokens_from_string(string: str, encoding_name: str ="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
import random
random.seed(42)

import ollama
   
def generate(llm,prompt,dict_query,stereotype=None) -> str: # vamos a ninjearla, a ver si así se mantienen las respuestas !!
    
    dict_format = {'users_history': dict_query['users_history'],
                   'to_recommend':dict_query['to_recommend']}
                   
    if not isinstance(dict_format['to_recommend'],str):
        
        aa = sorted(dict_query['to_recommend'])
        dict_format['to_recommend'] = '"' + '", "'.join(aa) + '"'
                                      
    if 'intersection' in dict_query and len(dict_query['intersection']) > 0:
        dict_format['intersection'] = 'This is the "group preferences": ' + dict_query['intersection']
    else:
        dict_format['intersection'] = 'There are no common "group preferences."'
        
    if stereotype is not None:
        dict_format['stereotype'] = 'In addition, we have the following information for some of the users:\n ' + stereotype
    else:
        dict_format['stereotype'] = ' '
    
    model_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_TEMPLATE), # system role
        ("human",  prompt) # human, the user text  
        ])

    message = model_prompt.format(**dict_format)
    
    
    print('-----------------------------------------------------')
    nn = num_tokens_from_string(message)
    print(num_tokens_from_string(message), "tokens (approx.)")
    
    if nn > 20000:
        print('Context too long!')
        return ''
    
    chain = model_prompt | llm 
    response = chain.invoke(dict_format)

    print("Reponse:", response)
    print('-----------------------------------------------------')

    return response
   
my_models = {}

# my_models["mistral:7b-instruct"] = ChatOllama(
#     model="mistral:7b-instruct", temperature=0.0, #format="str",
#     # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )

# my_models["gemma:7b"] = ChatOllama(
#     model="gemma:7b", temperature=0.0, #format="str",
#     # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )

# my_models["gemma:2b"] = ChatOllama(
#     model="gemma:2b-instruct", temperature=0.0, #format="str",
#     cache = False,
#     cache_prompt=False,
#     # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )

# my_models["llama3"] = ChatOllama(
    # model="llama3", temperature=0.0, #format="str",
    # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )

# my_models["llama2:7b-chat"] = ChatOllama(
#     model="llama2:7b-chat", temperature=0.0, #format="str",
#     # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )


os.environ["OPENAI_API_KEY"] = "TODO"
my_models['gpt-3.5-turbo'] = ChatOpenAI(
    model='gpt-3.5-turbo', temperature=0.0, #format="str",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)



SYSTEM_TEMPLATE_v0 = """You are a helpful movie recommender system. Your task is to recommend movies to a group of people based on their waching history.
You will receive:
    * The group watching history.
    * The individual user watching history.
    * User information (if available).
    * The set of movies to potentially recommend.

Your task is to use all the provided information to generate a list of recommended movies. You have access to all the information you need.
"""

SYSTEM_TEMPLATE = """You are a helpful movie recommender system. Your task is to recommend movies to a group of people based on their waching history.
You will receive:
    * The group preferences.
    * The individual user preferences.
    * User information (if available).
    * The set of movies to recommend.

Your task is to use all the provided information to generate a list of recommended movies. You have access to all the information you need.
"""

PROMPT_v0 = """
{intersection}

Also, here are the last movies watched by each user:
{users_history}

{stereotype}

Movies to recommend: {to_recommend}

Your task is:
1. From the list "movies to recommend", pick 10 movies that would satisfy the group as a whole and rank them. Use only the movies in the list, do not add any additional movie. Do not change the given movies.
2. Return your answer in a JSON format including the key 'movies' and a list with the ranked movies.

Your JSON answer:
"""

PROMPT_v1 = """
{intersection}

Also, here are the last movies watched by each user:
{users_history}

{stereotype}

Movies to recommend: {to_recommend}

Your task is:
1. Rank the "movies to recommend" based on how well they would satisfy the group as a whole. Rank 1 should be the movie that best satisfies the group. Please, use only the movies in the list, do not add any additional movie. Do not change the given movies. Do not change the given titles.
2. Return your answer in a JSON format including the key 'movies' and the list with the ranked movies.

Your JSON answer:
"""

PROMPT_v2 = """
{intersection}

Also, here are the last movies watched by each user:
{users_history}

{stereotype}

Movies to recommend: {to_recommend}

Your task is:
1. Rank the "movies to recommend" based on how well they would satisfy the group as a whole. Rank 1 should be the movie that best satisfies the group. Please, use only the movies in the list, do not add any additional movie. Do not change the given movies. Do not change the given titles.
2. Return your answer in a JSON format including the key 'movies' and the list with the ranked movies.

Note that "movies to recommend" is not sorted. Use only "movies to recommend". Do not add any extra movie. 

Your JSON answer:
"""

PROMPT_v3 = """
{intersection}

Also, here are the last movies watched by each user:
{users_history}

{stereotype}

Movies to recommend: {to_recommend}

Your task is:
1. Pick 10 movies from "movies to recommend" and sort them based on how well they would satisfy the group as a whole. Position 1 should be the movie that best satisfies the group. Please, use only the movies in the list, do not add any additional movie. Do not change the given movies. Do not change the given titles.
2. Return your answer in a JSON format including the key 'movies' and the list with the ranked movies.

Note that "movies to recommend" is alphabetically sorted, and that order those not reflect the group preferences. Use only "movies to recommend". Do not add any extra movie. 

Your JSON answer:
"""

PROMPT_v4 = """
{intersection}

These are the "individual user watching history":
{users_history}

{stereotype}

Movies to recommend: {to_recommend}

Your task is:
1. Using the "group watching history" and the "individual user watching history", pick 10 movies from "movies to recommend" and sort them based on how well they would satisfy the group as a whole. Position 1 should be the movie that best satisfies the group. Please, use only the movies in the list, do not add any additional movie. Do not change the given movies. Do not change the given titles.
2. Return your answer in a JSON format including the key 'movies' and the list with the ranked movies.

Note that "movies to recommend" is alphabetically sorted, and that order those not reflect the group preferences. Use only "movies to recommend". Do not add any extra movie. 

Your JSON answer:
"""

PROMPT_v5 = """
{intersection}

These are the "individual user watching history":
{users_history}

{stereotype}

Movies to recommend: {to_recommend}

Your task is:
1. Using the "group watching history" and the "individual user watching history", pick 10 movies from "movies to recommend" and sort them based on how well they would satisfy the group as a whole. Position 1 should be the movie that best satisfies the group. Please, use only the movies in the list, do not add any additional movie. Do not change the given movies. Do not change the given titles.
2. Return your answer in a JSON format including the key 'movies' and the list with the ranked movies.

All the information you need is available in this conversation, focus on the "group watching history", the "individual user watching history", the "movies to recommend" and the "user information" if provided.
Note that "movies to recommend" is alphabetically sorted, and that order those not reflect the group preferences. Use only "movies to recommend". Do not add any extra movie. 

Your JSON answer:

"""

PROMPT = """
{intersection}

These are the "individual user preferences":
{users_history}

{stereotype}

Movies to recommend: {to_recommend}

Your task is:
1. Using the "group preferences" and the "individual user preferences", pick 10 movies from "movies to recommend" and sort them based on how well they would satisfy the group as a whole. Position 1 should be the movie that best satisfies the group. Please, use only the movies in the list, do not add any additional movie. Do not change the given movies. Do not change the given titles.
2. Return your answer in a JSON format including the key 'movies' and the list with the ranked movies.

All the information you need is available in this conversation, focus on the "group preferences", the "individual user preferences", the "movies to recommend" and the "user information" if provided.
Note that "movies to recommend" is alphabetically sorted, and that order those not reflect the group preferences. Use only "movies to recommend". Do not add any extra movie. 

Your JSON answer:

"""


dir_path = './'

for model, llm in my_models.items():

    for query_file in os.listdir(dir_path):
        
        if not query_file.startswith('groups_') or '__queries' not in query_file:
            continue
        
        if '_divergent-group-combinatorial' not in query_file:
            continue

        if '1000g_5u' not in query_file:
            continue

        if '_mcr5' in query_file:
            continue

        print('---------------',query_file)
        
        group_query = pd.read_pickle(dir_path + query_file)
        
        results = []
        results_file = dir_path + 'results___' + model.replace(':','') + '__' + query_file
        print('__________',results_file)
        
        if os.path.exists(results_file):
            results = pd.read_pickle(results_file)
        
        # for i in tqdm(range(0,len(results))): # hacemos el chequeo de los que ya se corrieron!
        #     for j in tqdm(range(0,len(results[i]))):
        #         rr = results[i][j][1]
        #         rr = rr.content if not isinstance(rr,str) else rr
        #         if 'am unable to generate a JSON' not in rr: 
        #             continue
        #         print('Retrying... ')
        #         group = group_query[i]
        #         stereotype = group_query[i]['stereo_combinations'][j]
                
        #         with Timeout(90):
        #             try:
        #                 response = generate(llm,PROMPT,group,stereotype[1]) 
        #             except Exception as e:
        #                print('Timeout error!!', e)
        #                response = str({'movies':[]})
        #         results[i][j] = (stereotype[0],response) # update the position
        
        #     with open(results_file,'wb') as file: # salvamos por cada grupo
        #         pickle.dump(results,file)   
        
        how_many = min(100,len(group_query))

        for i in tqdm(range(len(results),how_many)): 
            
            group = group_query[i]
            results_g = [] 
            for stereotype in tqdm(group['stereo_combinations']):
                with Timeout(90):
                    try:
                        response = generate(llm,PROMPT,group,stereotype[1]) 
                    except Exception as e:
                       print('Timeout error!!', e)
                       response = str({"movies":[]})
                results_g.append((stereotype[0],response))
                
            results.append(results_g) 
            
            with open(results_file,'wb') as file: 
                pickle.dump(results,file)   