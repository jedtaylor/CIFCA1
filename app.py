

import os 

from tqdm import tqdm

import huggingface_hub

import streamlit as st 

import PyPDF2

from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv

import openllm

import pinecone

load_dotenv()

# streamlit bits
st.title('CIFCA')
prompt = st.text_input('Plug in your prompt here') 


# prompt templates
template1 = PromptTemplate(
    input_variables=["topic"],
    template="Briefly tell me about: {topic}"
    )

template2 = PromptTemplate(
    input_variables=["topic"],
    template="Give me an overview of: {topic}"
    )

# LLMs
llm = HuggingFaceHub(
    repo_id="gpt2-xl",
    model_kwargs={'temperature': 0.2, 'max_length': 20}
)

chain1 = LLMChain(llm=llm, prompt=template1, verbose=True, output_key='one')
chain2 = LLMChain(llm=llm, prompt=template2, verbose=True, output_key='two')

if prompt:
    response1 = chain1.run(prompt)
    response2 = chain2.run(prompt)
    st.write("one:", response1)
    st.write("two:", response2)