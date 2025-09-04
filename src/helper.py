from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from src.prompt import *
from langchain.text_splitter import split_text_on_tokens, Tokenizer
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings

# Load HUGGINGFACEHUB_API_TOKEN
load_dotenv()
token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set!")



def file_processing(file_path):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content
        

    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id, max_length=512)

    def encode(text):
        return tokenizer.encode(text)

    def decode(tokens):
        return tokenizer.decode(tokens, skip_special_tokens=True)

    tokenizer_local = Tokenizer(
        tokens_per_chunk=500,
        chunk_overlap=50,
        encode=encode,
        decode=decode
    )

    chunks_ques_gen = split_text_on_tokens(text=question_gen, tokenizer=tokenizer_local)


    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]


    document_answer_gen = document_ques_gen.copy()

    return document_ques_gen, document_answer_gen



def llm_pipeline(file_path):

    document_ques_gen, document_answer_gen = file_processing(file_path)

    model_id = 'google/flan-t5-large'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto')
    ques_gen_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=450)
    llm_ques_gen_pipeline = HuggingFacePipeline(pipeline=ques_gen_pipeline)

   

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline, 
                                            chain_type = "refine", 
                                            verbose = True, 
                                            question_prompt=PROMPT_QUESTIONS, 
                                            refine_prompt=REFINE_PROMPT_QUESTIONS)

    ques_list = []
    for chunk in document_ques_gen:
        ques = ques_gen_chain.run([chunk])
        ques_list.append(ques)

    

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    model_id = 'google/flan-t5-large'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto')
    answer_gen_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=450)
    llm_answer_gen = HuggingFacePipeline(pipeline=answer_gen_pipeline)



    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
                                                chain_type="stuff", 
                                                retriever=vector_store.as_retriever())

    return answer_generation_chain, filtered_ques_list


