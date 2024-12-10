import os
import shutil

import openai
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']


DATA_PATH = "data/text"
CHROMA_PATH = "chroma"

def load_documents():
  loader = DirectoryLoader(DATA_PATH, glob="*.md")
  documents = loader.load()
  return documents

def split_text(documents):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
  )

  chunks = text_splitter.split_documents(documents)
  return chunks

def save_to_chroma(chunks):
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

  db = Chroma.from_documents(
    chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH,
  )
  db.persist()
  print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
  documents = load_documents()
  chunks = split_text(documents)
  save_to_chroma(chunks)
