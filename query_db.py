import os

import openai
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def create_db():
  embedding_function = OpenAIEmbeddings()
  db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
  return db

def query_db(db, query_text):
  results = db.similarity_search_with_relevance_scores(query_text, k=3)
  if len(results) == 0 or results[0][1] < 0.7:
    print(f"Unable to find matching results.")
    return
  
  context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=context_text, question=query_text)
  print(prompt)

  model = ChatOpenAI()
  response_text = model.predict(prompt)
  
  sources = [doc.metadata.get("source", None) for doc, _score in results]
  formatted_response = f"Response: {response_text}\nSources: {sources}"
  print(formatted_response)


if __name__ == "__main__":
  db = create_db()
  query = input()
  query_db(db, query)
