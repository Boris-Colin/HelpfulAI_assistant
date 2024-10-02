import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader  # we can only read pdfs, but there are other loaders (txt)

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

documents = []
docs_path = "docs"
persist_directory = 'InnovationP'  # this is the directory in which we'll store our vector database
embedding_f = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True})

chat_model = ChatOpenAI(
    model_name='gpt-4o',
    temperature=0.3
)

for filename in os.listdir(docs_path):

    if filename.endswith('.pdf'):
        file_path = os.path.join(docs_path, filename)
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())


with open('temporaryF.txt', 'w', encoding='utf-8') as file:
    co = ''
    file.write(co)

with open('temporary.txt', 'w', encoding='utf-8') as file:
    content = ''
    for i in range(len(documents)):
        content += documents[i].page_content
        word_count = len(content.split())
    file.write(content)

k = 1
size = 4000
while size > 2500:
    k += 1
    size = word_count / k

# so now we have even sizes for the rough chunking, we can proceed to chunk.
t_docs = []
with open('temporary.txt', 'r', encoding='utf-8') as file:
    content_T = ''
    for line in file:
        w_count = len(content_T.split())
        if w_count < size:
            # hope this simple condition will be good enough, but if most docs have hyper long lines...
            content_T += line
        else:
            t_docs.append(content_T)
            content_T = ''

    # After the loop, check if there is any remaining content to be added
    if content_T.strip():  # This ensures we don't add empty strings
        t_docs.append(content_T)

print(len(t_docs))

# Now I need to read through the 'rough' chunks to divide them further.
sys_prompt: PromptTemplate = PromptTemplate(
    input_variables=["current_chunk"],
    template="""**System Prompt**: Your task is to read and understand a text, then identify and mark 
    only significant shifts in semantic meaning with the marker (-o-). 
    A significant shift involves a change in the main topic, introduction of a new concept, or a shift to 
    a different aspect of discussion that marks a clear departure from the previous focus. 
    Minor details, elaborations, or explanations within the same overarching topic should not be considered as
     necessitating a new semantic marker.

    1. **Broad Topic Changes**: Mark transitions only when the text moves from one broad topic to another. 

    2. **New Concepts or Ideas**: When a new concept or idea is introduced that changes the direction or focus of 
    the discussion, mark this as a significant shift. 

    3. **Context and Flow**: Consider the overall context and flow of the text. Some shifts might be subtle 
    and require understanding the text as a whole to identify. Not every new sentence or paragraph indicates 
    a semantic shift.

    4. **Avoid Over-Segmentation**: Be cautious of over-segmenting the text. While each sentence can carry unique 
    information, the goal is to identify shifts that represent a significant change in thematic or topical focus.

    Remember, the objective is to enhance comprehension and analysis by segmenting the text based 
    on meaningful transitions in content, not simply to mark the end of sentences or minor topic progressions.
    Think about it step by step: does this sentence need the previous one to be understandable?
    Yes? then group them together.
    **End of System Prompt**
    """)
system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

student_prompt: PromptTemplate = PromptTemplate(
    input_variables=["current_chunk"],
    template="Follow your Instructions and semantically separate: {current_chunk}")
student_message_prompt = HumanMessagePromptTemplate(prompt=student_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, student_message_prompt])

semantic_chain = LLMChain(llm=chat_model, prompt=chat_prompt, output_key='semantic_output')

for i in range(len(t_docs)):
    res = semantic_chain.invoke({'current_chunk': t_docs[i]})
    t_docs[i] = res['semantic_output']

# Now that I have the dividers, I can write them in the second text file.
with open('temporaryF.txt', 'a', encoding='utf-8') as file:
    content = ''
    for j in range(len(t_docs)):
        content += t_docs[j]
    file.write(content)

text_chunks = []
with open('temporaryF.txt', 'r', encoding='utf-8') as file:
    content = file.read()
    # Split the content by '-o-' to separate into chunks
    text_chunks = content.split('-o-')

text_chunks = [chunk.strip() for chunk in text_chunks]


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata


documents = [Document(page_content=chunk) for chunk in text_chunks]

vdb = Chroma.from_documents(
    documents=documents,
    embedding=embedding_f,
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"}
)

vdb.persist()
