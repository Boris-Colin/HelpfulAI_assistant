import os
import sys
import re
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.chains import TransformChain, SequentialChain

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain_community.llms import Ollama
from ollama import Client

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

current_memo_path = "chat_history.txt"
full_memo_path = "Conversation.txt"

user_response = input("Would you like to continue your previous conversation? (yes/no): ")
if user_response.lower() == 'no':
    # we need to reset both files
    with open(full_memo_path, 'w') as file:
        file.write("[0]\n\n")

    with open(current_memo_path, 'w') as file:
        file.write("")

# Ok, this is where I'll write the part dealing with the last exchange
# this part is only supposed to return the list with the number of all the exchanges in Conversation.txt
exchange_numbers = []
exchanges_text = []
with open(full_memo_path, 'r') as file:
    j = -1
    for line in file:
        if line.startswith('['):
            number = int(line.strip()[1:-1])
            exchange_numbers.append(number)
            j = j + 1
            exchanges_text.append('')
        if line.strip():
            line = re.sub(r'(\r\n|\r|\n)+', ' ', line)  # Replace newlines with spaces
            exchanges_text[j] += line
# it works, the list is well done
# I'll just need to set len(exchange_numbers) at the end
# but here I'm getting the three last exchanges: (slicing is flexible so no need to account for first iterations)
last_elements = ' '.join(exchanges_text[-3:])


persist_directoryT = 'To_dest'
embedding_f = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True})
vectorstore = Chroma(persist_directory=persist_directoryT, embedding_function=embedding_f)

chat_model = ChatOpenAI(
    model_name='gpt-3.5-turbo-16k',
    temperature=0.2
)

sys_prompt: PromptTemplate = PromptTemplate(
    input_variables=["output_text", "user_query"],
    template="""**System Prompt** Your role is to  respond to queries based solely on the information provided 
    to you. That information will start with **Start of Context** and ends at **End of Context**. 
    The rest is the prompt or query.

    It is essential to rely strictly on this context, without drawing on any external knowledge.
    Adhere to the following guidelines when crafting your responses:
    
    1. **Directly Relevant Queries**: If the query is specific and DIRECTLY related to the information within 
    your provided context, provide an answer that is detailed and relevant. If not it is fine to say that you don't know.

    2. **General or Ambiguous Queries**: If the query is general, ambiguous, or seems to imply a need for context 
    beyond what is provided (e.g., "Could you provide more details about that?" without specifying "that"), 
    acknowledge this explicitly. Indicate that you are unable to provide a detailed response based on the current context.
    The same is true if the user mentions a past conversation or a previous exchange. 
    YOU DO NOT HAVE TO ANSWER if you are unsure. it is fine if you do not know. In fact it is worse if you try
    to answer a general question.

    This refined approach ensures you effectively communicate the limitations of the provided context and 
    guide users to where they might find a more complete answer, thus improving the user experience 
    by setting clear expectations. **End of System Prompt**
    """)

system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

student_prompt: PromptTemplate = PromptTemplate(
    input_variables=["output_text", "user_query"],
    template="Using only {output_text} answer {user_query}. Follow the Guidelines!")

student_message_prompt = HumanMessagePromptTemplate(prompt=student_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, student_message_prompt])


def transform_func(inputs: dict) -> dict:
    documents = inputs["original_sentence"]
    n = len(documents)
    cleaned_texts = "**Start of Context**"
    # I might add new things to delete later to improve the cleaning process.
    i = 0
    for document, score in documents:
        page_content = document.page_content
        page_content = re.sub(r'(\r\n|\r|\n)+', ' ', page_content)  # Replace newlines with spaces
        page_content = re.sub(r'[ \t]+', ' ', page_content)  # Collapse multiple spaces/tabs into one
        if i == n:
            cleaned_texts += page_content + "**End of Context**"
        else:
            cleaned_texts += page_content + " "
    cleaned_texts = cleaned_texts.strip()
    return {"output_text": cleaned_texts}


clean_extra_spaces_chain = TransformChain(
    input_variables=["original_sentence"],
    output_variables=["output_text"],
    transform=transform_func)

style_paraphrase_chain = LLMChain(llm=chat_model, prompt=chat_prompt, output_key='final_output')

# I wonder if I can put two SequentialChains together inside another SequentialChain
seq_chain = SequentialChain(
    chains=[clean_extra_spaces_chain, style_paraphrase_chain],
    input_variables=['original_sentence', 'user_query'],
    output_variables=['final_output'])
# And that's it for the 'retrieval' chain. Below starts the main chat chain

# I'll have to format the provided context as well as both histories.
sys_prompt2: PromptTemplate = PromptTemplate(
    input_variables=["final_output", "user_query", "current_chat_history"],
    template="""**System Prompt** Your role is to respond to queries based on the information provided 
    to you. There will be two sources of info: the chat history, delimited by **Chat history** and **/Chat history** 
    and the context indicated by **Start Context** and **End Context**.
    The rest is the prompt or query.

    It is essential to rely strictly on both context and chat history, without drawing on any external knowledge.
    Adhere to the following guidelines when crafting your responses:

    1. **Conflict**: If the context and the chat history are in disagreement, focus on the chat history.

    2. **History Queries**: The context you receive comes from another model without access to the chat history.
    Therefore you might get no context, or context about how the previous model cannot help when the user
    asks questions referring to past interactions with you.

    3. **Wrong Context** If you consider that the context provided does not answer the question at all
    , ignore the context. Do not provide,information that is not in your chat history nor context. 
    It is fine not to answer in those case.

    This refined approach ensures you effectively communicate the limitations of the provided context and 
    guide users to where they might find a more complete answer, thus improving the user experience 
    by setting clear expectations. **End of System Prompt**
    """)
system_prompt2 = SystemMessagePromptTemplate(prompt=sys_prompt2)

# it might be useful to decompose current_chat_history into current_chat_history and latest_exchanges
student_prompt2: PromptTemplate = PromptTemplate(
    input_variables=["final_output", "user_query", "current_chat_history"],
    template="""Using only **Start Context** {final_output} **End Context** and 
    **Chat history** {current_chat_history} **/Chat history**; answer {user_query}. Follow the Guidelines!""")

student_prompt_2 = HumanMessagePromptTemplate(prompt=student_prompt2)
chat_prompt2 = ChatPromptTemplate.from_messages(
    [system_prompt2, student_prompt_2])

main_chain = LLMChain(llm=chat_model, prompt=chat_prompt2, output_key='user_output')

# creation of the chain in charge of memory:
sys_promptSM: PromptTemplate = PromptTemplate(
    input_variables=["user_output", "user_query", "current_chat_history"],
    template="""**System Prompt** Your role is to use both the current chat history, and the latest exchange
    between the user and the model to create a summary to save the context. The summary you will create will replace
    the current chat history and will be used by another model, so it needs to only keep the important,
    non-redundant information.

    It is essential to rely strictly on both context and chat history, without drawing on any external knowledge.
    Adhere to the following guidelines when crafting your responses:

    1. **Named Entities**: If the user mentions his name/ identity, you should make sure that the summary always include
    it.

    2. **Topics**: Pay attention to the user's responses to decide what information to keep in the summary.
    If the user is dissatisfied with the model's response, then there is no need to include much of it.

    3. **Brevity and Uniqueness** Try to keep as much relevant information as possible in your summaries, however, 
    your context window is finite, so keep the summaries you do UNDER 2500 words.

    4. **Self-Assessment for Relevance**: After drafting a summary, briefly assess its content for 
    redundancy and relevance. Avoid adding repetitive information unless it is essential for clarity or continuity.

    This approach ensures you effectively improves the user experience 
    by allowing the main model to keep giving relevant answers. **End of System Prompt**
    """)
system_promptSM = SystemMessagePromptTemplate(prompt=sys_promptSM)

student_promptSM: PromptTemplate = PromptTemplate(
    input_variables=["user_output", "user_query", "current_chat_history"],
    template="""Using {current_chat_history}, {user_query} and {user_output} create a new summary.
             Follow the Guidelines!""")
student_prompt_SM = HumanMessagePromptTemplate(prompt=student_promptSM)
chat_prompt2 = ChatPromptTemplate.from_messages(
    [system_promptSM, student_prompt_SM])

memory_summary_chain = LLMChain(llm=chat_model, prompt=chat_prompt2, output_key='new_chat_history')

seq_chain2 = SequentialChain(
    chains=[main_chain, memory_summary_chain],
    input_variables=["final_output", "user_query", "current_chat_history"],
    output_variables=['user_output', 'new_chat_history'])

while True:
    query = input("Ask anything!: ")
    if query == "exit" or query == "quit" or query == "q":
        # those are the conditions to exit the loop
        print('Exiting')
        sys.exit()

    else:
        test_text = vectorstore.similarity_search_with_score(query, 8)
        result = seq_chain.invoke({'original_sentence': test_text, 'user_query': query})
        # print("Retriever's reponse: \n")
        # print(result['final_output'])

        # Using retrieval chain's results to test the main chain.
        input2 = result['final_output']

        with open(current_memo_path, 'r') as file:
            chat_history = file.read()

        chat_history = chat_history + last_elements

        tests2 = seq_chain2.invoke({
            'final_output': input2,
            'user_query': query,
            'current_chat_history': chat_history})

        print('\nAnswer: ')
        response = tests2['user_output']
        print(response)
        # print('\nNew Chat History: ')
        new_chat_history = tests2['new_chat_history']
        # print(new_chat_history)

        with open(current_memo_path, 'w') as file:
            file.write(new_chat_history)

        # This is the part that write the exchanges, but it needs to now the number of the previous exchange.
        with open(full_memo_path, 'a') as file:
            num = str(len(exchange_numbers))
            exchange = '[' + num + ']\n(user question): ' + query + '\n(model response): ' + response + '\n\n'
            file.write(exchange)
