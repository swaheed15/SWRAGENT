import os
import logging
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import PubMedAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType 
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from scholarly import scholarly
from dotenv import load_dotenv# Import ChatGroq from the appropriate library
import requests
from xml.etree import ElementTree

# Load environment variables
load_dotenv()

# Retrieve API keys from environment or secrets
pubmed_api_key = st.secrets["PUBMED_API_KEY"]

NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

def build_esearch_url(query: str):
    return f"{NCBI_BASE_URL}esearch.fcgi?db=pubmed&term={query}&api_key={pubmed_api_key}&retmode=xml"

def fetch_pubmed_results(query: str):
    esearch_url = build_esearch_url(query)
    response = requests.get(esearch_url)
    if response.status_code == 200:
        return response.text  # Return XML response from PubMed
    else:
        raise ValueError("Failed to fetch PubMed results. Please try again.")

def parse_pubmed_xml(xml_data):
    root = ElementTree.fromstring(xml_data)
    articles = []
    for docsum in root.findall(".//DocSum"):
        title = docsum.find(".//Item[@Name='Title']").text
        articles.append(title)
    return articles

def pubmed_tool_func(query: str, callback=None):
    try:
        result = fetch_pubmed_results(query)
        parsed_result = parse_pubmed_xml(result)
        if callback:
            callback(parsed_result)
        return parsed_result
    except Exception as e:
        return str(e)

pubmed_tool = Tool(
    name="PubMedQuery",
    description="Search PubMed for medical and scientific literature using NCBI E-utilities.",
    func=pubmed_tool_func,
    is_single_input=True
)

def google_scholar_query(query, num_results=10):
    search_results = scholarly.search_pubs(query)
    results = []
    try:
        for _ in range(num_results):
            result = next(search_results)
            if "bib" in result:
                results.append(result["bib"])
    except StopIteration:
        pass
    return results

google_scholar_tool = Tool(
    name="GoogleScholarQuery",
    description="Search Google Scholar for academic articles.",
    func=google_scholar_query,
    is_single_input=True
)

tools = [pubmed_tool, google_scholar_tool]

st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input("Please Enter your Groq API key:", type="password")


from langchain_groq import ChatGroq 
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="gemma2-9b-it",
    streaming=True
)

st.title("Research Agent")
st.write("This agent helps you search PubMed and Google Scholar")

top_k_results = st.slider("Top Results:", 1, 10, 5)
doc_content_chars_max = st.slider("Max Characters:", 100, 500, 250)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "Assistant", "content": "Hi, I am your research assistant. How can I help you?"}]

with st.container():
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Search: "):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    pubmed_tool.top_k_results = top_k_results
    pubmed_tool.doc_content_chars_max = doc_content_chars_max

    logging.info(f"Top K Results: {top_k_results}, Max Characters: {doc_content_chars_max}")

    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )  

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
