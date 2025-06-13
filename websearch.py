import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults

# ----------------------------
# Gemini API Key Setup
# ----------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyDUqoyAAYv4Veg3hwrfELljrC_-MkcJMpg"  # Replace with your Gemini API key

# ----------------------------
# Model & Tool Initialization
# ----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
search_tool = DuckDuckGoSearchResults()

# ----------------------------
# Streamlit UI Setup
# ----------------------------
st.set_page_config(page_title="🔍 Web Search with Gemini", layout="centered")
st.title("🌐 Ask Anything (Web Search + Gemini)")

user_query = st.text_input("Enter your search query:")

if st.button("🔎 Search"):
    if not user_query.strip():
        st.warning("Please enter a query to search.")
    else:
        try:
            with st.spinner("Searching the web..."):

                # Step 1: Run web search
                results = search_tool.run(user_query)

                # Step 2: Safely format results
                if isinstance(results, str):
                    search_summary = results
                elif isinstance(results, list):
                    search_summary = "\n\n".join([
                        f"{res.get('title', '')}\n{res.get('snippet', '')}\n{res.get('link', '')}"
                        if isinstance(res, dict) else str(res)
                        for res in results
                    ])
                else:
                    search_summary = str(results)

                # Optional: Debug raw output
                # st.write("🔍 Raw Search Output:", results)

                # Step 3: Prompt template
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant. Use the search results below to answer the user's question."),
                    ("human", "Search Results:\n{context}\n\nQuestion:\n{query}")
                ])

                chain: Runnable = prompt | llm

                # Step 4: Get response from Gemini
                response = chain.invoke({"context": search_summary, "query": user_query})
                answer = response.content if hasattr(response, "content") else str(response)

            st.success("✅ Answer generated!")
            st.markdown(f"**Answer:**\n\n{answer}")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
