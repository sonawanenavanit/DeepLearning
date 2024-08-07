import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# Set ip the Streamlit app
st.set_page_config(page_title="Text To Math Problem Solver")
st.title("Text To Math Problem Solver")

groq_api_key=st.sidebar.text_input(label="Groq API Key", type="password")

if not groq_api_key:
    st.info("Please add your api key to contunue")
    st.stop()

llm=language_model=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

# Initializing tools 

wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet and solving your math problem"

)

# Initiallize the Math Tool

math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related question"
)

prompt = """
You are a agent tasked for solving users mathematical question. Logically arrive at the solution and display it point wise for the question below.
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Combine all the tools into chain

chain=LLMChain(llm=llm,prompt=prompt_template)

reasoing_tool=Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

# Initialize the agents

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoing_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_error=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm an a Math Chatbot who can answer all your math question"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Function to generate the response
def generate_response(question):
    response=assistant_agent.invoke({"input":question})
    return response

# Lets start the interaction
question=st.text_area("Enter youyr question:")
if st.button("Find my answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])

            st.session_state.messages.append({"role":"assitant","content":response})
            st.write("### Response:")
            st.success(response)

    else:
        st.warning("Please enter the Input")



