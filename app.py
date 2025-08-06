from typing import Dict, TypedDict, List, Optional
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from datetime import datetime
from IPython.display import display , Image , Markdown
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


# DuckDuckGo search tool

ddg_search = DuckDuckGoSearchResults(
    backend="api",
    max_results=3,
    time_limit="Y",
)

# In-memory storage for conversations
conversation_memory = {}


def show_md_file(content):
    """Saves content to a temp file and displays it"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(content)
        f.flush()
        display(Markdown(content))
        return f.name
    
    
    
class State(TypedDict):
    query: str
    category: str
    response: str

class LearningResourceAgent:
    def __init__(self, prompt):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]

    def TutorialAgent(self, user_input):
        agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False)
        response = agent_executor.invoke({"input": user_input})
        return str(response.get('output')).replace("```markdown", "").strip()

    def QueryBot(self, user_input):
        session_id = "default_session"
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []
        
        # Add user message to memory
        conversation_memory[session_id].append(HumanMessage(content=user_input))
        
        # Create full prompt with memory
        full_prompt = self.prompt + conversation_memory[session_id][-10:]  # Last 10 messages
        
        response = self.model.invoke(full_prompt)
        
        # Add AI response to memory
        conversation_memory[session_id].append(AIMessage(content=response.content))
        
        return response.content

class InterviewAgent:
    def __init__(self, prompt):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]

    def Interview_questions(self, user_input):
        agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False, handle_parsing_errors=True)
        response = agent_executor.invoke({"input": user_input, "chat_history": []})
        return str(response.get('output')).replace("```markdown", "").strip()

    def Mock_Interview(self):
        session_id = "mock_interview_session"
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []
        
        initial_message = 'I am ready for the interview.'
        conversation_memory[session_id].append(HumanMessage(content=initial_message))
        
        full_prompt = self.prompt + conversation_memory[session_id][-10:]
        response = self.model.invoke(full_prompt)
        
        conversation_memory[session_id].append(AIMessage(content=response.content))
        return response.content

class ResumeMaker:
    def __init__(self, prompt):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = prompt
        self.tools = [DuckDuckGoSearchResults()]

    def Create_Resume(self, user_input):
        agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False, handle_parsing_errors=True)
        response = agent_executor.invoke({"input": user_input, "chat_history": []})
        return str(response.get('output')).replace("```markdown", "").strip()

class JobSearch:
    def __init__(self, prompt):
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)
        self.prompt = prompt
        self.tools = DuckDuckGoSearchResults()

    def find_jobs(self, user_input):
        results = self.tools.invoke(user_input)
        chain = self.prompt | self.model  
        jobs = chain.invoke({"result": results}).content
        return str(jobs).replace("```markdown", "").strip()

def categorize(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories:\n"
        "1: Learn Generative AI Technology\n"
        "2: Resume Making\n"
        "3: Interview Preparation\n"
        "4: Job Search\n"
        "Give the number only as an output.\n\n"
        "Query: {query}"
    )
    
    chain = prompt | llm
    category = chain.invoke({"query": state["query"]}).content
    return {"category": category}

def handle_learning_resource(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following user query:\n"
        "- Tutorial: For creating tutorials, blogs, or documentation\n"
        "- Question: For general questions about generative AI\n"
        "Give 'Tutorial' or 'Question' only.\n\n"
        "Query: {query}"
    )
    
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"category": response}

def handle_interview_preparation(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following user query:\n"
        "- Mock: For mock interviews\n"
        "- Question: For interview questions or preparation\n"
        "Give 'Mock' or 'Question' only.\n\n"
        "Query: {query}"
    )
    
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content
    return {"category": response}



def job_search(state: State) -> State:
    """Provide a job search response based on user query requirements."""
    query = state["query"].strip()
    
    # More flexible validation that handles different phrasings
    if not any(phrase in query.lower() for phrase in ["looking for", "find", "search for"]) or " in " not in query.lower():
        return {
            "response": "🔍 Please format your job search request clearly:\n\n"
                       "**Looking for [Job Role] in [Location]**\n\n"
                       "Examples:\n"
                       "- \"Looking for Data Scientist in London\"\n"
                       "- \"Find Machine Learning Engineer positions in Bangalore\"\n"
                       "- \"Search for AI Researcher roles in New York\""
        }
    
    try:
        # Improved query parsing
        if "looking for" in query.lower():
            role_part = query.split("looking for")[1].split(" in ")[0].strip()
        else:
            role_part = query.split(" in ")[0].replace("find", "").replace("search for", "").strip()
        
        role = role_part.replace("positions", "").replace("roles", "").strip()
        location = query.split(" in ")[1].split(" for ")[0].strip()
        
        # Get results with better error handling
        results = get_job_results(role, location)
        
        if results and "no results" not in results.lower():
            return {
                "response": f"✅ Here are current opportunities for {role} in {location}:\n\n{results}\n\n"
                          f"💡 Tip: For more options, try [LinkedIn Jobs](https://www.linkedin.com/jobs) or [Indeed](https://www.indeed.com)"
            }
        else:
            return {
                "response": f"⚠️ Couldn't find current openings for {role} in {location}.\n\n"
                          f"Suggestions:\n"
                          f"1. Try broader search terms (e.g., 'AI Engineer' instead of 'PyTorch Specialist')\n"
                          f"2. Check these direct links:\n"
                          f"   - [LinkedIn {role} jobs in {location}](https://www.linkedin.com/jobs/search/?keywords={role.replace(' ', '%20')}&location={location.replace(' ', '%20')})\n"
                          f"   - [Indeed {role} jobs in {location}](https://www.indeed.com/jobs?q={role.replace(' ', '%20')}&l={location.replace(' ', '%20')})\n"
                          f"3. Try again in 30 minutes"
            }
            
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower() or "202" in error_msg:
            error_display = "Our job search service is currently busy."
        else:
            error_display = "There was an unexpected error."
            
        return {
            "response": f"⚠️ {error_display}\n\n"
                      f"Please try these alternatives:\n"
                      f"1. [LinkedIn Jobs](https://www.linkedin.com/jobs)\n"
                      f"2. [Indeed](https://www.indeed.com)\n"
                      f"3. [Glassdoor](https://www.glassdoor.com/Jobs/index.htm)\n\n"
                      f"Or try your search again in 15 minutes."
        }





def get_job_results(role: str, location: str, retries: int = 2) -> str:
    """Helper function to handle job searches with retries"""
    import time
    import random
    
    for attempt in range(retries + 1):
        try:
            prompt = ChatPromptTemplate.from_template('''
            Find current job openings for {role} in {location}.
            Return 3-5 results in this format:
            
            ### {role} Opportunities in {location}
            
            [Company Name] - [Job Title]
            - Location: [Specific Address/Area]
            - Skills: [Key skills required]
            - Apply: [Link or "Company website"]
            
            If no results found, say "No recent openings found."
            ''')
            
            jobSearch = JobSearch(prompt)
            
            # Random delay to avoid rate limiting (1-3 seconds)
            time.sleep(1 + random.random() * 2)
            
            results = jobSearch.find_jobs(f"{role} in {location}")
            
            if "no results" in results.lower() or "no openings" in results.lower():
                return ""
                
            return results
            
        except Exception as e:
            if attempt == retries:
                raise e
            time.sleep(3)  # Wait longer between retries
    
    return ""




def handle_resume_making(state: State) -> State:
    """Generate a customized resume based on user details for a tech role in AI and Generative AI."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are a skilled resume expert with extensive experience in crafting resumes tailored for tech roles, especially in AI and Generative AI. 
        Your task is to create a resume template for an AI Engineer specializing in Generative AI, incorporating trending keywords and technologies in the current job market. 
        Feel free to ask users for any necessary details such as skills, experience, or projects to complete the resume. 
        Try to ask details step by step and try to ask all details within 4 to 5 steps.
        Ensure the final resume is in .md format.'''),
       MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    resumeMaker = ResumeMaker(prompt)
    path = resumeMaker.Create_Resume(state["query"])
    show_md_file(path)
    return {"response": path}

def ask_query_bot(state: State) -> State:
    """Provide detailed answers to user queries related to Generative AI."""
    system_message = '''You are an expert Generative AI Engineer with extensive experience in training and guiding others in AI engineering. 
    You have a strong track record of solving complex problems and addressing various challenges in AI. 
    Your role is to assist users by providing insightful solutions and expert advice on their queries.
    Engage in a back-and-forth chat session to address user queries.'''
    prompt = [SystemMessage(content=system_message)]

    learning_agent = LearningResourceAgent(prompt)

    path = learning_agent.QueryBot(state["query"])
    show_md_file(path)
    return {"response": path}


def tutorial_agent(state: State) -> State:
    """Generate a tutorial blog for Generative AI based on user requirements."""
    system_message = '''You are a knowledgeable assistant specializing as a Senior Generative AI Developer with extensive experience in both development and tutoring. 
         Additionally, you are an experienced blogger who creates tutorials focused on Generative AI.
         Your task is to develop high-quality tutorials blogs in .md file with Coding example based on the user's requirements. 
         Ensure tutorial includes clear explanations, well-structured python code, comments, and fully functional code examples.
         Provide resource reference links at the end of each tutorial for further learning.'''
    prompt = ChatPromptTemplate.from_messages([("system", system_message),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),])
    #agent_scratchpad is a function that formats the intermediate steps of the agent's actions and observations into a string. 
    #This function is used to keep track of the agent's thoughts or actions during the execution of the program. But its not necessary, we can do without this so we will not include it only define it.
    learning_agent = LearningResourceAgent(prompt)
    path = learning_agent.TutorialAgent(state["query"])
    show_md_file(path)
    return {"response": path}

def interview_topics_questions(state: State) -> State:
    """Provide a curated list of interview questions related to Generative AI based on user input."""
    system_message = '''You are a good researcher in finding interview questions for Generative AI topics and jobs.
                     Your task is to provide a list of interview questions for Generative AI topics and job based on user requirements.
                     Provide top questions with references and links if possible. You may ask for clarification if needed.
                     Generate a .md document containing the questions.'''
    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_message),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                        ("placeholder", "{agent_scratchpad}"),])
    interview_agent = InterviewAgent(prompt)
    path = interview_agent.Interview_questions(state["query"])
    show_md_file(path)
    return {"response": path}

def mock_interview(state: State) -> State:
    """Conduct a mock interview for a Generative AI position, including evaluation at the end."""
    system_message = '''You are a Generative AI Interviewer. You have conducted numerous interviews for Generative AI roles.
         Your task is to conduct a mock interview for a Generative AI position, engaging in a back-and-forth interview session.
         The conversation should not exceed more than 15 to 20 minutes.
         At the end of the interview, provide an evaluation for the candidate.'''
    prompt = [SystemMessage(content=system_message)]
    interview_agent = InterviewAgent(prompt)
    path = interview_agent.Mock_Interview()
    show_md_file(path)
    return {"response": path}


# Routing functions
def route_query(state: State):
    if '1' in state["category"]:
        return "handle_learning_resource"
    elif '2' in state["category"]:
        return "handle_resume_making"
    elif '3' in state["category"]:
        return "handle_interview_preparation"
    elif '4' in state["category"]:
        return "job_search"
    else:
        return "handle_learning_resource"  # Default

def route_interview(state: State) -> str:
    if 'Question'.lower() in state["category"].lower():
        return "interview_topics_questions"
    elif 'Mock'.lower() in state["category"].lower():
        return "mock_interview"
    else:
        return "mock_interview"

def route_learning(state: State):
    if 'Question'.lower() in state["category"].lower():
        return "ask_query_bot"
    elif 'Tutorial'.lower() in state["category"].lower():
        return "tutorial_agent"
    else:
        return "ask_query_bot"

# Create workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("categorize", categorize)
workflow.add_node("handle_learning_resource", handle_learning_resource)
workflow.add_node("handle_resume_making", handle_resume_making)
workflow.add_node("handle_interview_preparation", handle_interview_preparation)
workflow.add_node("job_search", job_search)
workflow.add_node("mock_interview", mock_interview)
workflow.add_node("interview_topics_questions", interview_topics_questions)
workflow.add_node("tutorial_agent", tutorial_agent)
workflow.add_node("ask_query_bot", ask_query_bot)

# Add edges
workflow.add_edge(START, "categorize")

workflow.add_conditional_edges(
    "categorize",
    route_query,
    {
        "handle_learning_resource": "handle_learning_resource",
        "handle_resume_making": "handle_resume_making",
        "handle_interview_preparation": "handle_interview_preparation",
        "job_search": "job_search"
    }
)

workflow.add_conditional_edges(
    "handle_interview_preparation",
    route_interview,
    {
        "mock_interview": "mock_interview",
        "interview_topics_questions": "interview_topics_questions",
    }
)

workflow.add_conditional_edges(
    "handle_learning_resource",
    route_learning,
    {
        "tutorial_agent": "tutorial_agent",
        "ask_query_bot": "ask_query_bot",
    }
)

# End edges
workflow.add_edge("handle_resume_making", END)
workflow.add_edge("job_search", END)
workflow.add_edge("interview_topics_questions", END)
workflow.add_edge("mock_interview", END)
workflow.add_edge("ask_query_bot", END)
workflow.add_edge("tutorial_agent", END)

# Compile workflow
graph_app = workflow.compile()

# FastAPI setup
app = FastAPI(title="GenAI Career Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    category: str
    response: str

@app.post("/query", response_model=QueryResponse)
async def process_query(data: QueryRequest):
    """Single API endpoint for all GenAI career assistance queries with memory and UI output"""
    try:
        # Initialize session memory if not exists
        session_id = "default_session"
        if session_id not in conversation_memory:
            conversation_memory[session_id] = []
        
        # Process query through workflow
        results = graph_app.invoke({"query": data.query})
        
        # Add to memory
        user_msg = HumanMessage(content=data.query)
        ai_msg = AIMessage(content=results.get("response", ""))
        
        conversation_memory[session_id].extend([user_msg, ai_msg])
        
        # Keep only last 10 messages
        if len(conversation_memory[session_id]) > 10:
            conversation_memory[session_id] = conversation_memory[session_id][-10:]
        
        return QueryResponse(
            category=results.get("category", "general"),
            response=results.get("response", "Sorry, I couldn't process your request.")
        )
    except Exception as e:
        return QueryResponse(
            category="error",
            response=f"An error occurred: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)