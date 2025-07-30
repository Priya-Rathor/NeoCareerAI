from typing import Dict, TypedDict, Optional, List
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Configure logging to reduce terminal output
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

class State(TypedDict):
    query: str
    category: str
    response: str
    conversation_history: Optional[List[Dict[str, str]]]

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    category: str
    response: str
    session_id: Optional[str] = None

class ConversationRequest(BaseModel):
    query: str
    session_id: str
    continue_conversation: bool = True

class ConversationResponse(BaseModel):
    response: str
    session_id: str
    conversation_ended: bool = False

# Memory management for sessions
class SessionManager:
    def __init__(self, max_sessions: int = 100, max_messages_per_session: int = 20):
        self.sessions = {}
        self.max_sessions = max_sessions
        self.max_messages_per_session = max_messages_per_session
    
    def get_session(self, session_id: str) -> List:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]
    
    def add_message(self, session_id: str, role: str, content: str):
        session = self.get_session(session_id)
        session.append({"role": role, "content": content})
        
        # Trim messages if exceeding limit
        if len(session) > self.max_messages_per_session:
            session[:] = session[-self.max_messages_per_session:]
        
        # Clean up old sessions if exceeding max sessions
        if len(self.sessions) > self.max_sessions:
            oldest_session = min(self.sessions.keys())
            del self.sessions[oldest_session]
    
    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

# Global session manager
session_manager = SessionManager()

class BaseAgent:
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o", 
            temperature=0, 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.tools = [DuckDuckGoSearchResults()]

class LearningResourceAgent(BaseAgent):
    def create_tutorial(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Senior Generative AI Developer and experienced blogger. 
            Create high-quality tutorial content in markdown format with:
            - Clear explanations
            - Well-structured Python code with comments
            - Functional code examples
            - Resource reference links at the end
            Return only the markdown content without any wrapper text."""),
            ("human", "{input}")
        ])
        
        agent = create_tool_calling_agent(self.model, self.tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=False,
            handle_parsing_errors=True
        )
        
        response = agent_executor.invoke({"input": query})
        return response.get('output', '').replace("```markdown", "").strip()

    def answer_query(self, query: str, session_id: str) -> str:
        system_message = """You are an expert Generative AI Engineer with extensive experience. 
        Provide insightful solutions and expert advice. Keep responses concise and helpful."""
        
        # Get conversation history
        history = session_manager.get_session(session_id)
        
        # Build prompt with history
        messages = [SystemMessage(content=system_message)]
        for msg in history[-10:]:  # Keep last 10 messages
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=query))
        
        # Get response
        response = self.model.invoke(messages)
        
        # Update session
        session_manager.add_message(session_id, "user", query)
        session_manager.add_message(session_id, "assistant", response.content)
        
        return response.content

class InterviewAgent(BaseAgent):
    def generate_questions(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a researcher specializing in Generative AI interview questions.
            Provide a curated list of interview questions based on user requirements.
            Format as markdown with clear sections and include references where possible.
            Return only the markdown content."""),
            ("human", "{input}")
        ])
        
        agent = create_tool_calling_agent(self.model, self.tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=False,
            handle_parsing_errors=True
        )
        
        response = agent_executor.invoke({"input": query})
        return response.get('output', '').replace("```markdown", "").strip()

    def conduct_mock_interview(self, query: str, session_id: str) -> str:
        system_message = """You are a Generative AI Interviewer conducting a professional mock interview.
        Ask relevant questions, provide feedback, and maintain a professional tone.
        Keep the interview focused and provide constructive feedback."""
        
        # Get conversation history
        history = session_manager.get_session(session_id)
        
        # Build prompt with history
        messages = [SystemMessage(content=system_message)]
        for msg in history[-15:]:  # Keep last 15 messages for interview context
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=query))
        
        # Get response
        response = self.model.invoke(messages)
        
        # Update session
        session_manager.add_message(session_id, "user", query)
        session_manager.add_message(session_id, "assistant", response.content)
        
        return response.content

class ResumeAgent(BaseAgent):
    def create_resume(self, query: str, session_id: str) -> str:
        system_message = """You are a skilled resume expert specializing in AI and Generative AI roles.
        Create professional resumes with trending keywords and technologies.
        Ask for necessary details step by step and format the final resume in markdown."""
        
        # Get conversation history
        history = session_manager.get_session(session_id)
        
        # Build prompt with history
        messages = [SystemMessage(content=system_message)]
        for msg in history[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=query))
        
        # Get response
        response = self.model.invoke(messages)
        
        # Update session
        session_manager.add_message(session_id, "user", query)
        session_manager.add_message(session_id, "assistant", response.content)
        
        return response.content

class JobSearchAgent(BaseAgent):
    def search_jobs(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            """Refactor the following job search results into a well-structured markdown format
            that users can easily reference. Include job titles, companies, locations, and key requirements.
            
            Content: {result}"""
        )
        
        # Search for jobs
        search_results = self.tools[0].invoke(query)
        
        # Format results
        chain = prompt | self.model
        response = chain.invoke({"result": search_results})
        
        return response.content.replace("```markdown", "").strip()

# Initialize agents
learning_agent = LearningResourceAgent()
interview_agent = InterviewAgent()
resume_agent = ResumeAgent()
job_search_agent = JobSearchAgent()

# LLM for categorization
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

def categorize(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following query into one of these categories (return only the number):\n"
        "1: Learn Generative AI Technology\n"
        "2: Resume Making\n"
        "3: Interview Preparation\n"
        "4: Job Search\n\n"
        "Query: {query}"
    )
    
    chain = prompt | llm
    category = chain.invoke({"query": state["query"]}).content.strip()
    return {"category": category}

def handle_learning_resource(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize this query (return only the category name):\n"
        "- Tutorial: For creating tutorials, documentation, or guides\n"
        "- Question: For general questions about generative AI\n\n"
        "Query: {query}"
    )
    
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content.strip()
    return {"category": response}

def handle_interview_preparation(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Categorize this query (return only the category name):\n"
        "- Mock: For mock interview requests\n"
        "- Question: For interview preparation questions\n\n"
        "Query: {query}"
    )
    
    chain = prompt | llm
    response = chain.invoke({"query": state["query"]}).content.strip()
    return {"category": response}

def tutorial_agent(state: State) -> State:
    response = learning_agent.create_tutorial(state["query"])
    return {"response": response}

def ask_query_bot(state: State) -> State:
    session_id = state.get("session_id", "default")
    response = learning_agent.answer_query(state["query"], session_id)
    return {"response": response}

def interview_topics_questions(state: State) -> State:
    response = interview_agent.generate_questions(state["query"])
    return {"response": response}

def mock_interview(state: State) -> State:
    session_id = state.get("session_id", "default")
    response = interview_agent.conduct_mock_interview(state["query"], session_id)
    return {"response": response}

def handle_resume_making(state: State) -> State:
    session_id = state.get("session_id", "default")
    response = resume_agent.create_resume(state["query"], session_id)
    return {"response": response}

def job_search(state: State) -> State:
    response = job_search_agent.search_jobs(state["query"])
    return {"response": response}

# Routing functions
def route_query(state: State):
    category = state["category"].strip()
    if '1' in category:
        return "handle_learning_resource"
    elif '2' in category:
        return "handle_resume_making"
    elif '3' in category:
        return "handle_interview_preparation"
    elif '4' in category:
        return "job_search"
    else:
        return "handle_learning_resource"  # Default fallback

def route_interview(state: State) -> str:
    if 'question' in state["category"].lower():
        return "interview_topics_questions"
    else:
        return "mock_interview"

def route_learning(state: State):
    if 'question' in state["category"].lower():
        return "ask_query_bot"
    else:
        return "tutorial_agent"

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
app_workflow = workflow.compile()

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown - cleanup sessions
    session_manager.sessions.clear()

api = FastAPI(
    title="GenAI Career Assistant API",
    description="AI-powered career assistance for Generative AI professionals",
    version="1.0.0",
    lifespan=lifespan
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/")
async def root():
    return {"message": "GenAI Career Assistant API is running!"}

@api.get("/health")
async def health_check():
    return {"status": "healthy", "active_sessions": len(session_manager.sessions)}

@api.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Main endpoint to process any career-related query"""
    try:
        # Prepare state
        state = {
            "query": request.query,
            "session_id": request.session_id or "default"
        }
        
        # Process through workflow
        results = app_workflow.invoke(state)
        
        return QueryResponse(
            category=results.get("category", "unknown"),
            response=results.get("response", "No response generated"),
            session_id=request.session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@api.post("/conversation", response_model=ConversationResponse)
async def continue_conversation(request: ConversationRequest):
    """Endpoint for ongoing conversations (interviews, resume building, Q&A)"""
    try:
        # Check if user wants to end conversation
        if request.query.lower() in ['exit', 'quit', 'end', 'stop']:
            session_manager.clear_session(request.session_id)
            return ConversationResponse(
                response="Conversation ended. Thank you!",
                session_id=request.session_id,
                conversation_ended=True
            )
        
        # Process query through workflow
        state = {
            "query": request.query,
            "session_id": request.session_id
        }
        
        results = app_workflow.invoke(state)
        
        return ConversationResponse(
            response=results.get("response", "No response generated"),
            session_id=request.session_id,
            conversation_ended=False
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversation error: {str(e)}")

@api.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session"""
    session_manager.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}

@api.get("/sessions")
async def get_active_sessions():
    """Get count of active sessions"""
    return {"active_sessions": len(session_manager.sessions)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)