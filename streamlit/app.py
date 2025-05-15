# 1. Import Required Libraries
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from IPython.display import Image, display
from typing import List
from typing_extensions import TypedDict
import os
import json
from pylatexenc.latex2text import LatexNodes2Text
import re
import tempfile
from pathlib import Path
from datetime import datetime
import time
import random
import random

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

class GraphState(TypedDict):
    question: str
    raw_solution: str
    generation: str
    documents: List[Document]
    web_search_needed: str
    web_search_sufficient: str
    web_results: str
    human_feedback: str
    should_end: bool

def clean_latex(latex_str):
    latex_str = latex_str.replace('\\\\', '\\')
    latex_str = re.sub(r"\\\[|\\\]", "", latex_str)
    latex_str = re.sub(r"\$+", "", latex_str)
    return latex_str

def latex_to_text(latex_str):
    cleaned = clean_latex(latex_str)
    return LatexNodes2Text().latex_to_text(cleaned)

def load_math_dataset(uploaded_files):
    """Loads JSON files from uploaded files, cleans LaTeX, and returns LangChain Documents."""
    documents = []
    temp_dir = tempfile.TemporaryDirectory()
    
    for file in uploaded_files:
        if file.name.endswith(".json"):
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
                
            try:
                with open(temp_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Extract and clean problem and solution text
                problem = data.get("problem", "")
                solution = data.get("solution", "")
                level = data.get("level", "")
                qtype = data.get("type", "")
                clean_problem = latex_to_text(problem.strip())
                clean_solution = latex_to_text(solution.strip())
                content = f"Problem:\n{clean_problem}\n\nSolution:\n{clean_solution}"
                metadata = {
                    "level": level,
                    "type": qtype
                }
                documents.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
    st.write(f"Loaded {len(documents)} documents.")
    return documents

def initialize_components():
    """Initialize all components and store in session state"""
    if not st.session_state.initialized:
        st.session_state.embedding_model = HuggingFaceEmbeddings(
                                            model_name="BAAI/bge-small-en",
                                            model_kwargs={"device": "cpu", "token": "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"},
                                            encode_kwargs={"normalize_embeddings": True}
                                        )
        st.session_state.llm = ChatGoogleGenerativeAI(
                                    model="gemini-2.0-flash",
                                    temperature=0,
                                    google_api_key="AIzaSyBZMwQLB0cEHjaj7P4ADwXcFkkJRSQCFE8"
                                )
        
        st.session_state.eval_llm = ChatGoogleGenerativeAI(
                                    model="gemini-2.0-flash",
                                    temperature=0,
                                    google_api_key="AIzaSyBZMwQLB0cEHjaj7P4ADwXcFkkJRSQCFE8"
                                )

        st.session_state.tv_search = TavilySearchResults(tavily_api_key="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", max_results=3, search_depth='advanced')

        if os.path.exists("./math_db"):
            st.session_state.chroma_db = Chroma(
                embedding_function=st.session_state.embedding_model,
                persist_directory="./math_db",
                collection_name="math_knowledge"
            )
            st.write("✅ Chroma index loaded from disk.")
        else:
            # Create new ChromaDB instance with initial documents
            st.session_state.chroma_db = Chroma(
                embedding_function=st.session_state.embedding_model,
                persist_directory="./math_db",
                collection_name="math_knowledge"
            )
            # Add initial math knowledge documents
            initial_docs = [
                # Basic Arithmetic
                Document(page_content="Addition: a + b = c", metadata={"type": "arithmetic"}),
                Document(page_content="Subtraction: a - b = c", metadata={"type": "arithmetic"}),
                Document(page_content="Multiplication: a * b = c", metadata={"type": "arithmetic"}),
                Document(page_content="Division: a / b = c", metadata={"type": "arithmetic"}),
                Document(page_content="Order of Operations: PEMDAS", metadata={"type": "arithmetic"}),
                Document(page_content="Fractions: a/b + c/d = (ad + bc)/bd", metadata={"type": "arithmetic"}),
                Document(page_content="Decimals: 0.1 + 0.2 = 0.3", metadata={"type": "arithmetic"}),
                Document(page_content="Percentages: x% of y = (x/100)*y", metadata={"type": "arithmetic"}),

                # Algebra
                Document(page_content="Linear Equations: ax + b = c", metadata={"type": "algebra"}),
                Document(page_content="Quadratic Formula: x = [-b ± √(b²-4ac)]/2a", metadata={"type": "algebra"}),
                Document(page_content="Polynomial Factorization", metadata={"type": "algebra"}),
                Document(page_content="Exponent Rules: a^m * a^n = a^(m+n)", metadata={"type": "algebra"}),
                Document(page_content="Logarithms: log_b(x) = y", metadata={"type": "algebra"}),
                Document(page_content="Complex Numbers: a + bi", metadata={"type": "algebra"}),
                Document(page_content="Matrices: Matrix Operations", metadata={"type": "algebra"}),

                # Geometry
                Document(page_content="Pythagorean Theorem: a² + b² = c²", metadata={"type": "geometry"}),
                Document(page_content="Area of Circle: πr²", metadata={"type": "geometry"}),
                Document(page_content="Volume of Sphere: (4/3)πr³", metadata={"type": "geometry"}),
                Document(page_content="Trigonometry: sin²θ + cos²θ = 1", metadata={"type": "geometry"}),
                Document(page_content="Angles: Complementary and Supplementary", metadata={"type": "geometry"}),
                Document(page_content="Circles: Arc Length and Sector Area", metadata={"type": "geometry"}),
                Document(page_content="Triangles: Area = (1/2)bh", metadata={"type": "geometry"}),

                # Calculus
                Document(page_content="Derivative Rules: d/dx(x^n) = nx^(n-1)", metadata={"type": "calculus"}),
                Document(page_content="Chain Rule: d/dx[f(g(x))] = f'(g(x))g'(x)", metadata={"type": "calculus"}),
                Document(page_content="Taylor Series Expansion", metadata={"type": "calculus"}),
                Document(page_content="Fourier Transform", metadata={"type": "calculus"}),
                Document(page_content="Limits: lim x→a f(x)", metadata={"type": "calculus"}),
                Document(page_content="Continuity and Differentiability", metadata={"type": "calculus"}),
                Document(page_content="Partial Derivatives", metadata={"type": "calculus"}),

                # Advanced Mathematics
                Document(page_content="Linear Algebra: Matrix Multiplication", metadata={"type": "advanced"}),
                Document(page_content="Differential Equations", metadata={"type": "advanced"}),
                Document(page_content="Complex Analysis", metadata={"type": "advanced"}),
                Document(page_content="Number Theory: Prime Factorization", metadata={"type": "advanced"}),
                Document(page_content="Topology: Open and Closed Sets", metadata={"type": "advanced"}),
                Document(page_content="Probability: Bayes' Theorem", metadata={"type": "advanced"}),
                Document(page_content="Statistics: Central Limit Theorem", metadata={"type": "advanced"}),
                Document(page_content="Set Theory: Union and Intersection", metadata={"type": "advanced"}),
                Document(page_content="Graph Theory: Eulerian and Hamiltonian Paths", metadata={"type": "advanced"}),
                Document(page_content="Combinatorics: Permutations and Combinations", metadata={"type": "advanced"}),
                Document(page_content="Mathematical Induction: Base case and inductive step", metadata={"type": "advanced"}),
                Document(page_content="Abstract Algebra: Groups, Rings, and Fields", metadata={"type": "advanced"}),
                Document(page_content="Real Analysis: Limits, Continuity, and Differentiability", metadata={"type": "advanced"}),
                Document(page_content="Numerical Methods: Root Finding and Approximation", metadata={"type": "advanced"}),
                Document(page_content="Optimization: Linear and Nonlinear Programming", metadata={"type": "advanced"}),
                Document(page_content="Game Theory: Nash Equilibrium", metadata={"type": "advanced"}),
                Document(page_content="Mathematical Logic: Propositional and Predicate Calculus", metadata={"type": "advanced"}),
                Document(page_content="Category Theory: Functors and Natural Transformations", metadata={"type": "advanced"})
            ]
            st.session_state.chroma_db.add_documents(initial_docs)
            if not hasattr(st.session_state, 'initial_docs'):
                st.session_state.initial_docs = [Document(page_content="Mathematical knowledge base initialized")]
            
            st.session_state.chroma_db.add_documents(st.session_state.initial_docs)
            st.write("✅ Created new Chroma index with initial math knowledge.")
            
            st.session_state.initialized = True


def configure_retriever():
    """Configure the document retriever with the current Chroma DB"""
    if st.session_state.chroma_db:
        st.session_state.retriever = st.session_state.chroma_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.8 
            }
        )
        return True
    return False

def input_guardrails(state):
    with st.status("Checking if the question alligna to mathematics..."):
        question = state["question"]
        input_guard_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are Professor Mathis, the world's foremost mathematical scientist with comprehensive knowledge spanning all mathematical domains and their real world applications. Your expertise ranges from fundamental arithmetic to cutting edge mathematical theories. Engage users with both mathematical precision and approachable wisdom. For general inquiries about mathematics, provide insightful responses that demonstrate the beauty and importance of mathematics. For non mathematical questions, gracefully redirect the conversation to mathematical topics while maintaining a friendly and professional tone.\nHere are your key directives:\n1. Be the ultimate authority in mathematics, demonstrating both depth and breadth of knowledge\n2. Engage in meaningful mathematical discussions that inspire curiosity\n3. When faced with general questions, showcase the omnipresence of mathematics in our world\n4. Maintain a balance between academic rigor and approachable communication\n5. Demonstrate how mathematics shapes our understanding of the universe\nExample Responses:\n- 'Mathematics is the language of the universe, and I have ve dedicated my life to understanding its profound beauty. What aspect of this incredible field fascinates you most?'\n- 'From the symmetry of snowflakes to the equations governing black holes, mathematics is everywhere. What mathematical concept would you like to explore today?'\n- 'As a mathematician, I see patterns and structures that connect all aspects of our world. How can I help you appreciate the mathematical foundations of your question?"
            ),
                ("human", "{question}")
            ])
        
        input_guard_chain = input_guard_prompt | st.session_state.llm | StrOutputParser()
        response = input_guard_chain.invoke({"question": question})
        
        # If it's not math-related
        if "I'm sorry, I can only help with math questions." in response:
            return {**state, "generation": response, "should_end": True}
        
    # Otherwise continue with an empty generation
    return {**state, "generation": ""}

def retrieve(state):
    with st.status("Knowledge Base..."):
        question = state["question"]
        if state.get("generation") and "I'm sorry" in state["generation"]:
            return state
        
        if not configure_retriever():
            st.warning("No retriever configured.")
            return {**state, "documents": []}
            
        docs = st.session_state.retriever.get_relevant_documents(question)
        st.write(f"Found {len(docs)} potentially relevant documents")
        return {**state, "documents": docs}

def document_grader(state):
    with st.status("Grading document..."):
        if not state["documents"]:
            st.write("No documents found, will need web search")
            return {**state, "web_search_needed": "Yes"}
        
        grader_prompt = ChatPromptTemplate.from_template("""
        You are an advanced mathematics document relevance grader with expertise in all mathematical domains.

        Question:
        {question}

        Documents:
        {documents}

        Your task is to:
        1. Thoroughly analyze mathematical concepts across all domains (arithmetic, algebra, geometry, calculus, advanced)
        2. Verify completeness of solutions, including all necessary steps, proofs, and derivations
        3. Identify missing mathematical components with precision
        4. Evaluate problem difficulty and match with appropriate solution methods
        5. Provide detailed reasoning with mathematical rigor
        6. Output a JSON with:
        - "relevant_documents": [list of indices of relevant docs],
        - "is_sufficient": true/false,
        - "missing_elements": [specific missing mathematical components if insufficient],
        - "reasoning": [detailed mathematical explanation of relevance including concepts, difficulty, and type]

        Return your output as JSON only.
        """)

        parser = JsonOutputParser()

        def grade_documents(inputs):
            question = inputs["question"]
            documents = [doc.page_content for doc in inputs["documents"]]
            return {"question": question, "documents": documents}
            
        document_grader_chain = RunnableLambda(grade_documents) | grader_prompt | st.session_state.eval_llm | parser

        result = document_grader_chain.invoke({"question": state["question"], "documents": state["documents"]})
        relevant_indices = result.get("relevant_documents", [])
        is_sufficient = result.get("is_sufficient", False)
        relevant_docs = [state["documents"][i] for i in relevant_indices if i < len(state["documents"])]
        
        st.write(f"{len(relevant_docs)} relevant docs found. Sufficient: {is_sufficient}")
        
        return {
            **state,
            "documents": relevant_docs,
            "web_search_needed": "No" if is_sufficient else "Yes"
        }

def web_search(state):
    if state.get("web_search_needed") == "Yes":
        with st.status("Performing Web Search..."):
            time.sleep(1)  # Give UI time to update
            # Perform web search
            try:
                results = st.session_state.tv_search.invoke({"query": state["question"]})
                
                # Check if results are empty
                if not results:
                    st.write("No web results found")
                    return {**state, "web_results": "", "web_search_sufficient": "No"}
                
                # Combine contents of the search results
                web_context = "\n\n".join(r["content"] for r in results)
                st.write(f"Found {len(results)} web results")
                
                return {**state, "web_results": web_context, "web_search_sufficient": "Unknown"}
            except Exception as e:
                st.error(f"Web search error: {str(e)}")
                return {**state, "web_results": "", "web_search_sufficient": "No"}
    
    return {**state, "web_results": "", "web_search_sufficient": "NotNeeded"}
        
def assess_web_results(state):
    # Skip if web search wasn't needed
    if state["web_search_sufficient"] == "NotNeeded":
        return state
    
    # Skip if web results are already known to be insufficient
    if state["web_search_sufficient"] == "No":
        return {**state, "generation": "Sorry, I couldn't find any reliable information online to answer your question.", "should_end": True}
    
    with st.status("🔍 Assessing Web..."):
        # Define the assessment prompt
        assess_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a mathematics expert evaluating whether web search results contain sufficient information to solve a math problem."
             "Your task is to determine if the provided web content has relevant mathematical information to answer the question."
             "Return ONLY 'Yes' if the content is sufficient, or 'No' if it lacks necessary information."),
            ("human", 
             "Question: {question}\n\nWeb Content:\n{web_results}")
        ])
        
        assessment_chain = assess_prompt | st.session_state.llm | StrOutputParser()
        result = assessment_chain.invoke({
            "question": state["question"],
            "web_results": state["web_results"]
        })
        
        st.write(f"Web Results Assessment: {result}")
        
        if "No" in result:
            return {**state, 
                    "web_search_sufficient": "No", 
                    "generation": "Sorry, I couldn't find any reliable information online to answer your question.",
                    "should_end": True}
        
    return {**state, "web_search_sufficient": "Yes", "should_end": False}

def generate(state):
    with st.status("Generating Solution..."):
        if state.get("generation") and not state.get("human_feedback"):
            return state
        
        # Clear generation if there's new feedback
        if state.get("human_feedback"):
            state["generation"] = None

        docs = "\n\n".join(d.page_content for d in state["documents"])
        context = docs + "\n\n" + state["web_results"]
        fb = state.get("human_feedback", "")
        prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are an advanced mathematics professor with expertise across all mathematical domains.\n\n"
        "Your task is to:\n"
        "1. Provide complete, rigorous mathematical solutions with all necessary proofs and derivations\n"
        "2. Include alternative solution approaches when applicable\n"
        "3. Verify mathematical correctness through logical consistency and domain specific validation\n"
        "4. For graphical concepts, provide a clear description and analysis instead of visual representation\n"
        "5. Ensure responses are concise and avoid duplication\n"
        "6. Use natural, human-like language without technical symbols or dashes"
        ),
        ("human", "{context}\n\n{feedback}")
        ])

        chain = prompt | st.session_state.llm | StrOutputParser()
        response = chain.invoke({"context": context, "feedback": fb})
        
        # Remove duplicate responses and improve visual handling
        if "[Visual representation would be shown here]" in response:
            response = response.replace("[Visual representation would be shown here]", "")
            response = response.strip()
            if not response:
                response = "Here's the explanation of the visual concept:"
        
        return {**state, "raw_solution": response}

# Output Guardrails
def output_guardrails(state):
    with st.status("Validating Solution..."):
        raw = state.get("raw_solution", "")
        if not raw:
            return state

        check_prompt = ChatPromptTemplate.from_messages([
            ("system", 
            "You are a mathematics expert responsible for validating the accuracy and completeness of step by step solutions.\n\n"
            "Given a math question and a proposed solution:\n"
            "1. If the solution is entirely correct, includes all necessary steps, and clearly explains the reasoning, return the solution exactly as-is.\n"
            "2. If the solution is incorrect, missing key steps, or lacks clear logic, rewrite it completely with detailed, correct, step by step reasoning.\n\n"
            "Do not include any additional commentary or notes. Return only the corrected (or confirmed) solution."),
            
            ("human", 
            "Question:\n{question}\n\nProposed Solution:\n{raw_solution}")
        ])

        chain = check_prompt | st.session_state.eval_llm | StrOutputParser()
        validated = chain.invoke({
            "question": state["question"],
            "raw_solution": raw
        })

        # Place the final solution into 'generation'
        return {**state, "generation": validated}
    
def save_feedback(question, solution, feedback, rating=None):
    feedback_file = Path("./feedback_data/feedback_log.json")
    feedback_file.parent.mkdir(exist_ok=True, parents=True)

    # Use readable datetime format
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = {
        "timestamp": timestamp,
        "question": question,
        "solution": solution,
        "feedback": feedback,
        "rating": rating
    }

    # Load existing data if file exists
    if feedback_file.exists():
        with open(feedback_file, "r") as f:
            try:
                feedback_data = json.load(f)
            except json.JSONDecodeError:
                feedback_data = []
    else:
        feedback_data = []

    # Append the new entry
    feedback_data.append(entry)

    # Save back to the file
    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=2)
    
    st.success(f"Feedback saved!")

def human_in_the_loop(state):
    # Initialize session state variables if not present
    if "feedback_iterations" not in st.session_state:
        st.session_state.feedback_iterations = 0
    if "phase" not in st.session_state:
        st.session_state.phase = "rating"
    if "processing_feedback" not in st.session_state:
        st.session_state.processing_feedback = False
    
    # Get current iteration and phase
    it = st.session_state.get("feedback_iterations", 0)
    phase = st.session_state.get("phase", "rating")
    
    # Create truly unique keys for widgets
    # Generate unique widget ID based on question and iteration
    if "widget_id" not in st.session_state:
        st.session_state.widget_id = (
            f"{hash(state['question'])}_"
            f"{st.session_state.feedback_iterations}_"
            f"{time.time()}"
        )
    wid = st.session_state.widget_id
    suf = f"_iter{it}_{wid}_{time.time()}"
    
    # Display solution information
    st.subheader("🔎 Generated Solution")
    if it > 0:
        st.info(f"Iteration {it} - Solution based on your feedback")
    st.markdown(f"**Question:** {state['question']}")
    st.markdown(state["generation"])
    st.write("---")
    
    # Skip feedback process for non-math questions
    low = state["generation"].lower() if state["generation"] else ""
    if "only help with math" in low or "couldn't find any reliable information" in low:
        st.session_state.submitted = False
        return {**state, "human_feedback": ""}
    
    # Maximum iterations check
    if st.session_state.feedback_iterations >= 3:
        st.warning("Maximum feedback iterations (3) reached. Final solution saved.")
        save_feedback(state["question"], state["generation"], "Max iterations reached", rating=None)
        reset_feedback_state()
        return {**state, "human_feedback": ""}
    
    # STEP 1: Rating phase
    if phase == "rating":
        import uuid
        slider_key = f"rating_{st.session_state.feedback_iterations}"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = 1
        rating = st.slider('Rate the solution (1-5 stars)', min_value=1, max_value=5, step=1, value=st.session_state[slider_key], key=f'rating_{time.time()}_{random.randint(1, 1000000)}')
        # Remove direct assignment to st.session_state[slider_key] after widget instantiation
        # Instead, update only the current_rating key for feedback tracking
        st.session_state[f"current_rating_{st.session_state.feedback_iterations}"] = rating
        if st.button("Submit Rating", key=f"submit{suf}"):
            st.session_state[f"current_rating_{st.session_state.feedback_iterations}"] = rating
            st.session_state.phase = "approval"
            st.rerun()
        return state
    
    # STEP 2: Approval phase
    if phase == "approval":
        st.write(f"Rating: {st.session_state.get(f'current_rating_{st.session_state.feedback_iterations}', 1)} ⭐")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Accept", key=f"accept{suf}"):
                save_feedback(
                    state["question"], state["generation"], "", 
                    rating=st.session_state.get(f"current_rating_{st.session_state.feedback_iterations}", 1)
                )
                st.success("Solution accepted!")
                reset_feedback_state()
                return {
                    **state,
                    "human_feedback": "",
                    "raw_solution": None,
                    "generation": "",
                    "feedback_iterations": st.session_state.feedback_iterations + 1
                }
        save_feedback(
            state["question"], state["generation"], "", 
            rating=st.session_state.get(f"current_rating_{st.session_state.feedback_iterations}", 1)
        )
        st.success("Solution accepted!")
        reset_feedback_state()
        return {
            **state,
            "human_feedback": "",
            "raw_solution": None,
            "generation": "",
            "feedback_iterations": st.session_state.feedback_iterations + 1
        }
        with c2:
            if st.button("❌ Request Changes", key=f"request{suf}"):
                st.session_state.phase = "feedback"
                st.rerun()
        return state
    
    # STEP 3: Feedback and regeneration phase
    if phase == "feedback":
        fb = st.text_area("✏️ Enter your feedback:", key=f"fb{suf}")
        submit_fb = st.button("Submit Feedback", key=f"fb_submit{suf}")
        
        if submit_fb and fb.strip():
            st.session_state.processing_feedback = True
            st.session_state.current_feedback = fb
            st.session_state.last_question = state["question"]
            st.session_state.last_generation = state["generation"]
            save_feedback(
                state["question"], state["generation"], fb, rating=st.session_state.get(f"current_rating_{st.session_state.feedback_iterations}", 1)
            )
            return {
                **state,
                "human_feedback": fb,
                "raw_solution": None,
                "generation": "",
                "feedback_iterations": st.session_state.feedback_iterations + 1
            }
            
            # Create a new state with feedback
            new_state = GraphState(
                question=state["question"],
                human_feedback=fb,
                raw_solution=None,
                generation="",
                documents=state.get("documents", []),
                web_search_needed=state.get("web_search_needed", ""),
                web_search_sufficient=state.get("web_search_sufficient", ""),
                web_results=state.get("web_results", ""),
                should_end=False
            )
            
            # Store this temporary state to show a placeholder while processing
            st.session_state.temp_state = new_state
            
            try:
                with st.spinner("🧮 Generating new solution with your feedback..."):
                    # Resume graph from 'generate'
                    math_agent = build_math_agent()
                    new_state = math_agent.invoke(new_state)
                
                # Increment feedback iteration
                st.session_state.feedback_iterations += 1
                st.session_state.phase = "rating"
                st.session_state.widget_id = str(random.randint(10000, 99999))
                st.session_state.processing_feedback = False
                
                # Update the current state with the new solution
                st.session_state.current_state = new_state
                return new_state
            except Exception as e:
                st.error(f"Error regenerating solution: {str(e)}")
                st.session_state.processing_feedback = False
                
                # If error occurs, show the previous state
                return state
        
        # If user hasn't submitted feedback yet, show the current state
        return state

def reset_feedback_state():
    """Reset all feedback related session state variables"""
    st.session_state.phase = "rating"
    st.session_state.feedback_iterations = 0
    st.session_state.submitted = False
    st.session_state.processing_feedback = False
    st.session_state.current_state = None
    st.session_state.widget_id = str(random.randint(10000, 99999))

def stop(state):
    """End the workflow"""
    return state

def build_math_agent():
    """Build the MATH Agent workflow"""
    agent = StateGraph(GraphState)

    agent.add_node("input_guardrails", input_guardrails)
    agent.add_node("retrieve", retrieve)
    agent.add_node("grade", document_grader)
    agent.add_node("web_search", web_search)
    agent.add_node("assess_web_results", assess_web_results)
    agent.add_node("generate", generate)
    agent.add_node("output_guardrails", output_guardrails)
    agent.add_node("human_review", human_in_the_loop)
    agent.add_node("stop", stop)
    agent.add_edge("stop", END)

    agent.set_entry_point("input_guardrails")

    agent.add_conditional_edges(
        "input_guardrails",
        lambda state: "end" if state.get("should_end", False) else "continue",
        {
            "end": "stop",
            "continue": "retrieve"
        }
    )

    agent.add_edge("retrieve", "grade")
    agent.add_conditional_edges(
        "grade", 
        lambda state: "No" if state["web_search_needed"] else "Yes",
        {
            "Yes": "generate",     # Use retrieved docs
            "No": "web_search"     # Perform web search if no relevant doc
        }
    )
    agent.add_edge("web_search", "assess_web_results")
    agent.add_conditional_edges(
        "assess_web_results",
        lambda state: "end" if state.get("should_end", False) else "continue",
        {
            "end": "stop",
            "continue": "generate" 
        }
    )
    agent.add_edge("generate", "output_guardrails")
    agent.add_edge("output_guardrails", "human_review")

    agent.add_conditional_edges(
        "human_review",
        lambda state: "regenerate" if state["human_feedback"] else "complete",
        {
            "regenerate": "generate",
            "complete": END
        }
    )

    return agent.compile()

def main():
    
    st.set_page_config(page_title="Dr. Mathos", page_icon="🔮", layout="wide")
    st.title("🔮 Dr. Mathos: The Quantum Mind of Mathematics")
    st.markdown(""" your geeky math professor from another dimension, here to decode the universe one problem at a time.""")
    
    initialize_components()
    retriever_configured = configure_retriever()
    st.session_state.web_search_tool = st.session_state.tv_search
    
    tab1, tab2 = st.tabs(["Ask Questions", "View Feedback"])
    
    with tab1:
        st.header("Ask Your Math Question")
        
        if retriever_configured:
            st.success("✅ Knowledge base is ready for queries.")
        else:
            st.warning("⚠️ No knowledge base available. The agent will rely on web search.")
        
        # Initialize state variables if needed
        if "current_state" not in st.session_state:
            st.session_state.current_state = None
        if "is_generating" not in st.session_state:
            st.session_state.is_generating = False
        if "widget_id" not in st.session_state:
            st.session_state.widget_id = str(random.randint(10000, 99999))
        
        # Question submission form
        with st.form(key="query_form"):
            user_query = st.text_input("Ask Your Math Question", key="query_input")
            submit_button = st.form_submit_button("Solve", disabled=st.session_state.get("is_generating", False))
        
        # Process new question submission
        if submit_button and user_query.strip():
            # Reset feedback process when a new question is submitted
            reset_feedback_state()
            
            st.session_state.is_generating = True
            with st.spinner("Solving Question..."):
                math_agent = build_math_agent()
                initial_state = GraphState(
                    question=user_query,
                    raw_solution="",
                    generation="",
                    documents=[],
                    web_search_needed="",
                    web_search_sufficient="",
                    web_results="",
                    human_feedback="",
                    should_end=False
                )
                result = math_agent.invoke(initial_state)
            
            st.session_state.current_state = result
            st.session_state.is_generating = False
        
        # Process and display current solution with feedback loop
        if st.session_state.current_state:
            # Show processing indicator if feedback is being processed
            if st.session_state.get("processing_feedback", False):
                st.info("Processing feedback")
                if st.session_state.get("temp_state"):
                    # Show a placeholder with the previous solution and the feedback
                    temp_state = st.session_state.temp_state
                    st.subheader("Previous Solution")
                    st.markdown(f"**Question:** {temp_state['question']}")
                    st.markdown(st.session_state.get("last_generation", ""))
                    st.write("---")
                    st.subheader("Your Feedback")
                    st.write(st.session_state.get("current_feedback", ""))
            else:
                # Normal flow - show the solution with feedback options
                final_state = human_in_the_loop(st.session_state.current_state)
                st.session_state.current_state = final_state
    
    # Feedback records tab
    with tab2:
        st.header("Feedback Records")
        feedback_file = Path("./feedback_data/feedback_log.json")
        if feedback_file.exists():
            try:
                with open(feedback_file, "r") as f:
                    feedback_data = json.load(f)
                if feedback_data:
                    st.write(f"Found {len(feedback_data)} feedback records.")
                    json_str = json.dumps(feedback_data, indent=2)
                    st.download_button("Download Feedback Data", json_str, "math_agent_feedback.json", "application/json")
                    for i, entry in enumerate(reversed(feedback_data)):
                        with st.expander(f"Entry {len(feedback_data) - i}: {entry['timestamp']}"):
                            st.write(f"**Question:** {entry['question']}")
                            st.write(f"**Solution:** {entry['solution']}")
                            if entry.get("rating"):
                                st.write(f"**Rating:** {'⭐' * entry['rating']}")
                            else:
                                st.write("**Rating:** Not provided")
                            st.write(f"**Feedback:** {entry.get('feedback', 'None (Accepted)')}")
                else:
                    st.info("No feedback records found yet.")
            except Exception as e:
                st.error(f"Error loading feedback: {str(e)}")
        else:
            st.info("No feedback records found yet.")

if __name__ == "__main__":
    main()
