import operator
from typing import Annotated, List, TypedDict, Union
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from src.rag import get_retriever




class AgentState(TypedDict):
    input_text: str
    parsed_data: dict
    retrieved_context: str
    solution_plan: str
    final_answer: str
    verification_status: str
    critique: str
    messages: Annotated[List[str], operator.add]



llm = ChatOpenAI(model="gpt-4o", temperature=0)




def parser_node(state: AgentState):
    """Agent 1: Parser"""
    print("--- 1. PARSER AGENT ---")
    
    parsed_data = {"problem": state["input_text"],
                   "topic": "Math", "needs_clarification": False}
    return {"parsed_data": parsed_data, "messages": ["Parser: Processed input."]}


def solver_node(state: AgentState):
    """Agent 3: Solver"""
    print("--- 3. SOLVER AGENT ---")

   
    retriever = get_retriever()

   
    docs = retriever.invoke(state["parsed_data"]["problem"])
    context = "\n".join([d.page_content for d in docs])

    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a JEE Math Tutor. Solve the problem using the provided context. Show steps."),
        ("user", "Problem: {problem}\nContext: {context}")
    ])
    response = llm.invoke(prompt.format(
        problem=state["parsed_data"]["problem"], context=context))

    return {"solution_plan": response.content, "retrieved_context": context}


def verifier_node(state: AgentState):
    """Agent 4: Verifier"""
    print("--- 4. VERIFIER AGENT ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Check the math solution for errors. Return 'APPROVED' if correct, 'REJECTED' if wrong."),
        ("user", "Problem: {problem}\nSolution: {solution}")
    ])
    response = llm.invoke(prompt.format(
        problem=state["parsed_data"]["problem"], solution=state["solution_plan"]))
    status = "rejected" if "REJECTED" in response.content else "approved"
    return {"verification_status": status, "critique": response.content}


def explainer_node(state: AgentState):
    """Agent 5: Explainer"""
    print("--- 5. EXPLAINER AGENT ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Format the solution into a clear, friendly explanation for a student."),
        ("user", "Solution: {solution}")
    ])
    response = llm.invoke(prompt.format(solution=state["solution_plan"]))
    return {"final_answer": response.content}



workflow = StateGraph(AgentState)
workflow.add_node("parser", parser_node)
workflow.add_node("solver", solver_node)
workflow.add_node("verifier", verifier_node)
workflow.add_node("explainer", explainer_node)

workflow.set_entry_point("parser")
workflow.add_edge("parser", "solver")
workflow.add_edge("solver", "verifier")
workflow.add_edge("verifier", "explainer")
workflow.add_edge("explainer", END)

app_graph = workflow.compile()
