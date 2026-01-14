from src.agents import app_graph

# 1. Define the test question
# We ask a calculus question to test if it retrieves the "Power Rule"
question = "What is the derivative of x^5? Explain it simply."

print(f"🚀 TEST STARTED: '{question}'")
print("--------------------------------------------------")

# 2. Define the input state
inputs = {
    "input_text": question,
    "messages": []
}

# 3. Run the Agents
# This triggers: Parser -> Solver -> Verifier -> Explainer
try:
    final_state = app_graph.invoke(inputs)

    # 4. Print the result
    print("\n✅ FINAL ANSWER FROM AGENTS:")
    print("--------------------------------------------------")
    print(final_state["final_answer"])
    print("--------------------------------------------------")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("Make sure you have your .env file setup and 'src/agents.py' saved.")
