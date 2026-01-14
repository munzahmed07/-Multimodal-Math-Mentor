from src.agents import app_graph


question = "What is the derivative of x^5? Explain it simply."

print(f" TEST STARTED: '{question}'")
print("--------------------------------------------------")

# 2. Define the input state
inputs = {
    "input_text": question,
    "messages": []
}


try:
    final_state = app_graph.invoke(inputs)

    
    print("\n FINAL ANSWER FROM AGENTS:")
    print("--------------------------------------------------")
    print(final_state["final_answer"])
    print("--------------------------------------------------")

except Exception as e:
    print(f"\n ERROR: {e}")
    print("Make sure you have your .env file setup and 'src/agents.py' saved.")
