from src.agents import app_graph

inputs = {"input_text": "What is the derivative of x^2?", "messages": []}

print("🚀 Starting Agent Workflow...")
for output in app_graph.stream(inputs):
    for key, value in output.items():
        print(f"Finished Node: {key}")

# Get final state
final_state = app_graph.invoke(inputs)
print("\n--- FINAL ANSWER ---")
print(final_state["final_answer"])
