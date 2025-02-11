from langsmith import Client

# Connect LangSmith to your locally hosted Llama server
client = Client(api_url="http://129.241.113.29:8081")  # Your Llama server IP

# Test prompt refinement
response = client.run_model(
    model="Meta-Llama-3.1-8B-Instruct",
    prompt="Refine this prompt: How do I optimize my Llama 3.1 model for chat applications?",
    parameters={"temperature": 0.7}
)

print(response)
