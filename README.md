A simple Streaming API for langchain's tool calling agent built on FastAPI. It uses MongoDB to store and update chat histories against each session ID. Uses Pinecone for the vector Database.

To run it:
1) Install all requirements
2) Make sure to put all API keys in a .env file
3) run "uvicorn api:app" on the terminal



Example curl request to the api:

curl -X POST "http://localhost:8000/ask" \
-H "Content-Type: application/json" \
-d '{
  "input": "Your input text here",
  "session_id": "your_session_id_here"
}'
