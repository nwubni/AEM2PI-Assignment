Overview and Report
This project builds on the knowledge gained from Module 1.
It uses the concept of RAG to implement a costumer inquiry mechanism that answers questions based on company's data to give accurate, informed, and verifyable responses.
It has the basic moderation and cost and latency metrics for the model in place.
In order to avoid bloating the project and focus on RAG, I have excluded the tests from the first module.
Here are some assumptions:
1. Tests for moderation and model api calls are available.
2. Rate limiting has been properly setup to prevent service abuse.

Things included
1. Moderation
2. Response validation

This RAG project is about....
It starts by building the data pipeline, taking the FAQ documents of the company and chunking them.
Next, an embedding is generated for each chunk and stored using FAISS.

Data Pipeline
Steps
1. chunking
Discuss document structure
Uses sliding window
2. embedding
Uses sentence transformer. the embedding was normalized
3. saving
saves as json to have key value pairs for easy referencing by faiss embedding id

Query Pipeline
With the data pipeline in place, the next process is to accept user queries and answer them based on the FAQ document using the embedded form of the user query to match answers semantically.
1. Reads user query from the command line
2. Moderates input
3. Generates user input embedding and normalizes the user query vector
4. Searches for vector similary using pregenerated vectores from FAQ chunks and returns the 3 most relevant matches
5. Adds chunks and sources to prompt and sends to LLM for a respoonse
6. Logs metrics and response in `outputs/sample_queries.json` composed of cost, latency, timestamp, user_question, system response, actions, relative chuncks used to generate response, and their sources. 


Key steps to keep tokens withing limit and keep API costs low:
1. Since user queries can be highly repetitive, this project would benefit from query caching, which it already implements in the `scripts` folder
2. Deduplication would help to remove redundancy in prompts and reduce model costs.


Limitations
1. No query caching