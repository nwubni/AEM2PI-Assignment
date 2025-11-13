## Project Overview and Report
This project builds on the knowledge gained from Module 1.

However, this time, it uses Retrieval-Augmented Generation (RAG) to implement an intelligent FAQ Support Chatbot for an HR SaaS company. Customer support receives repetitive questions daily about policies, features, and procedures already documented in internal FAQs and guides. This system answers employee questions by retrieving relevant information from the company's documentation to provide context and generate referenceable responses.
Rather than implementing the entire pipeline from scratch, this project leverages the LangChain library to implement the RAG pipeline. This standardizes the codebase to make the artifact production-ready and easier to scale and maintain with multiple contributors. Additionally, I used open-source components such as `langchain-text-splitters` to chunk documents, `langchain-huggingface` to embed, and `faiss` to store them locally for free. Only LLM API calls may incur charges.

In order to avoid bloating the project and focus on the core concept of RAG, the following assumptions apply:
1. Moderation has been fully implemented and tests for it are available.
2. Rate limiting has been properly set up to prevent service abuse.

This RAG project builds an intelligent FAQ Support Chatbot for an HR SaaS company that answers employee questions about policies, features, and procedures using the company's internal documentation to provide accurate, informed, and verifiable responses.
The system starts by building the data pipeline, which takes the company document(s) such as FAQs and Policies and splits them into chunks.
Next, an embedding is generated for each chunk and stored in FAISS, a vector database.

Features
- LangChain-based RAG pipeline (FAISS + HuggingFace embeddings + OpenAI LLM)
- Context truncation preserving complete Q&A pairs to control tokens and cost
- Response caching with query normalization for faster, cheaper repeated queries
- Automated evaluator that scores answers on relevance, accuracy, completeness, and clarity
- Cost and latency tracking for each query, logged to `outputs/sample_queries.json`
- Comprehensive pytest suite (retrieval, caching, truncation, evaluator, edge cases)

Architecture
- `data/faq_document.txt`: The company's FAQ document.
- `src/build_index.py`: Builds the FAISS index from FAQ documents, adds rich metadata per chunk.
- `src/query.py`: End-to-end query pipeline (retrieval, truncation, LLM, evaluator, metrics logging).
- `scripts/cache.py`: Persistent JSON cache with query normalization and context hashing.
- `scripts/evaluator.py`: Lightweight LLM-based evaluator returning structured JSON scores on a 0-10 scale.
- `prompts/main_prompt.txt`: The underlying system prompt for constrained, source-grounded answers. The prompt uses few-shot examples to guide the LLM to provide answers based on the 3 top relevant FAQ chunks for the user's query.
- `storage/faiss_index`: Directory for storing the FAISS index.
- `storage/response_cache.json`: File for storing cached responses.
- `outputs`: Directory for storing the sample queries and evaluation results.

### Data Pipeline
Steps
1. Chunking
   - The FAQ document is split using a sliding window (`chunk_size=1000`, `chunk_overlap=200`) through `RecursiveCharacterTextSplitter` to maintain chunk context between Q&A pairs.
   - Each chunk carries metadata: `id`, `document_id`, `chunk_index`, `token_count`, `char_count`, and `created_at`. These additional data can be useful for extending functionalities to include keyword searches or retrieval filtering.
2. Embedding
   - Text is converted to vectors using `all-MiniLM-L6-v2` (which has a 384-dimensional vector space) via `langchain-huggingface`.
3. Indexing and Storage
   - Chunks and embeddings are stored in a FAISS index (using cosine similarity via inner product) for high-recall semantic search.

Data Pipeline Build Command:
```bash
python -m src.build_index data/faq_document.txt
```



### Query Pipeline
With the data pipeline in place, the query process answers user questions using semantic retrieval over the indexed FAQ chunks.

Steps:
1. Read and embed the user query via the CLI argument.
2. Retrieve the top-k (k=3) relevant chunks using the embedded user query via FAISS cosine similarity search.
3. Apply truncation to preserve complete Q&A pairs, remove redundancy, and keep prompts within token budgets.
4. Invoke the LLM (`gpt-4o-mini`) with a structured system prompt and the truncated context.
5. Get and log metrics and responses to `outputs/sample_queries.json` including `cost`, `latency`, `timestamp`, `user_question`, `system_answer`, `actions`, `chunks_related`, `source`, `confidence`, and `evaluation scores`.

Query Pipeline Command:
```bash
python -m src.query "User query"
```

**Model choice:** `gpt-4o-mini` was chosen because it offers strong price and performance with low latency and sufficient reasoning for FAQ-style answers.

System responses are constrained to the context of the FAQ document to ensure accuracy, relevance, transparency, and verifiability.

Here is an example of a system response for a question not covered in the FAQ:

User Query:
```bash
python -m src.query "What is the company's policy on sabbaticals?"
```
System Response:
```json
{
    "model": "gpt-4o-mini",
    "timestamp": "2025-11-11T11:20:27.043479+00:00",
    "tokens_prompt": 599,
    "tokens_completion": 38,
    "total_tokens": 637,
    "latency_ms": 2106.55,
    "estimated_cost_usd": 0.000113,
    "cache_hit": false,
    "actions": [
      "Suggest the employee contact HR for more information",
      "Direct the employee to check the employee handbook"
    ],
    "source": [
      "fallback"
    ],
    "user_question": "What is the company's policy on sabbaticals?",
    "system_answer": "We do not have information regarding sabbatical policies in our current FAQs.",
    "chunks_related": [
      "Time Off & Leave Policies\nQ: How do I request time off?\nA: Log into the employee portal, go to \"Time Off\" > \"Request Time Off,\" select your dates, choose the leave type (vacation, sick, personal), and submit. Requests are automatically routed to your manager for approval. You'll receive an email notification once approved or denied.\n\nQ: How much paid time off (PTO) do I accrue?\nA: Full-time employees accrue PTO at the following rates: 0-2 years: 15 days/year, 3-5 years: 20 days/year, 6+ years: 25 days/year. PTO accrues monthly and can be used after 90 days of employment. Part-time employees accrue PTO on a pro-rated basis.\nSource: data/faq_document.txt",
      "Q: Can I carry over unused PTO to the next year?\nA: You can carry over up to 5 days of unused PTO to the next calendar year. Any PTO in excess of 5 days will be forfeited if not used by December 31st. Carry-over requests must be submitted by November 15th.\n\nQ: What is the difference between sick leave and personal time?\nA: Sick leave is for illness, medical appointments, or caring for a sick family member. Personal time is for non-medical personal matters. Both count toward your PTO balance, but sick leave may be used without prior approval in emergency situations. Personal time requires manager approval in advance.\nSource: data/faq_document.txt",
      "Employee Onboarding & Account Setup\nQ: How do I access my employee portal for the first time?\nA: New employees receive a welcome email with login credentials and a link to the employee portal. If you haven't received this email, contact HR at hr@company.com or call extension 1234. You'll need to set up two-factor authentication on your first login.\n\nQ: I forgot my password. How do I reset it?\nA: Click \"Forgot Password?\" on the login page and enter your work email address. You'll receive a password reset link within 5 minutes. If you don't receive the email, check your spam folder or contact IT support at it-support@company.com.\nSource: data/faq_document.txt"
    ],
    "confidence": 20,
    "evaluation": {
      "overall_score": 5.4,
      "quality_level": "poor",
      "relevance_score": 2,
      "accuracy_score": 10,
      "completeness_score": 2,
      "clarity_score": 8,
      "improvements": [
        "Include a suggestion for the user to contact HR for more information on sabbatical policies.",
        "Provide a brief explanation of where such policy information might typically be found, such as in the employee handbook or by contacting HR directly."
      ]
    }
  }
  ```

Here is an example of a system response from within the FAQ context:

User Query:
  ```bash
  python -m src.query "How do I request time off?"
  ```
System Response:
  ```json
  {
    "model": "gpt-4o-mini",
    "timestamp": "2025-11-11T11:22:08.523218+00:00",
    "tokens_prompt": 573,
    "tokens_completion": 49,
    "total_tokens": 622,
    "latency_ms": 3200.42,
    "estimated_cost_usd": 0.000115,
    "cache_hit": false,
    "actions": [
      "Guide the employee to the Time Off section in the employee portal",
      "Confirm the dates and leave type for the request"
    ],
    "source": [
      "Time Off & Leave Policies"
    ],
    "user_question": "How do I request time off?",
    "system_answer": "Log into the employee portal, go to \"Time Off\" > \"Request Time Off,\" select your dates, choose the leave type (vacation, sick, personal), and submit. Requests are automatically routed to your manager for approval. You'll receive an email notification once approved or denied.",
    "chunks_related": [
      "Time Off & Leave Policies\nQ: How do I request time off?\nA: Log into the employee portal, go to \"Time Off\" > \"Request Time Off,\" select your dates, choose the leave type (vacation, sick, personal), and submit. Requests are automatically routed to your manager for approval. You'll receive an email notification once approved or denied.\n\nQ: How much paid time off (PTO) do I accrue?\nA: Full-time employees accrue PTO at the following rates: 0-2 years: 15 days/year, 3-5 years: 20 days/year, 6+ years: 25 days/year. PTO accrues monthly and can be used after 90 days of employment. Part-time employees accrue PTO on a pro-rated basis.\nSource: data/faq_document.txt",
      "Q: Can I carry over unused PTO to the next year?\nA: You can carry over up to 5 days of unused PTO to the next calendar year. Any PTO in excess of 5 days will be forfeited if not used by December 31st. Carry-over requests must be submitted by November 15th.\n\nQ: What is the difference between sick leave and personal time?\nA: Sick leave is for illness, medical appointments, or caring for a sick family member. Personal time is for non-medical personal matters. Both count toward your PTO balance, but sick leave may be used without prior approval in emergency situations. Personal time requires manager approval in advance.\nSource: data/faq_document.txt",
      "Q: How do I check my current PTO balance?\nA: Log into the employee portal and navigate to \"Time Off\" > \"My Balance.\" You'll see your accrued PTO, used PTO, pending requests, and available balance. The balance updates in real-time as requests are approved.\n\nQ: What is the company's policy on unpaid leave?\nA: Unpaid leave may be granted for extended personal circumstances, medical emergencies, or family care situations. Requests must be submitted at least 30 days in advance (except emergencies) and require approval from both your manager and HR. Unpaid leave is limited to 12 weeks per calendar year.\nSource: data/faq_document.txt"
    ],
    "confidence": 90,
    "evaluation": {
      "overall_score": 9.1,
      "quality_level": "excellent",
      "relevance_score": 10,
      "accuracy_score": 10,
      "completeness_score": 8,
      "clarity_score": 9,
      "improvements": []
    }
  }
  ```


Some major factors to consider carefully in AI systems are scaling and costs.
To minimize spending, which can be directly linked to prompt length, the system implements measures that reduce token length and cut API calls when possible. They are as follows:
1. Prompt truncation to keep the prompt's length within model limit and only use what’s necessary to give the LLM the context it needs to answer the question effectively.
2. Response caching to eliminate API calls for repeated user queries. Since the context of this project is to answer high volume of employee questions daily, caching is essential to reduce costs. The cache uses a composite key made from a normalized user query and a deterministic hash of the retrieved context (the chunks used in the prompt), to map semantically identical questions with the same context to the same entry. The cache persists to `storage/response_cache.json`.
3. Chunk retrieval is limited to top 3 to balance cost and answer quality.


### Evaluator
The system includes an LLM-based (`gpt-4o-mini`) automated evaluator that scores each answer on a scale of 0-10 across relevance, accuracy, completeness, clarity, and provides an overall score with suggested improvements. Results are logged with the query response metrics for monitoring and future iteration.

### Performance and Scalability
- Current dataset size: This project can easily support data size in the thousands. Therefore, the flat FAISS index (using cosine similarity) is sufficient for this project because it provides 100% recall with negligible latency for this use case. Additionally, it can scale well into the low tens of thousands of chunks before advanced FAISS indexes (IVF/HNSW) are necessary.
- The system tracks latency and costs per request to enable data-driven tuning.

### Testing
This project includes a test suite to ensure the application works as designed.
Use the following command to run the tests:
```bash
pytest tests/test_core.py -v
```

The test coverage includes:
- Query normalization, caching
- Context truncation (Q&A preservation, token and chunk limits)
- Document retrieval relevance and metadata
- Answer generation (prompt formatting, LLM response parsing)
- Evaluator scoring and logging
