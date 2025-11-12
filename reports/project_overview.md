## Project Overview and Report
This project builds on the knowledge gained from Module 1.<br>
It uses Retrieval-Augmented Generation (RAG) to implement a customer inquiry mechanism that answers questions based on the company’s internal data to provide context and generate referenceable responses.
Rather than implementing the entire pipeline from scratch, this project leverages the LangChain library to implement the RAG pipeline. This standardizes the codebase to make the artifact production-ready and easier to scale and maintain with multiple contributors. Additionally, LangChain's open-source components are used to chunk documents, embed, and store them locally for free. Only LLM API calls may incur charges.

In order to avoid bloating the project and focus on the core concept of RAG, the following assumptions apply:
1. Moderation has been fully implemented and tests for it are available.
2. Rate limiting has been properly set up to prevent service abuse.

This RAG project builds a customer inquiry system that answers questions using company's internal data to provide accurate, informed, and verifiable responses.
It starts by building the data pipeline, which takes the company document(s) such as FAQs and Policies and splits them into chunks.
Next, an embedding is generated for each chunk and stored in FAISS, a vector storage.

Features
- LangChain-based RAG pipeline (FAISS + HuggingFace embeddings + OpenAI LLM)
- Smart context truncation preserving complete Q&A pairs to control tokens and cost
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
- `prompts/main_prompt.txt`: The underlying system prompt for constrained, source-grounded answers. I used few-shot examples to guide the LLM to provide answers based on the 3 top relevant FAQ chunks for the user's query.
- `storage`: Directory for storing the FAISS index and response cache.
- `outputs`: Directory for storing the sample queries and evaluation results.

### Data Pipeline
Steps
1. Chunking
   - The FAQ document is split using a sliding window (`chunk_size=1000`, `chunk_overlap=200`) through `RecursiveCharacterTextSplitter` to maintain chunk context between Q&A pairs.
   - Each chunk carries metadata: `id`, `document_id`, `chunk_index`, `token_count`, `char_count`, and `created_at`. These additional data can be useful for extending functionalities to include keyword searches or retrieval filtering.
2. Embedding
   - Text is converted to vectors using `all-MiniLM-L6-v2` (which has a 384-dimensional vector space) via `langchain-huggingface`.
3. Indexing and Storage
   - Chunks and embeddings are stored in a FAISS index (`IndexFlatL2`) for high-recall semantic search.

Data Pipeline Build Command:
```bash
python -m src.build_index data/faq_document.txt
```



### Query Pipeline
With the data pipeline in place, the query process answers user questions using semantic retrieval over the indexed FAQ chunks.
<br><br>Steps:
1. Read and embed the user query via the CLI argument.
2. Retrieve the top-k (k=3) relevant chunks using the embedded user query via FAISS (Euclidean distance) similarity search.
3. Apply truncation to preserve complete Q&A pairs, remove redundancy, and keep prompts within token budgets.
4. Invoke the LLM (`gpt-4o-mini`) with a structured system prompt and the truncated context.
5. Get and log metrics and responses to `outputs/sample_queries.json` including `cost`, `latency`, `timestamp`, `user_question`, `system_answer`, `actions`, `related chunks`, `source`, `confidence`, and `evaluation scores`.

Query Pipeline Command:
```bash
python -m src.query "User query"
```

<strong>Model choice:</strong> I chose `gpt-4o-mini` because it offers strong price and performance with low latency and sufficient reasoning for FAQ-style answers.

System responses are constrained to the context of the FAQ document to ensure accuracy, relevance, transparency, and verifiability.
Here is an example of system response out of FAQ context.

User Query:
```bash
python -m src.query "Do I get discounts for bulk purchases?"
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
      "Suggest the user contact customer support for specific inquiries",
      "Check for any current promotions or sales"
    ],
    "source": [
      "fallback"
    ],
    "user_question": "Do I get discounts for bulk purchases?",
    "system_answer": "We do not have information regarding discounts for bulk purchases in our current FAQs.",
    "chunks_related": [
      "Q: Who pays for return shipping?\nA: We provide a pre-paid return label for items that are faulty or incorrect due to our error. For returns based on change of mind, the return shipping cost is the responsibility of the customer and will be deducted from your refund.\n\nAccount & Technical Issues\nQ: How do I reset my password?\nA: Click on \"Log In\" and then the \"Forgot Password?\" link. Enter the email address associated with your account, and we will send you a link to create a new password.\n\nQ: Can I check out as a guest?\nA: Yes, you can! However, creating an account allows you to track your orders easily, save your address for faster checkout, and view your order history.\nSource: data/faq_document.txt",
      "Returns, Exchanges & Refunds\nQ: What is your return policy?\nA: We offer a 30-day return policy from the date of delivery for most items in original, unworn, and resalable condition, with original tags and packaging. For hygiene reasons, some items (like underwear, swimwear, and pierced jewelry) are final sale unless faulty.\n\nQ: How do I start a return or exchange?\nA: The easiest way is to initiate the process through your account on our website under \"Order History.\" Alternatively, you can contact our support team, and we will guide you through the steps and provide a return shipping label if applicable.\n\nQ: How long does it take to process a refund?\nA: Once we receive your returned item, it takes 3-5 business days to inspect and process the refund. The time it takes for the refund to appear in your account will depend on your bank or payment provider (typically 5-10 additional business days).\nSource: data/faq_document.txt",
      "Q: Can I check out as a guest?\nA: Yes, you can! However, creating an account allows you to track your orders easily, save your address for faster checkout, and view your order history.\n\nQ: The website is not working correctly on my browser. What should I do?\nA: We recommend trying a hard refresh (Ctrl+F5 on Windows, Cmd+Shift+R on Mac), clearing your browser's cache and cookies, or trying a different browser like Chrome or Firefox. If the problem persists, please contact our support team and let us know what device and browser you are using.\n\nProduct & Inventory\nQ: How can I be notified when an out-of-stock item is back?\nA: On the product page of the item you want, simply click the \"Notify Me When Available\" button and enter your email address. We'll send you an alert the moment it's back in stock!\nSource: data/faq_document.txt"
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
        "Include a suggestion for the user to contact customer support for more information on bulk purchase discounts.",
        "Provide a brief explanation of where bulk purchase information might typically be found, such as on the website or in promotional materials."
      ]
    }
  }
  ```

Here is an example of system response in context of FAQ.

User Query:
  ```bash
  python -m src.query "What are your shipping options?"
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
      "Contact the user for their order number",
      "Check order status to confirm if it can be canceled"
    ],
    "source": [
      "Ordering & Payment"
    ],
    "user_question": "Can I cancel my order?",
    "system_answer": "You can cancel your order only if it hasn't yet entered the packing process. Please contact us immediately with your order number.",
    "chunks_related": [
      "Ordering & Payment\nQ: What payment methods do you accept?\nA: We accept all major credit/debit cards (Visa, MasterCard, American Express), PayPal, Apple Pay, and Google Pay. All payments are processed through a secure, encrypted gateway for your safety.\n\nQ: Is my payment information secure?\nA: Absolutely. We do not store your full payment details on our servers. All transactions are processed by PCI-DSS compliant payment gateways, ensuring the highest level of security.\n\nQ: Can I modify or cancel my order after placing it?\nA: We can only modify or cancel an order if it hasn't yet entered the packing process. Please contact us immediately at support@yourcompany.com or call us at [Your Phone Number] with your order number. We cannot guarantee changes once an order is being prepared for shipment.\nSource: data/faq_document.txt",
      "Q: Who pays for return shipping?\nA: We provide a pre-paid return label for items that are faulty or incorrect due to our error. For returns based on change of mind, the return shipping cost is the responsibility of the customer and will be deducted from your refund.\n\nAccount & Technical Issues\nQ: How do I reset my password?\nA: Click on \"Log In\" and then the \"Forgot Password?\" link. Enter the email address associated with your account, and we will send you a link to create a new password.\n\nQ: Can I check out as a guest?\nA: Yes, you can! However, creating an account allows you to track your orders easily, save your address for faster checkout, and view your order history.\nSource: data/faq_document.txt",
      "Q: Why was my payment declined?\nA: Payment can be declined for several reasons, including insufficient funds, incorrect card information, or a security check by your bank. Please verify your details and try again, or contact your bank for more information.\n\nShipping & Delivery\nQ: What are your shipping options and costs?\nA: We offer several shipping options at checkout, with costs calculated based on your location, package weight, and delivery speed. Standard shipping (3-7 business days) often has a flat rate or is free on orders over a certain amount. Express and Next-Day options are also available.\n\nQ: How long will it take to receive my order?\n\nStandard Shipping: 3-7 business days.\n\nExpress Shipping: 2-3 business days.\n\nNext-Day Shipping: Order by 1 PM local time for delivery the next business day (where available).\n\n*Please note: Processing time (1-2 business days) is separate from shipping time.*\nSource: data/faq_document.txt"
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


One of the major factors to consider carefully in AI systems is costs.
To minimize spending, which can be directly linked to prompt length, I took measures that reduce token length and cut API calls when possible. They are as follows:
1. Prompt truncation to keep the prompt's length within model limit and only use what’s necessary to give the LLM the context it needs to answer the question effectively.
2. Response caching to eliminate API calls for repeated user queries. Since the context of this project is to answer high volume of customer queries, caching is a must to reduce costs. The cache uses a composite key made from a normalized user query and a deterministic hash of the retrieved context (the chunks used in the prompt), to map semantically identical questions with the same context to the same entry. The cache persists to `storage/response_cache.json`.
3. Chunk retrieval is limited to top 3 to balance cost and answer quality.


### Evaluator
I implemented an LLM-based (`gpt-4o-mini`) automated evaluator that scores each answer on a scale of 0-10 across relevance, accuracy, completeness, clarity, and gives an overall score with suggested improvements. Results are logged with the query response metrics for monitoring and future iteration.

### Performance and Scalability
- Current dataset size: Since the data size is in the thousands, the `IndexFlatL2` (Euclidean distance) is sufficient for this project because it provides 100% recall with negligible latency for this use case. Additionally, it can scale well into the low tens of thousands of chunks before advanced FAISS indexes (IVF/HNSW) are necessary.
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
