┌─────────────────────────────────────────────────────────────┐
│  STEP 1: INGESTION (youtube_loader.py)                      │
│  User provides YouTube URL → Extract Video ID →             │
│  Fetch transcript using YouTubeTranscriptApi               │
│  Output: Raw transcript text                                │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: CHUNKING (chunker.py)                              │
│  Split transcript into smaller chunks                       │
│  • Chunk size: 100 characters                               │
│  • Tool: RecursiveCharacterTextSplitter                     │
│  Output: List of document chunks                            │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: EMBEDDING (embedder.py) ⭐ KEY STEP               │
│  Convert each chunk into vector embeddings                  │
│  • Model: nomic-embed-text:latest (Ollama)                 │
│  • Creates: 768D vector per chunk                           │
│  • Why: Enables semantic similarity search                  │
│  Output: Vector representations of chunks                   │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: VECTOR STORAGE (vector_store.py) ⭐              │
│  Store embeddings in ChromaDB with deduplication            │
│  • Dedup: MD5 hash of (video_id + text)                    │
│  • Persistent storage: VectorDataBase/chroma_db/            │
│  • Prevents duplicate embeddings (saves 30-40%)             │
│  Output: Persistent vector database                         │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: RETRIEVAL (retriever.py) ⭐ EMBEDDING USED       │
│  When user asks a question:                                 │
│  1. Convert user query to embedding                         │
│  2. Find 4 most similar chunks (k=4)                       │
│  3. Use cosine similarity in vector space                   │
│  Output: Top 4 most relevant chunks                         │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: FORMATTING (formatter.py)                          │
│  Join retrieved chunks with newlines                        │
│  Output: Context string for LLM                             │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  STEP 7: LLM GENERATION (generator.py + prompts.py)        │
│  • Build prompt with context + question                     │
│  • Send to Google Gemini 2.5 Flash LLM                      │
│  • LLM generates final answer from context only             │
│  Output: High-quality answer to user                        │
└─────────────────────────────────────────────────────────────┘



 for searching 


CHUNK EMBEDDINGS (Used only for SEARCH):



┌─────────────────┐
│ Chunk 1 text    │ → [Embedding vector] ✓ STORED in ChromaDB
├─────────────────┤
│ Chunk 2 text    │ → [Embedding vector] ✓ STORED in ChromaDB
├─────────────────┤
│ Chunk 3 text    │ → [Embedding vector] ✓ STORED in ChromaDB
└─────────────────┘

QUERY TIME:
┌─────────────────────────────────────┐
│ User Question                        │
│ "What is the main topic?"            │ → Convert to [Embedding vector]
└─────────────────────────────────────┘
             ↓ (SEMANTIC SEARCH in vector space - 768D)
        Find 4 most similar embeddings using cosine similarity
             ↓
┌────────────────────────────────────────────────────┐
│ TOP 4 RESULT CHUNKS (RAW TEXT ONLY)                │
│ ❌ NOT their embeddings                            │
│ ✓ Their ACTUAL TEXT content                        │
│                                                    │
│ Chunk A text content...                            │
│ Chunk B text content...                            │
│ Chunk C text content...                            │
│ Chunk D text content...                            │
└────────────────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────┐
│ BUILD PROMPT FOR LLM:                              │
│                                                    │
│ "You are a helpful assistant...                   │
│  Context: [Chunk A text + B + C + D RAW TEXT]     │
│  Question: What is the main topic?"               │
│                                                    │
│ ❌ Query embedding NOT passed                      │
│ ❌ Chunk embeddings NOT passed                     │
│ ✓ Query TEXT passed                               │
│ ✓ Chunk TEXT passed                               │
└────────────────────────────────────────────────────┘
             ↓
         LLM (Gemini) processes text and generates answer

Another one 


STEP 1: TEXT CREATION
┌─────────────────────────────────────┐
│ Original Chunk Text                 │
│ "The video discusses machine        │
│  learning concepts"                 │
└──────────┬──────────────────────────┘
           ↓
STEP 2: EMBEDDING GENERATION (nomic-embed-text)
┌─────────────────────────────────────┐
│ Convert text to 768D vector:        │
│ [0.234, -0.456, 0.789, ... 768 numbers]
└──────────┬──────────────────────────┘
           ↓
STEP 3: STORE BOTH (in ChromaDB)
┌──────────────────────────────────────────┐
│ ChromaDB stores:                         │
│ • Original TEXT                          │
│ • Its EMBEDDING (768D vector)            │
└──────────┬───────────────────────────────┘
           ↓
STEP 4: USER ASKS QUESTION
┌──────────────────────────────────────────┐
│ Query: "machine learning?"               │
└──────────┬───────────────────────────────┘
           ↓
STEP 5: EMBEDDING USED FOR SEARCH
┌──────────────────────────────────────────┐
│ 1. Convert query to embedding            │
│    [0.241, -0.450, 0.785, ... 768]      │
│ 2. Compare with all chunk embeddings     │
│ 3. Find 4 most similar (cosine distance) │
└──────────┬───────────────────────────────┘
           ↓
STEP 6: RETURN THE TEXT (not embedding!)
┌──────────────────────────────────────────┐
│ Top 4 Chunks returned WITH ORIGINAL TEXT:│
│                                          │
│ Chunk 1: "The video discusses..."       │
│ Chunk 2: "ML covers algorithms..."      │
│ Chunk 3: "Deep learning is..."          │
│ Chunk 4: "Neural networks process..."   │
└──────────┬───────────────────────────────┘
           ↓
STEP 7: PASS TEXT TO LLM
┌──────────────────────────────────────────┐
│ LLM receives TEXT content only           │
│ NOT the embeddings                       │
│ Generates answer based on TEXT           │
└──────────────────────────────────────────┘