# NASA Mission Intelligence: RAG Chat with Live Evaluation

A chat assistant that answers questions about the Apollo 11, Apollo 13 and Challenger (STS-51L) missions from 12
mission transcripts and reports. Answers cite the passages they came from, the assistant says plainly when the
documents don't cover a question, and every answer is scored live for faithfulness and relevance with RAGAS.

Built for the RAG project in Udacity's Generative AI program.

| | |
|---|---|
| **Data** | 12 text files: Apollo 11 (6), Apollo 13 (3), Challenger (3) |
| **Index** | ChromaDB, 8,008 chunks of about 1,000 characters with a 200-character overlap, `text-embedding-3-small`, cosine distance |
| **Answers** | `gpt-3.5-turbo` by default (GPT-4 can be chosen in the app), temperature 0.3, with `[Source N]` citations |
| **Evaluation** | RAGAS 0.4.3: Response Relevancy and Faithfulness, shown in the sidebar after each answer |
| **Interface** | Streamlit |

## How it works

1. **Indexing** (`embedding_pipeline.py`):
   - Splits each file into chunks at sentence endings, overlapping neighbouring chunks so a fact on a boundary
     stays whole in one of them.
   - Embeds new chunks 50 per API call and stores them in ChromaDB, tagged with their mission and source.
   - Gives each chunk a stable id such as `apollo11_<file>_chunk_0007`, so a re-run skips, updates or replaces
     existing chunks instead of duplicating them.
2. **Retrieval** (`rag_client.py`): finds the closest passages to the question. If the question names a mission,
   only that mission is searched. A follow-up that names no mission uses the mission from the latest question
   that did.
3. **Answering** (`llm_client.py`): the model answers from the retrieved passages only, cites them, says when
   they don't contain the answer, and labels anything that comes from general knowledge. The 10 most recent
   messages are sent along, so follow-ups like "what did he say next?" work.
4. **Evaluation** (`ragas_evaluator.py`): scores each answer, using two metrics that don't need a correct answer
   to compare against, since a live chat has none:
   - **Faithfulness:** is every claim supported by the retrieved passages?
   - **Response Relevancy:** does the answer address the question?

## What testing turned up

| Problem | Fix |
|---|---|
| Apollo 13 questions retrieved Apollo 11 passages, because the Apollo 11 report also discusses an oxygen-tank problem | Retrieval is filtered to the mission the question names |
| Re-running the pipeline with a different chunk size added 8,428 smaller chunks next to the original 8,008, mixing two chunkings in one index | The extra chunks were found by timestamp and deleted. Re-run with the original settings, or use `--update-mode replace` |
| RAGAS scored error messages from failed generations as if they were answers | Failed answers are marked "Not evaluated" |
| Requests through the Vocareum proxy timed out or went to the wrong endpoint | The `.env` file is loaded before any client is created, and timeouts were raised to 120 seconds |

[`REPORT.md`](REPORT.md) has sample questions with their scores and the full list of fixes. Its examples
include a correct refusal (the cost of Apollo 11 isn't in the documents) and an answer that faithfulness flagged
for leaning on the model's own knowledge.

## Running it

Needs Python 3.10 or newer and an OpenAI API key.

```bash
cd Project-NASA-Mission-Intelligence-Starter
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in this folder:

```bash
OPENAI_API_KEY="your-key"
OPENAI_BASE_URL="https://openai.vocareum.com/v1"   # only if you use the Vocareum proxy
```

### Fix the RAGAS import

RAGAS 0.4.3 imports `ChatVertexAI` and `VertexAI` from `langchain-community`, but newer LangChain versions moved
them to `langchain-google-vertexai`. Without this fix, importing RAGAS fails with
`ImportError: cannot import name 'ChatVertexAI'`. Patch the installed package once:

```bash
# find the file without importing ragas, since the import is what fails
RAGAS_BASE=$(python -c "import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec('ragas').origin), 'llms', 'base.py'))")
sed -i.bak 's/from langchain_community.chat_models.vertexai import ChatVertexAI/from langchain_google_vertexai import ChatVertexAI/' "$RAGAS_BASE"
sed -i.bak 's/from langchain_community.llms import VertexAI/from langchain_google_vertexai import VertexAI/' "$RAGAS_BASE"
python -W ignore -c "import ragas; print('RAGAS:', ragas.__version__)"   # expect: RAGAS: 0.4.3
```

The patch has to be applied again if RAGAS is reinstalled. A screenshot of the change is in
[`fix_ragas.png`](fix_ragas.png).

### Build the index, then start the app

The index isn't committed, so build it first. This takes a few minutes and embeds about 8,000 chunks:

```bash
set -a; source .env; set +a
python embedding_pipeline.py --openai-key "$OPENAI_API_KEY" --data-path data_text \
    --chunk-size 1000 --chunk-overlap 200
streamlit run chat.py
```

Pass `--chunk-size 1000 --chunk-overlap 200` explicitly. The script's defaults are 500 and 100, and mixing
settings in one index causes the duplication described above. Other options:
- `--update-mode skip|update|replace` controls what a re-run does with existing chunks;
- `--stats-only` shows what's in the index;
- `--delete-source <pattern>` removes one source's chunks;
- `--test-query "..."` runs a search after indexing.

In the app's sidebar you can choose the model, the number of passages to retrieve (3 by default), and the
mission filter (auto-detect by default), and turn evaluation off.

For a single question without the app: `python ask.py "Who was the commander of Apollo 11?"`

### Checking the answering

`python test_llm_client.py` calls the model on three cases:
- no context, where it should say it has no documents;
- an answerable question, where it should answer with citations;
- a question the context can't answer, where it should decline instead of inventing a figure.

It prints each result next to what to expect. It needs the API key.

## Known limitations

- **Missing source:** the Apollo 13 files are transcripts, with no accident report, so the system can't explain
  *why* the oxygen tank failed.
- **Comparisons:** a single search for a multi-mission question, such as "compare Apollo 11 and 13", can return
  passages from only one mission. Searching each mission and merging the results would balance them.
- **Evaluation time:** scoring takes about 9 seconds per answer. Answers show up in about 2 seconds, and the
  scores follow.
- **`--batch-size` has no effect:** the option is parsed but not passed through, so batches are always 50.

## Files

| File | Contents |
|---|---|
| `embedding_pipeline.py` | Chunking, embedding and indexing, with command-line options |
| `rag_client.py` | Retrieval and the mission filter |
| `llm_client.py` | The answering prompt and the OpenAI call |
| `ragas_evaluator.py` | Faithfulness and relevancy scoring |
| `chat.py` | The Streamlit app |
| `ask.py`, `test_llm_client.py` | A command-line question, and checks on the answering |
| `data_text/` | The 12 mission documents |
| `REPORT.md` | Challenges, sample answers with scores, testing and next steps |
