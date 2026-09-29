# NASA Mission Intelligence — Submission Report

A retrieval-augmented chat assistant for NASA mission documents, with real-time answer quality scoring.

| | |
|---|---|
| **Data** | 12 text files: Apollo 11 (6), Apollo 13 (3), Challenger STS-51L (3) |
| **Index** | ChromaDB, 8,008 chunks of ~1,000 characters (200 overlap), `text-embedding-3-small`, cosine distance |
| **Answers** | `gpt-3.5-turbo`, temperature 0.3, cites passages as `[Source N]` |
| **Evaluation** | RAGAS 0.4.3: Response Relevancy and Faithfulness |
| **Interface** | Streamlit (`chat.py`) |

## How to run

```bash
set -a; source .env; set +a        # OPENAI_API_KEY and OPENAI_BASE_URL (Vocareum proxy)

# Build the index (re-runs skip chunks that already exist)
python embedding_pipeline.py --openai-key "$OPENAI_API_KEY" --data-path data_text \
    --chunk-size 1000 --chunk-overlap 200

streamlit run chat.py
```

## 1. Challenges and solutions

**Requests to the Vocareum proxy failed.** OpenAI clients read `OPENAI_BASE_URL` when they are created, so the pipeline calls `load_dotenv()` before building any client; otherwise requests went to api.openai.com and the key was rejected. The proxy can also take ~15 s to connect, longer than the client's 5 s connect timeout, so timeouts were raised to 120 s (60 s to connect).

**Chunks had to be meaningful and re-runs safe.** `chunk_text` cuts at the nearest sentence ending before the size limit and overlaps neighbouring chunks, so a fact on a boundary survives whole in one of them; it also always advances, whatever the size and overlap settings. Chunk IDs are stable (`<mission>_<file>_chunk_0007`), so a re-run skips, updates or replaces existing chunks instead of duplicating them.

**Re-running with different chunk settings mixed two chunkings.** Because IDs only record the chunk number, re-running at 500 characters on a 1,000-character index skipped the low-numbered chunks and added 8,428 smaller ones covering the same text. The added chunks were identified by their processing timestamp and deleted, restoring the original 8,008. Lesson: re-run with the original `--chunk-size`/`--chunk-overlap`, or use `--update-mode replace` when changing them.

**Embedding 8,000+ chunks one at a time was slow.** New chunks are embedded 50 per API call. Each file is processed in its own `try`, so one bad file is logged and counted without stopping the run.

**Most RAGAS metrics need a reference answer.** BLEU, ROUGE and context precision "with reference" compare against a ground-truth answer, which a live chat does not have. The evaluator uses the two reference-free metrics: Response Relevancy (does the answer address the question?) and Faithfulness (is each claim supported by the retrieved passages?). Also fixed: the TODO's model name `test-embedding-3-small` is a typo for `text-embedding-3-small`, and a `NaN` score (RAGAS could not parse the judge's output) is reported as an error because the sidebar's progress bar rejects it.

**Apollo 13 questions retrieved Apollo 11 passages.** Unfiltered search returned 2 of 3 passages from the Apollo 11 mission report, which also discusses an oxygen-tank anomaly, so Apollo 11 details could be presented as Apollo 13 facts. Retrieval is now filtered to the mission a question names (see §2).

**Evaluation scored error messages.** A failed generation returns error text, which RAGAS scored as if it were an answer. It is now marked "Not evaluated", and the evaluator uses the API key entered in the sidebar rather than the one in `.env`.

**`get_collection_info` returning `None` crashed the end of a run.** `main()` calls `.get()` on the result after all embedding work is done, so errors return an empty dict instead.

## 2. Additional features and improvements

- **Mission filter with auto-detection.** Questions naming Apollo 11, Apollo 13 or Challenger/STS-51L search only that mission; follow-ups that name none inherit the mission of the latest earlier question that does; questions naming several missions search all of them. A sidebar control overrides it.
- **Grounded answering.** The system prompt requires `[Source N]` citations, saying plainly when the documents lack the answer, and labelling any general knowledge.
- **Bounded history.** The 10 most recent messages are sent with each question, so follow-ups work without long chats overflowing the model's context.
- **Update modes.** `--update-mode skip | update | replace` controls how re-runs treat existing chunks; `--stats-only`, `--delete-source` and `--test-query` support maintenance.
- **Graceful failures.** Missing key: the app warns and hides the chat. Invalid key: the error appears as a chat reply, with no crash. A failing metric does not discard the others.

## 3. Sample queries and responses

Actual output with the default settings (3 passages, auto-detected mission filter). Scores run from 0 to 1.

| # | Question | Expected | Actual response (abridged) | Relevancy | Faithfulness |
|---|---|---|---|---|---|
| 1 | Who was the commander of Apollo 11? | Neil Armstrong, cited | "The commander of Apollo 11 was Neil A. Armstrong [Source 1]." | 1.000 | 1.000 |
| 2 | *(follow-up)* What did he say when he first stepped onto the Moon? | Resolves "he" to Armstrong; quotes him | "…he famously said, 'This is one small step for a man, one giant leap for mankind' [Source 1]." | 0.892 | 1.000 |
| 3 | What was the Challenger's altitude and velocity shortly before the accident? | Figures from the flight audio | "…altitude was 9 nautical miles, and its velocity was 2900 ft per second [Source 2]." | 0.993 | 1.000 |
| 4 | What did mission control say right after the Challenger broke up? | Controllers' statements from the audio | "…flight controllers… reported that they did not see anything unusual up to that point [Source 1]." | 0.839 | 0.500 |
| 5 | How much did the Apollo 11 mission cost? | Declines: not in the documents | "The cost of the Apollo 11 mission was not explicitly mentioned in the provided sources…" | 0.000 | 1.000 |
| 6 | What went wrong with the oxygen tank on Apollo 13? | The tank explosion and its cause | Describes a burst disk rupturing at ~1937 psi [Source 2]. | 0.749 | 0.750 |
| 7 | Compare the Apollo 11 and Apollo 13 missions. | Balanced comparison from both missions | A structured comparison, but all 3 passages were from Apollo 13; most Apollo 11 points came from general knowledge. | 0.838 | 0.350 |

What the scores show:

- **1–3** are grounded answers, and faithfulness confirms it. Query 2 shows history working.
- **5** is the correct behaviour, a refusal rather than an invented figure. Response Relevancy scores refusals near 0 by design, so a low relevancy score is not always a bad answer.
- **6** is limited by the data, not the code. The Apollo 13 files are transcripts and contain no accident report, so the cause of the explosion is not in the index. The retrieved passages describe the lunar module helium tank's burst disk, which the model mislabelled as the oxygen tank.
- **7** shows faithfulness flagging an answer that leans on the model's own knowledge.

### More fully grounded examples

Each of these is factually correct, cited, and fully supported by the retrieved passages (faithfulness 1.000). Each was asked in a new conversation.

| # | Question | Expected | Actual response | Relevancy | Faithfulness |
|---|---|---|---|---|---|
| 8 | When did Apollo 11 launch? | July 16, 1969, from Kennedy Space Center | "Apollo 11 launched on July 16, 1969, at 8:32 a.m. e.s.t. from Kennedy Space Center, Florida [Source 1][Source 2]." | 1.000 | 1.000 |
| 9 | What was the call sign of the Apollo 11 lunar module? | Eagle | "The call sign of the Apollo 11 lunar module was "Eagle" [Source 3]." | 1.000 | 1.000 |
| 10 | Who was the lunar module pilot on Apollo 11? | Buzz Aldrin | "The Lunar Module Pilot on Apollo 11 was Edwin E. Aldrin, Jr. [Source 1]" | 0.987 | 1.000 |
| 11 | Who were the crew members of Apollo 13? | Lovell, Swigert, Haise | "Commander James A. (Jim) Lovell, Jr.; Command Module Pilot John L. (Jack) Swigert, Jr.; Lunar Module Pilot Fred W. Haise, Jr. [Source 1]" | 1.000 | 1.000 |
| 12 | How did the Apollo 13 crew use the lunar module after the accident? | As a lifeboat for the return to Earth | "…the crew used the lunar module as a lifeboat to come back to Earth using its consumables for life support [Source 1]… using the lunar module's descent propulsion system for a transearth injection burn [Source 3]." | 0.719 | 1.000 |
| 13 | What did mission control tell Challenger at throttle up? | "Challenger, go at throttle up" | "Mission control told Challenger "Challenger go with throttle up" at throttle up [146:26] [146:30]…" | 0.846 | 1.000 |

Notes:

- **12** scores lower on relevancy only because the answer also covers the return burn, which goes beyond the question.
- **13** quotes the transcript's own wording ("go with throttle up"). It cites transcript timestamps instead of `[Source N]`, because the Challenger audio chunks carry timestamps like `[146:26]`.

## 4. Testing summary

All three README checkpoints were verified by driving the running app in a headless browser and by scripted calls through the same code:

| Checkpoint | Result |
|---|---|
| 1. Basic functionality | LLM client (4/4 behaviours), backend discovery, pipeline run (12 files, 0 errors), RAGAS scores |
| 2. Integration | End-to-end chat, live sidebar metrics, missing and invalid key handling |
| 3. Advanced features | Each mission filter returned 10/10 passages from its mission; follow-ups resolved; history capped at 10 messages; full re-run of 16,436 chunks with 0 errors |

Performance, median of 3 questions: search 0.7 s, answer on screen **2.0 s**, scores on screen 11.4 s. Evaluation is ~80% of the total and runs after the answer is shown; it can be switched off in the sidebar.

## 5. Known limitations and next steps

- **Coverage:** adding the Apollo 13 Review Board report would let the system answer why the tank failed.
- **Comparisons:** a single search for a multi-mission question can return passages from one mission only. Retrieving per mission and merging would balance them.
- **Pipeline:** `--batch-size` is parsed but not passed through (batches are always 50), and skip mode checks each chunk's existence with a separate lookup, which dominates re-run time.
- **Evaluation speed:** ~9 s per answer. Scoring in the background, or on request, would keep the interface faster.
