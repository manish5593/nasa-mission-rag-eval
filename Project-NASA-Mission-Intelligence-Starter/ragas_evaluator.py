from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    # The key and OPENAI_BASE_URL (Vocareum proxy) come from the environment. The
    # proxy can take ~15s to connect, so the default timeout is too tight.
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo", timeout=120))

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small", timeout=120)
    )

    # Only reference-free metrics: the chat app has no ground-truth answer, so
    # BLEU, ROUGE and context precision "with reference" cannot be computed.
    metrics = {
        "response_relevancy": ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    }
    # Faithfulness checks the answer's claims against the retrieved context, so it needs some.
    if contexts:
        metrics["faithfulness"] = Faithfulness(llm=evaluator_llm)

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=list(contexts),
    )

    scores = {}
    errors = []
    # Scored one at a time so a single failing metric doesn't discard the others.
    for name, metric in metrics.items():
        try:
            score = float(metric.single_turn_score(sample))
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue
        # RAGAS returns NaN when it can't parse the LLM's output, and chat.py's
        # st.progress() rejects NaN (NaN is the only value not equal to itself).
        if score != score:
            errors.append(f"{name}: returned NaN")
        else:
            scores[name] = score

    if not scores:
        return {"error": "; ".join(errors) or "No metrics could be computed"}

    return scores
