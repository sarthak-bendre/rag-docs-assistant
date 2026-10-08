"""Step 4 - Generate: grounded answer with numbered citations, or an explicit 'not found'."""
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .retrieve import Hit, Retriever

NOT_FOUND = "Not found in the documents."

SYSTEM = f"""You are a technical assistant for maintenance engineers.
Answer the question using ONLY the numbered context passages below. Rules:
- Every factual statement must cite its passage(s) like [1] or [2][3].
- Quote numbers, units, limits and part numbers exactly as written in the passage.
- Do not use outside knowledge. Do not guess.
- Only state what the passages explicitly say. Never infer an answer from loosely related text
  (e.g. do not turn a passage about grounding into an answer about PPE).
- If the passages answer only part of the question, give that part and say what is not covered.
- If the passages do not contain the answer, reply exactly: "{NOT_FOUND}"
- Be concise: a direct answer first, then any essential conditions or warnings."""

HUMAN = """Context passages:
{context}

Question: {question}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]

    @property
    def not_found(self) -> bool:
        return NOT_FOUND.lower().rstrip(".") in self.text.lower()


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {h.source}, {h.page_label})\n{h.text}" for i, h in enumerate(hits, 1)
    )


def answer(question: str, retriever: Retriever, llm: BaseChatModel, k: int = 5,
           mode: str = "hybrid", rerank: bool = True) -> Answer:
    hits = retriever.search(question, k=k, mode=mode, rerank=rerank)
    if not hits:
        return Answer(question, NOT_FOUND, [])
    msg = (PROMPT | llm).invoke({"context": format_context(hits), "question": question})
    return Answer(question, msg.content.strip(), hits)
