"""LLM-based query processing: rewriting and compound-question decomposition."""
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class QueryProcessor:
    """
    Enhances raw user queries before retrieval using an LLM.

    Two strategies are provided:

    rewrite(question)
        Rephrases a vague or conversational question into a concise,
        retrieval-friendly form.  The rewritten query produces embeddings
        that align more tightly with the document vector space.

    decompose(question)
        Splits compound or comparative questions into independent
        sub-queries so each can be retrieved separately, then merged.

    Both operations are optional; the RAGAssistant falls back to the raw
    query when query processing is disabled.
    """

    _REWRITE_PROMPT = ChatPromptTemplate.from_template(
        """You are an expert at reformulating questions for semantic document retrieval.
Rewrite the question below to be more precise and retrieval-friendly.
- Keep it as a single, self-contained question.
- Do not add facts not implied by the original.
- Return only the rewritten question with no preamble.

Original question: {question}
Rewritten question:"""
    )

    _DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
        """You are an expert at breaking down complex questions.
If the question below contains multiple distinct sub-questions or asks for a comparison,
split it into focused individual questions — one per line.
If it is already a single focused question, return it unchanged.
Return only the questions, one per line, with no numbering or bullets.

Question: {question}
Sub-questions:"""
    )

    def __init__(self, llm):
        """
        Args:
            llm: A LangChain chat model (e.g. ChatOpenAI) used for rewriting/decomposition.
        """
        self._llm = llm
        self._rewrite_chain = self._REWRITE_PROMPT | llm | StrOutputParser()
        self._decompose_chain = self._DECOMPOSE_PROMPT | llm | StrOutputParser()

    def rewrite(self, question: str) -> str:
        """
        Rewrite a query to improve retrieval precision.

        Example:
            "Tell me about AI" → "What are the core concepts and applications of Artificial Intelligence?"

        Args:
            question: The original user question.

        Returns:
            A retrieval-optimised version of the question.
        """
        rewritten = self._rewrite_chain.invoke({"question": question}).strip()
        return rewritten if rewritten else question

    def decompose(self, question: str) -> List[str]:
        """
        Decompose a compound question into focused sub-queries.

        Example:
            "Compare deep learning and quantum computing"
            → ["What is deep learning?", "What is quantum computing?"]

        Args:
            question: The original user question (may be compound).

        Returns:
            List of sub-queries. Returns [question] unchanged if it is already focused.
        """
        raw = self._decompose_chain.invoke({"question": question}).strip()
        parts = [line.strip() for line in raw.splitlines() if line.strip()]
        return parts if parts else [question]
