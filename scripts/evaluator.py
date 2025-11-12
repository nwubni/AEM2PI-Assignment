"""
Evaluator agent for scoring RAG system responses.
It scores responses based on relevance, accuracy, completeness, and clarity.
"""

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Model for evaluation
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


class AnswerEvaluator:
    """Evaluates the quality of RAG system responses."""

    def __init__(self, model=MODEL):
        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=0.3, max_tokens=300)

    def create_evaluation_prompt(self, query: str, answer: str, context: str) -> str:
        """Create the evaluation prompt for the LLM."""
        return f"""You are an expert evaluator for a customer support RAG system. Evaluate the quality of the answer based on the following criteria:

**User Question:**
{query}

**Retrieved Context:**
{context[:1500]}...

**System Answer:**
{answer}

**Evaluation Criteria:**
1. **Relevance (0-10)**: Do the retrieved chunks contain information relevant to the question?
2. **Accuracy (0-10)**: Is the answer factually correct based on the context provided?
3. **Completeness (0-10)**: Does the answer fully address all aspects of the question?
4. **Clarity (0-10)**: Is the answer clear, concise, and well-structured?

**Instructions:**
- Score each criterion from 0-10 (10 being perfect)
- Provide a brief justification for each score
- Calculate an overall quality score (weighted average: relevance 30%, accuracy 40%, completeness 20%, clarity 10%)
- Suggest improvements if the overall score is below 7

**Output Format (JSON only):**
{{
  "relevance": {{"score": <0-10>, "justification": "<brief explanation>"}},
  "accuracy": {{"score": <0-10>, "justification": "<brief explanation>"}},
  "completeness": {{"score": <0-10>, "justification": "<brief explanation>"}},
  "clarity": {{"score": <0-10>, "justification": "<brief explanation>"}},
  "overall_score": <0-10>,
  "quality_level": "<excellent|good|acceptable|poor>",
  "improvements": ["<suggestion 1>", "<suggestion 2>"]
}}"""
    # End of evaluation prompt construction

    def evaluate(self, query: str, answer: str, context_chunks: list) -> Dict[str, Any]:
        """
        Evaluate the quality of a RAG response.

        Args:
            query: The user's question
            answer: The system's answer
            context_chunks: List of retrieved context chunks

        Returns:
            Dictionary containing evaluation scores and feedback
        """
        # Prepare context
        context = "\n\n".join(context_chunks[:3])  # Use top 3 chunks

        # Create evaluation prompt
        eval_prompt = self.create_evaluation_prompt(query, answer, context)

        # Get evaluation from LLM
        try:
            response = self.llm.invoke(eval_prompt)
            evaluation = json.loads(response.content)

            # Add quality level if not present
            if "quality_level" not in evaluation:
                score = evaluation.get("overall_score", 0)
                if score >= 8:
                    evaluation["quality_level"] = "excellent"
                elif score >= 6:
                    evaluation["quality_level"] = "good"
                elif score >= 4:
                    evaluation["quality_level"] = "acceptable"
                else:
                    evaluation["quality_level"] = "poor"

            return evaluation

        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "relevance": {"score": 5, "justification": "Unable to evaluate"},
                "accuracy": {"score": 5, "justification": "Unable to evaluate"},
                "completeness": {"score": 5, "justification": "Unable to evaluate"},
                "clarity": {"score": 5, "justification": "Unable to evaluate"},
                "overall_score": 5.0,
                "quality_level": "acceptable",
                "improvements": ["Evaluation failed - review manually"],
                "error": "Could not parse JSON",
            }

    def should_show_to_user(
        self, evaluation: Dict[str, Any], threshold: float = 6.0
    ) -> bool:
        """
        This function decides if the evaluation score is high enough to show to the user.

        Args:
            evaluation: The evaluation dictionary
            threshold: Minimum acceptable quality score (default: 6.0)

        Returns:
            True if answer should be shown, False if it should be escalated
        """
        return evaluation.get("overall_score", 0) >= threshold

    def get_quality_summary(self, evaluation: Dict[str, Any]) -> str:
        """Get a human-readable summary of the evaluation."""
        score = evaluation.get("overall_score", 0)
        level = evaluation.get("quality_level", "unknown")

        summary = f"Quality Score: {score:.1f}/10 ({level.upper()})\n"
        summary += (
            f"- Relevance: {evaluation.get('relevance', {}).get('score', 0)}/10\n"
        )
        summary += f"- Accuracy: {evaluation.get('accuracy', {}).get('score', 0)}/10\n"
        summary += (
            f"- Completeness: {evaluation.get('completeness', {}).get('score', 0)}/10\n"
        )
        summary += f"- Clarity: {evaluation.get('clarity', {}).get('score', 0)}/10\n"

        improvements = evaluation.get("improvements", [])
        if improvements and improvements[0]:
            summary += "\nSuggested Improvements:\n"
            for i, improvement in enumerate(improvements, 1):
                summary += f"  {i}. {improvement}\n"

        return summary
