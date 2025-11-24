from __future__ import annotations

import json
import re
import logging
from typing import Dict, Any, List

from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ensemble_reviews_agent")


class EnsembleReviewsAgent(SimulationAgent):
    """Agent that generates three stylistically-different reviews and uses a critic to combine them.

    The three reviews are intentionally diverse (e.g., concise/analytical/emotional) and a separate
    critic prompt evaluates and synthesizes them into a final, user-voice review. This provides
    a middle-ground between chain-of-thought and looped reasoning: multiple creative proposals
    plus a critical synthesis.
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.memory = MemoryDILU(llm=llm)

    def workflow(self) -> Dict[str, Any]:
        try:
            logger.info("Starting workflow for EnsembleReviewsAgent")

            # Fetch user and item data
            user_id = self.task.get("user_id")
            item_id = self.task.get("item_id")
            logger.debug(f"Task details - user_id: {user_id}, item_id: {item_id}")

            user_profile = self.interaction_tool.get_user(user_id=user_id)
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id) or []
            item_info = self.interaction_tool.get_item(item_id=item_id)
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id) or []
            logger.info(f"Fetched {len(user_reviews)} user reviews and {len(item_reviews)} item reviews")

            # Generate three stylistically distinct drafts
            drafts = self._generate_three_reviews(user_reviews, item_info, item_reviews)
            logger.info(f"Generated {len(drafts)} candidate reviews")

            # Critic synthesizes them into a single recommended review
            final = self._critic_aggregate(drafts, user_reviews, item_info, item_reviews)
            logger.info(f"Critic produced stars={final.get('stars')} review length={len(final.get('review', ''))}")

            # Final lightweight refinement to match user's voice
            refined = self._simple_refinement(final, user_reviews, item_reviews)
            logger.info("Refinement complete. Returning final review.")

            return refined

        except Exception as e:
            logger.exception("EnsembleReviewsAgent workflow failed: %s", e)
            return {"stars": 0.0, "review": ""}

    def _generate_three_reviews(self, user_reviews: List[Dict], item_info: Dict, item_reviews: List[Dict]) -> List[Dict[str, Any]]:
        """Ask the LLM to produce three different review types in one call."""
        try:
            logger.info("Generating three stylistically distinct reviews")
            user_examples = self._format_review_samples(user_reviews, max_samples=3)
            item_examples = self._format_review_samples(item_reviews, max_samples=3)
            item_details = json.dumps(item_info, ensure_ascii=False)[:400]

            # Extract key details from user and item data
            key_user_details = self._extract_key_details(user_reviews)
            key_item_details = self._extract_key_details(item_reviews)

            prompt = f"""Generate THREE distinct review drafts for this user-item pair.

CONTEXT:
User examples:
{user_examples}

Item examples:
{item_examples}

Item details:
{item_details}

Key user details:
{key_user_details}

Key item details:
{key_item_details}

Produce exactly three drafts, labeled with a type name (Concise, Analytical, Emotional).
For each draft output the following lines:
Type: [Concise|Analytical|Emotional]
stars: [1.0|2.0|3.0|4.0|5.0]
review: [the review text]

Instructions:
- Concise: 1-2 sentences, direct recommendation, clear rating justification.
- Analytical: 4-6 sentences, discuss concrete aspects and reasoning behind rating.
- Emotional: expressive personal narrative, first-person, emotionally colored language.

Make the three drafts different in length, tone, and content while staying realistic for the user's voice.
Ensure the reviews explicitly reference key user and item details where relevant.
"""

            messages = [{"role": "user", "content": prompt}]
            response = self.llm(messages=messages, temperature=0.5, max_tokens=800)

            logger.debug("LLM response received for three reviews")

            # Parse the three sections
            sections = re.split(r"(?m)^Type:\s*", response)
            candidates = []
            for sec in sections:
                if sec.strip():
                    match = re.search(r"stars:\s*([0-9]+(?:\.[0-9]+)?)\s*review:\s*(.+)$", sec, re.DOTALL)
                    if match:
                        candidates.append({
                            "type": sec.splitlines()[0].strip(),
                            "stars": float(match.group(1)),
                            "review": match.group(2).strip()
                        })

            # If LLM returned fewer than 3, pad by regenerating simple variants
            if len(candidates) < 3:
                logger.warning("Fewer than 3 drafts generated, padding with simple variants")
                while len(candidates) < 3:
                    candidates.append({
                        "type": "Fallback",
                        "stars": 3.0,
                        "review": "This is a fallback review due to insufficient drafts."
                    })

            logger.info(f"Generated {len(candidates)} review drafts")
            return candidates

        except Exception as e:
            logger.exception("Failed to generate three reviews: %s", e)
            return []

    def _extract_key_details(self, reviews: List[Dict]) -> str:
        """Extract key details from reviews to provide more context."""
        if not reviews:
            return "No significant details available."

        key_details = []
        for review in reviews[:3]:
            rating = review.get("stars", "N/A")
            text = review.get("review", "").strip()
            key_details.append(f"Rating: {rating}, Excerpt: {text[:100]}...")

        return "\n".join(key_details)

    def _critic_aggregate(self, drafts: List[Dict[str, Any]], user_reviews: List[Dict], item_info: Dict, item_reviews: List[Dict]) -> Dict[str, Any]:
        """Given multiple drafts, ask the LLM to critique and synthesize a final review."""
        try:
            logger.info("Starting critique and aggregation of drafts")
            user_examples = self._format_review_samples(user_reviews, max_samples=3)
            item_details = json.dumps(item_info, ensure_ascii=False)[:400]

            drafts_text = "\n\n".join([f"Draft {i+1} (Type: {d.get('type')}):\nstars: {d.get('stars')}\nreview: {d.get('review')}" for i,d in enumerate(drafts)])

            prompt = f"""You are a critic that reviews multiple candidate reviews and synthesizes the best final review.

Item details:
{item_details}

User examples:
{user_examples}

Candidate drafts:
{drafts_text}

Task:
1) For each draft, give a one-line critique (what is good, what to avoid).
2) Choose the best elements from the drafts and produce a final REVIEW in the user's voice.
3) Output EXACTLY the following format:
Critiques:
- Draft 1: [one-line critique]
- Draft 2: [one-line critique]
- Draft 3: [one-line critique]
Final:
stars: [1.0|2.0|3.0|4.0|5.0]
review: [final synthesized review text]

Make the final review concise but specific, and prefer concrete details from the candidate drafts and item examples. Match the user's voice from the examples above.
"""

            messages = [{"role": "user", "content": prompt}]
            response = self.llm(messages=messages, temperature=0.2, max_tokens=500)

            logger.debug("LLM response received for critique and aggregation")

            # Try to extract Final block
            final_match = re.search(r"Final:\s*\nstars\s*[:=]\s*\[?\s*([0-9]+(?:\.[0-9]+)?)\s*\]?\s*\nreview\s*[:=]\s*(.+)$", response, re.IGNORECASE | re.DOTALL)
            if final_match:
                stars = float(final_match.group(1))
                review = final_match.group(2).strip()
                logger.info("Critique and aggregation successful")
                return {"stars": max(1.0, min(5.0, stars)), "review": review}

            # Fallback: parse with existing parser
            logger.warning("Failed to parse final review block, using fallback parser")
            parsed = self._parse_review_output(response)
            return {"stars": parsed.get("stars", 3.0), "review": parsed.get("review", "")}

        except Exception as e:
            logger.exception("Failed to critique and aggregate drafts: %s", e)
            return {"stars": 0.0, "review": ""}

    def _simple_refinement(self, draft: Dict[str, Any], user_reviews: List[Dict], item_reviews: List[Dict]) -> Dict[str, Any]:
        """Lightweight refinement to adjust style to user's voice.

        This step verifies style/vocabulary alignment with a quick LLM pass.
        """
        review_text = draft.get("review", "")
        stars = draft.get("stars", 3.0)

        user_examples = self._format_review_samples(user_reviews, max_samples=3)
        item_examples = self._format_review_samples(item_reviews, max_samples=2)

        prompt = f"""Refine the following review so it matches the user's voice in the examples.

User examples:
{user_examples}

Item examples:
{item_examples}

Generated review:
Rating: {stars}
Review: {review_text}

If the review already matches the user's voice closely, return it unchanged. Otherwise, produce a revised review that keeps the same rating but better matches tone, vocabulary, and length of the user examples.

Output format:
Refined Review: [the review text]
"""

        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages, temperature=0.1, max_tokens=400)

        refined_match = re.search(r"Refined Review:\s*(.+)$", response, re.IGNORECASE | re.DOTALL)
        if refined_match:
            refined = refined_match.group(1).strip().strip('"\'')
            draft["review"] = refined
            return draft

        # fallback
        return draft

    def _parse_review_output(self, text: str) -> Dict[str, Any]:
        stars = 3.0
        review = ""
        try:
            stars_match = re.search(r"stars\s*[:=]\s*\[?\s*([0-9]+(?:\.[0-9]+)?)\s*\]?", text, re.IGNORECASE)
            if stars_match:
                stars = float(stars_match.group(1))
                stars = max(1.0, min(5.0, stars))

            review_match = re.search(r"review\s*[:=]\s*(.+?)(?=\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
            if review_match:
                review = review_match.group(1).strip()
            else:
                # fallback: take last paragraph
                review = text.strip().split('\n')[-1].strip()
        except Exception:
            review = text.strip()

        return {"stars": stars, "review": review}

    def _sample_diverse_reviews(self, reviews: List[Dict], max_samples: int = 5) -> List[Dict]:
        if len(reviews) <= max_samples:
            return reviews
        by_rating = {}
        for r in reviews:
            rating = r.get("stars", 3.0)
            by_rating.setdefault(rating, []).append(r)
        sampled = []
        for rating in sorted(by_rating.keys()):
            sampled.extend(by_rating[rating][:max(1, max_samples // len(by_rating))])
            if len(sampled) >= max_samples:
                break
        return sampled[:max_samples]

    def _format_review_samples(self, reviews: List[Dict], max_samples: int = 3) -> str:
        if not reviews:
            return "No past reviews available."
        sampled = self._sample_diverse_reviews(reviews, max_samples=max_samples)
        formatted = []
        for i, review in enumerate(sampled, 1):
            stars = review.get("stars", "N/A")
            text = review.get("text", "")
            formatted.append(f"Example {i}:\nRating: {stars} stars\nReview: {text}")
        return "\n\n".join(formatted)