from __future__ import annotations

import json
import re
import logging
from typing import Dict, Any, List

from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU
from Agents.agent_utils import (
    build_user_profile, build_item_profile, refine_review,
    sample_diverse_reviews, format_review_samples,
    parse_review_output, parse_refinement_output
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ensemble_reviews_agent")


class EnsembleReviewsAgent(SimulationAgent):
    """Agent that generates three stylistically-different reviews and uses a critic to combine them.

    The three reviews are intentionally diverse (e.g., concise/analytical/emotional) and a separate
    critic prompt evaluates and synthesizes them into a final, user-voice review. This provides
    a middle-ground between chain-of-thought and looped reasoning: multiple creative proposals
    plus a critical synthesis.
    """

    def __init__(self, llm: LLMBase, enable_refinement: bool = True, enable_profiling: bool = True):
        super().__init__(llm=llm)
        self.memory = MemoryDILU(llm=llm)
        self.enable_refinement = enable_refinement
        self.enable_profiling = enable_profiling

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

            # Conditional profiling based on toggle
            if self.enable_profiling:
                user_profile_json = build_user_profile(self.llm, user_reviews, user_profile)
                item_profile_json = build_item_profile(self.llm, item_reviews, item_info)
                logger.info("User and item profiles built")
            else:
                user_profile_json = json.dumps({"note": "Profiling disabled for ablation study"})
                item_profile_json = json.dumps({"note": "Profiling disabled for ablation study"})
                logger.info("Skipping profiling (disabled for ablation study)")

            # Generate three stylistically distinct drafts
            drafts = self._generate_three_reviews(user_reviews, item_info, item_reviews, 
                                                   user_profile_json, item_profile_json)
            logger.info(f"Generated {len(drafts)} candidate reviews")

            # Critic synthesizes them into a single recommended review
            final = self._critic_aggregate(drafts, user_reviews, item_info, item_reviews)
            logger.info(f"Critic produced stars={final.get('stars')} review length={len(final.get('review', ''))}")

            # Final lightweight refinement to match user's voice
            if self.enable_refinement:
                refined = refine_review(self.llm, final, user_profile_json, item_profile_json, 
                                      user_reviews, item_reviews)
                logger.info("Refinement complete. Returning final review.")
            else:
                refined = final
                logger.info("Skipping refinement (disabled for ablation study)")

            return refined

        except Exception as e:
            logger.exception("EnsembleReviewsAgent workflow failed: %s", e)
            return {"stars": 0.0, "review": ""}

    def _generate_three_reviews(self, user_reviews: List[Dict], item_info: Dict, item_reviews: List[Dict],
                               user_profile_json: str = None, item_profile_json: str = None) -> List[Dict[str, Any]]:
        """Ask the LLM to produce three different review types in one call."""
        try:
            logger.info("Generating three stylistically distinct reviews")
            user_examples = format_review_samples(user_reviews, max_samples=3)
            item_examples = format_review_samples(item_reviews, max_samples=3)
            item_details = json.dumps(item_info, ensure_ascii=False)[:400]

            # Extract key details from user and item data
            key_user_details = self._extract_key_details(user_reviews)
            key_item_details = self._extract_key_details(item_reviews)

            prompt = f"""Generate THREE distinct review drafts for this user-item pair.

CONTEXT:

USER PROFILE:
{user_profile_json if user_profile_json else "No user profile available"}

ITEM PROFILE:
{item_profile_json if item_profile_json else "No item profile available"}

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
            user_examples = format_review_samples(user_reviews, max_samples=3)
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
            parsed = parse_review_output(response)
            return {"stars": parsed.get("stars", 3.0), "review": parsed.get("review", "")}

        except Exception as e:
            logger.exception("Failed to critique and aggregate drafts: %s", e)
            return {"stars": 0.0, "review": ""}