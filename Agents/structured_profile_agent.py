"""Structured Profile Agent

This agent implements a structured workflow:
1. Build comprehensive user profile
2. Build item aspect profile
3. Cross-reasoning (user x item alignment)
4. Generate review in user voice
5. Self-critique and refinement
"""
from __future__ import annotations

import json
import re
import logging
from typing import Dict, Any, List

from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from Agents.agent_utils import (
    build_user_profile, build_item_profile, refine_review,
    sample_diverse_reviews, format_review_samples,
    parse_review_output, parse_refinement_output
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("structured_profile_agent")

class StructuredProfileAgent(SimulationAgent):
    """Agent with structured profiling approach"""

    def __init__(self, llm: LLMBase, enable_refinement: bool = True, enable_profiling: bool = True):
        super().__init__(llm=llm)
        self.enable_refinement = enable_refinement
        self.enable_profiling = enable_profiling

    def workflow(self) -> Dict[str, Any]:
        """Execute structured workflow: profile -> align -> generate -> refine."""
        if not self.interaction_tool:
            raise RuntimeError("interaction_tool is required")

        try:
            user_id = self.task.get("user_id")
            item_id = self.task.get("item_id")

            # Fetch all data upfront
            user_profile_raw = self.interaction_tool.get_user(user_id=user_id)
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id) or []
            item_info = self.interaction_tool.get_item(item_id=item_id)
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id) or []
            
            logger.info(f"Fetched {len(user_reviews)} user reviews, {len(item_reviews)} item reviews")

            # Conditional profiling based on toggle
            if self.enable_profiling:
                # Step 1: Build comprehensive user profile
                user_profile = build_user_profile(self.llm, user_reviews, user_profile_raw)
                logger.info(f"User profile built: {len(str(user_profile))} chars")

                # Step 2: Build item aspect profile
                item_profile = build_item_profile(self.llm, item_reviews, item_info)
                logger.info(f"Item profile built: {len(str(item_profile))} chars")

                # Step 3: Cross-reasoning (user x item alignment)
                alignment_plan = self._cross_reasoning(user_profile, item_profile, item_info)
            else:
                # Skip profiling
                user_profile = json.dumps({
                    "note": "Profiling disabled for ablation study",
                    "user_data": user_profile_raw
                })
                item_profile = json.dumps({
                    "note": "Profiling disabled for ablation study",
                    "item_data": item_info
                })
                alignment_plan = json.dumps({
                    "note": "Profiling disabled - using basic alignment",
                    "predicted_rating": 3.0
                })
                logger.info("Skipping profiling and alignment (disabled for ablation study)")
            logger.info(f"Alignment plan: {alignment_plan[:100]}...")

            # Step 4: Generate review in user voice
            draft = self._generate_draft(alignment_plan, user_profile, item_profile, user_reviews, item_reviews)
            logger.info(f"Draft: stars={draft['stars']}, review={draft['review'][:50]}...")

            # Conditional refinement based on toggle
            if self.enable_refinement:
                # Step 5: Self-critique and refinement
                final = refine_review(self.llm, draft, user_profile, item_profile, user_reviews, item_reviews)
            else:
                logger.info("Skipping refinement (disabled for ablation study)")
                final = draft
            
            return final

        except Exception as e:
            logger.exception("StructuredProfileAgent workflow failed: %s", e)
            return {"stars": 0.0, "review": ""}

    def _cross_reasoning(self, user_profile: str, item_profile: str, item_info: Dict) -> str:
        """Explicit reasoning about user x item alignment without loops."""
        
        prompt = f"""You are planning a review by analyzing user-item alignment.

USER PROFILE:
{user_profile}

ITEM PROFILE:
{item_profile}

Your task: Perform CROSS-REASONING to determine how THIS USER would review THIS ITEM.

Analyze the following aspects explicitly:

1. ASPECT PRIORITY MATCHING:
   - Which item aspects does the user care most about?
   - Which mentioned pros/cons align with user's typical focus areas?
   - What would this user notice first about this item?

2. SENTIMENT PREDICTION:
   - Based on user's rating tendency and item's qualities, what rating would they give?
   - Would they focus on positives, negatives, or balance?
   - What emotional tone would they use?

3. TOPIC SELECTION:
   - Which 2-4 specific aspects should the review mention?
   - What concrete details from item profile should be referenced?
   - What vocabulary from item reviews should be echoed?

4. STYLE MATCHING:
   - How would the user express their opinion (tone, formality, emotion)?
   - What sentence structure and length should be used?
   - What punctuation and grammar patterns to apply?
   - How comprehensive should the review be, in terms of detail, addressing pros and cons, etc.?

OUTPUT SCHEMA:
{{
  "predicted_rating": [1.0-5.0],
  "rating_justification": "why this rating",
  "key_aspects_to_mention": ["aspect1", "aspect2", "aspect3"],
  "aspect_sentiments": {{
    "aspect1": "[positive/negative/neutral] - why",
    "aspect2": "[positive/negative/neutral] - why"
  }},
  "specific_vocabulary_to_use": ["word1", "phrase2", "term3"],
  "intensity_words_to_use": ["specific intensity markers matching user's style: amazing/great/okay/terrible/somewhat/etc."],
  "concrete_details_to_include": ["specific menu item", "specific service detail", "specific use of literary device", "etc."],
  "emotional_angle": "how user would feel about this experience",
  "sentiment_strength": "[how strongly to express the sentiment: very positive/moderately positive/slightly positive/etc.]",
  "style_directives": {{
    "tone": "[casual/formal/etc.]",
    "length": "[X sentences or Y words]",
    "punctuation": "[heavy/minimal/etc.]",
    "structure": "[fragments/complete sentences/etc.]"
  }},
  "comprehensiveness": "[how wholistic, detailed, positive AND negative the reviewer is]"
}}

Provide ONLY the JSON object with your alignment analysis.
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages, temperature=0.1, max_tokens=700).strip()
        
        try:
            json.loads(response)
            return response
        except:
            logger.warning("Alignment plan not valid JSON, using raw response")
            return response

    def _generate_draft(self, alignment_plan: str, user_profile: str, item_profile: str,
                       user_reviews: List[Dict], item_reviews: List[Dict]) -> Dict[str, Any]:
        """Generate review using alignment plan and reference examples."""
        
        # Get sample reviews for style reference
        user_review_samples = format_review_samples(user_reviews, max_samples=4)
        
        prompt = f"""Generate a review following this alignment plan.

        ALIGNMENT PLAN (instructions for what to write):
        {alignment_plan}

        USER PROFILE (how to write it):
        {user_profile}

        ITEM PROFILE (context about the item):
        {item_profile}

        USER'S PAST REVIEW EXAMPLES (study for authentic voice):
        {user_review_samples}


        CRITICAL INSTRUCTIONS:

        1. FOLLOW THE ALIGNMENT PLAN:
        - Use the predicted rating
        - Mention the key aspects specified
        - Use the specific vocabulary and concrete details listed
        - Use the intensity markers specified (amazing/terrible/okay/etc.)
        - Match the sentiment strength indicated (very positive vs moderately positive)
        - Apply the emotional angle described
        - Follow all style directives (tone, length, punctuation, structure)
        - Consider user tastes and preferences (likes, dislikes, priorities in product/business/book, etc.)

        2. MIMIC THE USER'S AUTHENTIC VOICE:
        - Study the example reviews carefully
        - Copy their exact vocabulary level and sentence patterns
        - CRITICAL: Match their intensity markers (if they say "amazing" don't say "good", if they say "okay" don't say "great")
        - CRITICAL: Highlight areas of focus the user is likely to place importance upon (e.g. if they prioritize quality of a product over cost, address product quality)
        - Match the strength of their sentiment expression (extreme vs moderate language)
        - Match their grammar style (fragments vs. complete sentences)
        - Use their punctuation patterns (exclamations, ellipses, etc.)
        - Mirror their emotional expression style
        - Match their typical review length (count words/sentences in examples)
        - KEEP IT CONCISE: Most reviews are 2-5 sentences, not paragraphs (though if the reviewer writes paragraphs, emulate)
        - Mirror their comprehensiveness:
            * If they only focus on positives for positive reviews (or negatives for negative ones), only use those
            * If the review is extemely sparse or extremely detailed, emulate

        3. ENSURE SEMANTIC/TOPIC ALIGNMENT:
        - Include the concrete details from the alignment plan
        - Use vocabulary that appears in the item profile
        - Echo phrases from the similar item review when appropriate
        - Be specific about this business/book/item, not generic

        Think of it as: The ALIGNMENT PLAN tells you WHAT to say, the USER PROFILE tells you HOW to say it and WHAT to prioritize, and the EXAMPLES show you the user's authentic voice to copy.

        Output format:
        stars: [1.0|2.0|3.0|4.0|5.0]
        review: [your review text in the user's voice]
        """
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages, temperature=0.1, max_tokens=250)  # Reduced to encourage shorter, punchier reviews
        
        return parse_review_output(response)


