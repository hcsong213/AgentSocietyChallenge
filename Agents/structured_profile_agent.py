"""Structured Profile Agent - Non-looping approach with explicit profiling stages.

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
import os
from typing import Dict, Any, List

from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import OllamaLLM, LLMBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU

if __name__ == "__main__":
    from Util import format_llm_logs
else:
    try:
        from ..Util import format_llm_logs
    except ImportError:
        from Util import format_llm_logs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("structured_profile_agent")


class StructuredProfileAgent(SimulationAgent):
    """Agent with structured profiling approach - no reasoning loops."""

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)
        self.memory = MemoryDILU(llm=llm)

    def workflow(self) -> Dict[str, Any]:
        """Execute structured workflow: profile → align → generate → refine."""
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

            # Step 1: Build comprehensive user profile
            user_profile = self._build_user_profile(user_reviews, user_profile_raw)
            logger.info(f"User profile built: {len(str(user_profile))} chars")

            # Step 2: Build item aspect profile
            item_profile = self._build_item_profile(item_reviews, item_info)
            logger.info(f"Item profile built: {len(str(item_profile))} chars")

            # Step 3: Cross-reasoning (user x item alignment)
            alignment_plan = self._cross_reasoning(user_profile, item_profile, item_info)
            logger.info(f"Alignment plan: {alignment_plan[:100]}...")

            # Step 4: Generate review in user voice
            draft = self._generate_draft(alignment_plan, user_profile, item_profile, user_reviews, item_reviews)
            logger.info(f"Draft: stars={draft['stars']}, review={draft['review'][:50]}...")

            # Step 5: Self-critique and refinement
            final = self._simple_refinement(draft, user_profile, item_profile, user_reviews, item_reviews)
            
            return final

        except Exception as e:
            logger.exception("StructuredProfileAgent workflow failed: %s", e)
            return {"stars": 0.0, "review": ""}

    def _build_user_profile(self, user_reviews: List[Dict], user_profile_raw: Dict) -> str:
        """Build comprehensive structured user profile with liberal review sampling."""
        if not user_reviews:
            return json.dumps({
                "writing_tone": "neutral",
                "sentiment_tendency": "neutral",
                "vocabulary_patterns": [],
                "focus_aspects": [],
                "typical_length": "medium",
                "rating_patterns": {"average": 3.0}
            })
        
        # Sample generously - up to 15 reviews for comprehensive profiling
        sampled = self._sample_diverse_reviews(user_reviews, max_samples=15)
        reviews_text = "\n\n".join([
            f"Rating: {r.get('stars', 'N/A')} stars\nReview: {r.get('text', '')}"
            for r in sampled
        ])
        
        prompt = f"""Analyze this user's review patterns and create a STRUCTURED PROFILE.

User's review samples ({len(sampled)} reviews):
{reviews_text}

Additional user info:
{json.dumps(user_profile_raw, ensure_ascii=False)[:300]}

Create a comprehensive structured profile with the following schema:

OUTPUT SCHEMA (JSON format):
{{
  "user_name": "[user's name if available in profile, otherwise 'Unknown']",
  "writing_tone": "[casual/formal/sarcastic/enthusiastic/balanced/etc.]",
  "writing_tone_details": "[2-3 sentence description of their voice and style]",
  "sentiment_tendency": "[positive/neutral/negative/mixed]",
  "sentiment_details": "[how generous or harsh they are with ratings]",
  "vocabulary_patterns": ["list", "of", "characteristic", "words", "and", "phrases", "they", "use"],
  "grammar_style": "[complete sentences/fragments/run-ons/mix]",
  "punctuation_style": "[heavy exclamations/minimal punctuation/ellipses/etc.]",
  "focus_aspects": {{
    "restaurants": ["service", "food quality", "ambiance", "etc."],
    "general": ["what they typically comment on"]
  }},
  "typical_length": "[very short (1 sentence)/short (2-3 sentences)/medium (4-6)/long (7+)]",
  "typical_length_words": [average word count],
  "rating_patterns": {{
    "average": [average star rating],
    "tendency": "[harsh/generous/moderate]",
    "distribution": "[mostly 5s and 1s / balanced / etc.]"
  }},
  "personality_traits": ["descriptors of their personality in reviews"],
  "emotional_expression": "[reserved/enthusiastic/dramatic/matter-of-fact]"
}}

Provide ONLY the JSON object, no additional text.
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages, temperature=0.0, max_tokens=800).strip()
        
        # Try to parse as JSON, if it fails return the raw response
        try:
            json.loads(response)
            return response
        except:
            logger.warning("User profile not valid JSON, using raw response")
            return response

    def _build_item_profile(self, item_reviews: List[Dict], item_info: Dict) -> str:
        """Build structured item aspect profile with liberal review sampling."""
        if not item_reviews:
            return json.dumps({
                "common_themes": [],
                "sentiment_distribution": {},
                "pros": [],
                "cons": [],
                "polarizing_aspects": []
            })
        
        # Sample generously - up to 15 reviews for comprehensive profiling
        sampled = self._sample_diverse_reviews(item_reviews, max_samples=15)
        reviews_text = "\n\n".join([
            f"Rating: {r.get('stars', 'N/A')} stars\nReview: {r.get('text', '')}"
            for r in sampled
        ])
        
        prompt = f"""Analyze existing reviews for this item/business and create a STRUCTURED ASPECT PROFILE.

NOTE: The 'item' can be either a physical product OR a service/business (restaurant, hotel, etc.).

Item details:
{json.dumps(item_info, ensure_ascii=False)[:400]}

Item reviews ({len(sampled)} reviews):
{reviews_text}

Create a comprehensive structured profile with the following schema:

OUTPUT SCHEMA (JSON format):
{{
  "item_name": "[name of the item/business from the item details]",
  "item_type": "[product/restaurant/hotel/book/service/etc.]",
  "common_themes": ["list", "of", "aspects", "people", "discuss"],
  "theme_details": {{
    "theme1": "what people say about this theme",
    "theme2": "what people say about this theme"
  }},
  "sentiment_distribution": {{
    "overall_sentiment": "[positive/negative/mixed]",
    "average_rating": [average],
    "rating_breakdown": "description of rating distribution"
  }},
  "pros": ["specific", "positive", "aspects", "with", "exact", "vocabulary"],
  "cons": ["specific", "negative", "aspects", "with", "exact", "vocabulary"],
  "polarizing_aspects": ["things some love and others hate"],
  "common_vocabulary": ["words", "and", "phrases", "that", "appear", "frequently"],
  "emotional_tone": "[enthusiastic/disappointed/balanced/etc.]",
  "specific_mentions": ["concrete items/features people name"]
}}

Provide ONLY the JSON object, no additional text.
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages, temperature=0.0, max_tokens=800).strip()
        
        try:
            json.loads(response)
            return response
        except:
            logger.warning("Item profile not valid JSON, using raw response")
            return response

    def _cross_reasoning(self, user_profile: str, item_profile: str, item_info: Dict) -> str:
        """Explicit reasoning about user x item alignment without loops."""
        
        prompt = f"""You are planning a review by analyzing user-item alignment.

USER PROFILE:
{user_profile}

ITEM PROFILE:
{item_profile}

ITEM DETAILS:
{json.dumps(item_info, ensure_ascii=False)[:400]}

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
  "concrete_details_to_include": ["specific menu item", "specific service detail", "etc."],
  "emotional_angle": "how user would feel about this experience",
  "style_directives": {{
    "tone": "[casual/formal/etc.]",
    "length": "[X sentences or Y words]",
    "punctuation": "[heavy/minimal/etc.]",
    "structure": "[fragments/complete sentences/etc.]"
  }}
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
        user_review_samples = self._format_review_samples(user_reviews, max_samples=4)
        
        # Use memory-based similarity search for most relevant item review
        for review in item_reviews:
            review_text = review.get('text', '')
            if review_text:
                self.memory(f'review: {review_text}')
        
        similar_item_review = ""
        if user_reviews and self.memory.scenario_memory._collection.count() > 0:
            similar_item_review = self.memory(f'{user_reviews[0]["text"]}')
            logger.info(f"Found similar item review (length: {len(similar_item_review)} chars)")
        
        prompt = f"""Generate a review following this alignment plan.

ALIGNMENT PLAN (instructions for what to write):
{alignment_plan}

USER PROFILE (how to write it):
{user_profile}

ITEM PROFILE (context about the item):
{item_profile}

USER'S PAST REVIEW EXAMPLES (study for authentic voice):
{user_review_samples}

SIMILAR ITEM REVIEW (semantic/emotional reference):
{similar_item_review if similar_item_review else "No similar review found."}

CRITICAL INSTRUCTIONS:

1. FOLLOW THE ALIGNMENT PLAN:
   - Use the predicted rating
   - Mention the key aspects specified
   - Use the specific vocabulary and concrete details listed
   - Apply the emotional angle described
   - Follow all style directives (tone, length, punctuation, structure)

2. MIMIC THE USER'S AUTHENTIC VOICE:
   - Study the example reviews carefully
   - Copy their exact vocabulary level and sentence patterns
   - Match their grammar style (fragments vs. complete sentences)
   - Use their punctuation patterns (exclamations, ellipses, etc.)
   - Mirror their emotional expression style
   - Match their typical review length

3. ENSURE SEMANTIC/TOPIC ALIGNMENT:
   - Include the concrete details from the alignment plan
   - Use vocabulary that appears in the item profile
   - Echo phrases from the similar item review when appropriate
   - Be specific about this business, not generic

Think of it as: The ALIGNMENT PLAN tells you WHAT to say, the USER PROFILE tells you HOW to say it, and the EXAMPLES show you the user's authentic voice to copy.

Output format:
stars: [1.0|2.0|3.0|4.0|5.0]
review: [your review text in the user's voice]
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages, temperature=0.1, max_tokens=400)
        
        return self._parse_review_output(response)

    def _simple_refinement(self, draft: Dict[str, Any], user_profile: str, item_profile: str,
                          user_reviews: List[Dict], item_reviews: List[Dict]) -> Dict[str, Any]:
        """Self-critique and refinement to ensure alignment."""
        
        review_text = draft.get("review", "")
        stars = draft.get("stars", 3.0)
        
        # Sample reviews for comparison
        user_review_samples = self._format_review_samples(user_reviews, max_samples=3)
        item_review_samples = self._format_review_samples(item_reviews, max_samples=3)
        
        # Get similar item review from memory
        similar_item_review = ""
        if user_reviews and self.memory.scenario_memory._collection.count() > 0:
            similar_item_review = self.memory(f'{user_reviews[0]["text"]}')
        
        prompt = f"""Perform SELF-CRITIQUE on this generated review and refine if needed.

GENERATED REVIEW:
Rating: {stars} stars
Review: "{review_text}"

USER PROFILE:
{user_profile}

ITEM PROFILE:
{item_profile}

USER'S PAST REVIEW EXAMPLES:
{user_review_samples}

ITEM'S PAST REVIEW EXAMPLES:
{item_review_samples}

SIMILAR ITEM REVIEW:
{similar_item_review if similar_item_review else "No similar review available."}

CRITIQUE CHECKLIST:

1. STYLE CONSISTENCY:
   ✓ Does it match user's vocabulary level and sentence structure?
   ✓ Is formality/casualness consistent with user's style?
   ✓ Are punctuation patterns similar?
   ✓ Does length match user's typical length?
   ✓ Is grammar style consistent?

2. SENTIMENT/EMOTION ALIGNMENT:
   ✓ Does emotional tone match the rating intensity?
   ✓ Is sentiment consistent throughout?
   ✓ Does it convey appropriate emotion for this rating?

3. TOPIC/SEMANTIC ALIGNMENT:
   ✓ Does it mention specific, concrete features of this business?
   ✓ Does it use relevant vocabulary from item reviews?
   ✓ Are topics semantically similar to what others mention?
   ✓ Does it avoid generic phrases?

4. COHERENCE:
   ✓ Is it the right length?
   ✓ Does it provide meaningful information?
   ✓ Is it coherent and well-structured?

REFINEMENT STRATEGY:
- If style mismatches: adjust vocabulary, structure, formality
- If sentiment is off: strengthen or soften emotional expression
- If topics are generic: add specific business details
- If length is wrong: expand with specifics or condense
- Maintain the rating unless clearly misaligned

OUTPUT FORMAT:
Critique: [What needs refinement, if anything - be specific]
Refined Review: [The improved review, or original if no changes needed]
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages, temperature=0.1, max_tokens=500)
        
        refined_review = self._parse_refinement_output(response, original_review=review_text)
        logger.info(f"Refinement: {refined_review[:100]}...")
        draft["review"] = refined_review
        
        return draft

    def _parse_refinement_output(self, text: str, original_review: str) -> str:
        """Parse refinement output to extract the refined review."""
        try:
            refined_match = re.search(r"Refined Review:\s*(.+?)(?=\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
            if refined_match:
                refined = refined_match.group(1).strip().strip('"\'')
                return refined if refined else original_review
            
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.lower().startswith('refined review:'):
                    review_parts = [line.split(':', 1)[1].strip() if ':' in line else '']
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip() and not lines[j].lower().startswith('critique'):
                            review_parts.append(lines[j].strip())
                        elif lines[j].strip():
                            break
                    refined = ' '.join(review_parts).strip('"\'')
                    return refined if refined else original_review
            
            logger.warning("Could not parse refinement output, using original review")
            return original_review
            
        except Exception as e:
            logger.warning(f"Refinement parse error: {e}, using original review")
            return original_review

    def _parse_review_output(self, text: str) -> Dict[str, Any]:
        """Parse LLM output into stars and review."""
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
                for line in text.split("\n"):
                    if line.lower().startswith("review:"):
                        review = line.split(":", 1)[1].strip()
                        break
            
            if not review:
                review = text.strip()
        
        except Exception as e:
            logger.warning(f"Parse error: {e}")
            review = text if text else ""
        
        return {"stars": stars, "review": review}

    def _sample_diverse_reviews(self, reviews: List[Dict], max_samples: int = 5) -> List[Dict]:
        """Sample reviews to get diverse ratings."""
        if len(reviews) <= max_samples:
            return reviews
        
        by_rating = {}
        for r in reviews:
            rating = r.get("stars", 3.0)
            if rating not in by_rating:
                by_rating[rating] = []
            by_rating[rating].append(r)
        
        sampled = []
        for rating in sorted(by_rating.keys()):
            sampled.extend(by_rating[rating][:max(1, max_samples // len(by_rating))])
            if len(sampled) >= max_samples:
                break
        
        return sampled[:max_samples]

    def _format_review_samples(self, reviews: List[Dict], max_samples: int = 3) -> str:
        """Format sample reviews for style reference."""
        if not reviews:
            return "No past reviews available."
        
        sampled = self._sample_diverse_reviews(reviews, max_samples=max_samples)
        formatted = []
        for i, review in enumerate(sampled, 1):
            stars = review.get("stars", "N/A")
            text = review.get("text", "")
            formatted.append(f"Example {i}:\nRating: {stars} stars\nReview: {text}")
        
        return "\n\n".join(formatted)


if __name__ == "__main__":
    from Util import format_llm_logs
    import os

    sim = Simulator(data_dir="dataset", device="gpu", cache=True)
    here = os.path.dirname(__file__)

    task_set = "yelp"  # "goodreads" or "yelp"
    task_dir = os.path.join("example", "track1", task_set, "tasks")
    groundtruth_dir = os.path.join("example", "track1", task_set, "groundtruth")
    sim.set_task_and_groundtruth(task_dir=task_dir, groundtruth_dir=groundtruth_dir)

    llm = OllamaLLM(model="mistral")
    sim.set_agent(StructuredProfileAgent)
    sim.set_llm(llm)

    # Set to True for single task debugging
    if False:
        print("Running structured profile agent for task 0...")
        res = sim.run_single_task(task_index=0, wrap_llm_with_logger=True)
        
        output_data = {
            "output": res.get("output"),
            "llm_calls": res.get("llm_calls", [])
        }

        print(json.dumps({"output_present": bool(output_data["output"]), 
                         "llm_call_count": len(output_data["llm_calls"])}, indent=2))

        log_json_path = './Outputs/structured_profile_output.json'
        with open(log_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed logs to {log_json_path}")

        formatted_txt_path = './Outputs/structured_profile_output.txt'
        try:
            format_llm_logs(log_json_path, formatted_txt_path)
            print(f"Saved formatted LLM logs to {formatted_txt_path}")
        except Exception as e:
            logger.exception("Failed to format LLM logs: %s", e)
    else:
        outputs = sim.run_simulation(number_of_tasks=400)
        evaluation_results = sim.evaluate()
        with open(f'./Outputs/structured_profile_evaluation_{task_set}.json', 'w') as f:
            json.dump(evaluation_results, f, indent=4)

        print(f"The evaluation_results is: {evaluation_results}")
