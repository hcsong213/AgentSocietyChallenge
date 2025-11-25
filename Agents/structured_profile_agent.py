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
from typing import Dict, Any, List

from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase

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
                user_profile = self._build_user_profile(user_reviews, user_profile_raw)
                logger.info(f"User profile built: {len(str(user_profile))} chars")

                # Step 2: Build item aspect profile
                item_profile = self._build_item_profile(item_reviews, item_info)
                logger.info(f"Item profile built: {len(str(item_profile))} chars")

                # Step 3: Cross-reasoning (user x item alignment)
                alignment_plan = self._cross_reasoning(user_profile, item_profile, item_info)
            else:
                # Skip profiling - use minimal placeholders
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
                final = self._simple_refinement(draft, user_profile, item_profile, user_reviews, item_reviews)
            else:
                logger.info("Skipping refinement (disabled for ablation study)")
                final = draft
            
            return final

        except Exception as e:
            logger.exception("StructuredProfileAgent workflow failed: %s", e)
            return {"stars": 0.0, "review": ""}

    def _build_user_profile(self, user_reviews: List[Dict], user_profile_raw: Dict) -> str:
        """Build comprehensive structured user profile with review sampling."""

        # Detect review source (amazon / goodreads / yelp). Fall back to generic.
        source = None
        try:
            source = user_profile_raw.get("source", None)
            if not source and user_reviews:
                source = user_reviews[0].get("source", None)
        except:
            source = None
        source = (source or "").lower()

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

        if source == "amazon":
            prompt = f"""
Analyze this Amazon user's review patterns and create a STRUCTURED PROFILE.

User's review samples ({len(sampled)} reviews):
{reviews_text}

Additional user info:
{json.dumps(user_profile_raw, ensure_ascii=False)}

Create a comprehensive structured profile with the following schema:

OUTPUT SCHEMA (JSON format):
{{
  "user_name": "[user's name if available in profile, otherwise 'Unknown']",
  "writing_tone": "[casual/formal/sarcastic/enthusiastic/balanced/etc.]",
  "writing_tone_details": "[2-3 sentence description of their voice and style]",
  "sentiment_tendency": "[positive/neutral/negative/mixed]",
  "sentiment_details": "[how generous or harsh they are with ratings]",
  "vocabulary_patterns": ["list", "of", "characteristic", "words", "and", "phrases", "they", "use"],
  "intensity_markers": ["words that amplify emotion: amazing/terrible/extremely/very/somewhat/okay/etc."],
  "sentiment_intensity": "[how strongly they express opinions: extreme/moderate/mild]",
  "grammar_style": "[complete sentences/fragments/run-ons/mix]",
  "punctuation_style": "[heavy exclamations/minimal punctuation/ellipses/etc.]",
  "comprehensiveness": "[how thorough/sparse they are in their reviews, whether they address both positives and negatives for positive/negative reviews, etc.]",

  "product_focus": ["durability", "accuracy", "functionality", "value", "etc."],
  "purchase_patterns": "[categories/themes/uses shared between reviewed products, if any]"
  "value_sensitivity": "[frugal/high spender/etc.]",

  "typical_length": "[very short (1 sentence)/short (2-3 sentences)/medium (4-6)/long (7+)]",
  "typical_length_words": [average word count],
  "rating_patterns": {{
    "average": [average star rating],
    "tendency": "[harsh/generous/moderate]",
    "distribution": "[mostly 5s and 1s / balanced / etc.]"
  }},
  "personality_traits": ["descriptors of their personality inferred from reviews"],
  "emotional_expression": "[reserved/enthusiastic/dramatic/matter-of-fact]"
}}

Provide ONLY the JSON object, no additional text.
"""
            
        elif source == "goodreads":
            prompt = f"""
Analyze this Goodreads user's review patterns and create a STRUCTURED PROFILE.

User's review samples ({len(sampled)} reviews):
{reviews_text}

Additional user info:
{json.dumps(user_profile_raw, ensure_ascii=False)}

Create a comprehensive structured profile with the following schema:

OUTPUT SCHEMA (JSON format):
{{
  "user_name": "[user's name if available in profile, otherwise 'Unknown']",
  "writing_tone": "[casual/formal/sarcastic/enthusiastic/balanced/etc.]",
  "writing_tone_details": "[2-3 sentence description of their voice and style]",
  "sentiment_tendency": "[positive/neutral/negative/mixed]",
  "sentiment_details": "[how generous or harsh they are with ratings]",
  "vocabulary_patterns": ["list", "of", "characteristic", "words", "and", "phrases", "they", "use"],
  "intensity_markers": ["words that amplify emotion: amazing/terrible/extremely/very/somewhat/okay/etc."],
  "sentiment_intensity": "[how strongly they express opinions: extreme/moderate/mild]",
  "grammar_style": "[complete sentences/fragments/run-ons/mix]",
  "punctuation_style": "[heavy exclamations/minimal punctuation/ellipses/etc.]",
  "comprehensiveness": "[how thorough/sparse they are in their reviews, whether they address both positives and negatives for positive/negative reviews, etc.]",

  "literary_taste": ["genres", "favorite authors", "themes", "etc."],
  "interpretive_focus": ["plot", "characters", "world-building", "prose", "etc."],

  "typical_length": "[very short (1 sentence)/short (2-3 sentences)/medium (4-6)/long (7+)]",
  "typical_length_words": [average word count],
  "rating_patterns": {{
    "average": [average star rating],
    "tendency": "[harsh/generous/moderate]",
    "distribution": "[mostly 5s and 1s / balanced / etc.]"
  }},
  "personality_traits": ["descriptors of their personality inferred from reviews"],
  "emotional_expression": "[reserved/enthusiastic/dramatic/matter-of-fact]"
}}

Provide ONLY the JSON object, no additional text.
"""
            
        elif source == "yelp":

            yelp_info = {
                "user_id": user_profile_raw.get("user_id"),
                "name": user_profile_raw.get("name", "Unknown"),
                "review_count": user_profile_raw.get("review_count"),
                "useful": user_profile_raw.get("useful"),
                "funny": user_profile_raw.get("funny"),
                "cool": user_profile_raw.get("cool")
            }

            prompt = f"""
Analyze this Yelp user's review patterns and create a STRUCTURED PROFILE.

User's review samples ({len(sampled)} reviews):
{reviews_text}

Additional user info:
{json.dumps(user_profile_raw, ensure_ascii=False)}

Also consider the user's metadata as influencing factors:
- review_count (weighting factor for average behavior): {yelp_info['review_count']}
- useful votes (tendency to write helpful reviews): {yelp_info['useful']}
- funny votes (tendency toward humor in reviews): {yelp_info['funny']}
- cool votes (coolness/relatability): {yelp_info['cool']}

Create a comprehensive structured profile with the following schema:

OUTPUT SCHEMA (JSON format):
{{
  "user_name": "[user's name if available in profile, otherwise 'Unknown']",
  "writing_tone": "[casual/formal/sarcastic/enthusiastic/balanced/etc.]",
  "writing_tone_details": "[2-3 sentence description of their voice and style]",
  "sentiment_tendency": "[positive/neutral/negative/mixed]",
  "sentiment_details": "[how generous or harsh they are with ratings]",
  "vocabulary_patterns": ["list", "of", "characteristic", "words", "and", "phrases", "they", "use"],
  "intensity_markers": ["words that amplify emotion: amazing/terrible/extremely/very/somewhat/okay/etc."],
  "sentiment_intensity": "[how strongly they express opinions: extreme/moderate/mild]",
  "grammar_style": "[complete sentences/fragments/run-ons/mix]",
  "punctuation_style": "[heavy exclamations/minimal punctuation/ellipses/etc.]",
  "comprehensiveness": "[how thorough/sparse they are in their reviews, whether they address both positives and negatives for positive/negative reviews, etc.]",

  "business_focus": ["service", "ambiance", "facility quality/availability", "food quality", "price sensitivity", "parking availability", "etc."],
  "taste_patterns": "[cuisines/establishment styles/themes shared between reviewed businesses, if any]"

  "typical_length": "[very short (1 sentence)/short (2-3 sentences)/medium (4-6)/long (7+)]",
  "typical_length_words": [average word count],
  "rating_patterns": {{
    "average": [average star rating],
    "tendency": "[harsh/generous/moderate]",
    "distribution": "[mostly 5s and 1s / balanced / etc.]"
  }},
  "personality_traits": ["descriptors of their personality inferred from reviews"],
  "emotional_expression": "[reserved/enthusiastic/dramatic/matter-of-fact]"
}}

Provide ONLY the JSON object, no additional text.
"""
            
        else:
            prompt = f"""
Analyze this user's review patterns and create a GENERIC STRUCTURED PROFILE.

User's review samples ({len(sampled)} reviews):
{reviews_text}

Additional user info:
{json.dumps(user_profile_raw, ensure_ascii=False)}

Create a comprehensive structured profile with the following schema:

OUTPUT SCHEMA (JSON format):
{{
  "user_name": "[user's name if available in profile, otherwise 'Unknown']",
  "writing_tone": "[casual/formal/sarcastic/enthusiastic/balanced/etc.]",
  "writing_tone_details": "[2-3 sentence description of their voice and style]",
  "sentiment_tendency": "[positive/neutral/negative/mixed]",
  "sentiment_details": "[how generous or harsh they are with ratings]",
  "vocabulary_patterns": ["list", "of", "characteristic", "words", "and", "phrases", "they", "use"],
  "intensity_markers": ["words that amplify emotion: amazing/terrible/extremely/very/somewhat/okay/etc."],
  "sentiment_intensity": "[how strongly they express opinions: extreme/moderate/mild]",
  "grammar_style": "[complete sentences/fragments/run-ons/mix]",
  "punctuation_style": "[heavy exclamations/minimal punctuation/ellipses/etc.]",
  "comprehensiveness": "[how thorough/sparse they are in their reviews, whether they address both positives and negatives for positive/negative reviews, etc.]",

  "focus_aspects": {{
    "general": ["what they typically comment on"]
  }},

  "typical_length": "[very short (1 sentence)/short (2-3 sentences)/medium (4-6)/long (7+)]",
  "typical_length_words": [average word count],
  "rating_patterns": {{
    "average": [average star rating],
    "tendency": "[harsh/generous/moderate]",
    "distribution": "[mostly 5s and 1s / balanced / etc.]"
  }},
  "personality_traits": ["descriptors of their personality inferred from reviews"],
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
{json.dumps(item_info, ensure_ascii=False)}

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
        user_review_samples = self._format_review_samples(user_reviews, max_samples=4)
        
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
        
        return self._parse_review_output(response)

    def _simple_refinement(self, draft: Dict[str, Any], user_profile: str, item_profile: str,
                          user_reviews: List[Dict], item_reviews: List[Dict]) -> Dict[str, Any]:
        """Self-critique and refinement to ensure alignment."""
        
        review_text = draft.get("review", "")
        stars = draft.get("stars", 3.0)
        
        # Sample reviews for comparison
        user_review_samples = self._format_review_samples(user_reviews, max_samples=3)
        item_review_samples = self._format_review_samples(item_reviews, max_samples=3)

        
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

CRITIQUE CHECKLIST:

1. STYLE CONSISTENCY:
   - Does it match user's vocabulary level and sentence structure?
   - Is formality/casualness consistent with user's style?
   - Are punctuation patterns similar?
   - Does length match user's typical length?
   - Is grammar style consistent?

2. SENTIMENT/EMOTION ALIGNMENT:
   - Does emotional tone match the rating intensity?
   - Are the intensity markers appropriate (amazing/great/okay/bad/terrible)?
   - Does the sentiment strength match the user's typical expression level?
   - Is sentiment consistent throughout?
   - Does it convey appropriate emotion for this rating?
   - Are we using extreme words (amazing/terrible) vs moderate words (good/bad) correctly?

3. TOPIC/SEMANTIC ALIGNMENT:
   - Does it mention specific, concrete features of this product/book/business?
   - Does it reflect the personality of the user and highlight their priorities and preferences?
   - Does it use relevant vocabulary from item reviews?
   - Are topics semantically similar to what others mention?
   - Does it avoid generic phrases?

4. COHERENCE:
   - Is it the right length?
   - Does it provide meaningful information?
   - Is it coherent and well-structured?

REFINEMENT STRATEGY:
- If personality and preferences do not fit: increase the user's identity highlighted in user_profile
- If style mismatches: adjust vocabulary, structure, formality
- If sentiment intensity is off: use stronger/weaker intensity markers (amazing→great→good→okay)
- If sentiment is off: strengthen or soften emotional expression to match rating
- If topics are generic: add specific item details
- If length is wrong: condense to match user's typical 2-5 sentence reviews
- Maintain the rating unless clearly misaligned

OUTPUT FORMAT:
Critique: [What needs refinement, if anything - be specific]
Refined Review: [The improved review, or original if no changes needed]
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages, temperature=0.1, max_tokens=350)  # Reduced to maintain conciseness
        
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
            
            # Find where 'review:' starts and take everything after it (no truncation)
            review_start_idx = text.lower().find('review:')
            if review_start_idx != -1:
                review = text[review_start_idx + len('review:'):].strip()
            else:
                # Fallback: try regex match
                review_match = re.search(r"review\s*[:=]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
                if review_match:
                    review = review_match.group(1).strip()
            
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