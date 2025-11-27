"""Shared utility functions for agent implementations.

This module provides common functionality used across multiple agent architectures:
- User and item profiling
- Review refinement
- Review sampling and formatting
- Output parsing
"""
import json
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger("agent_utils")


def build_user_profile(llm, user_reviews: List[Dict], user_profile_raw: Dict) -> str:
    """Build comprehensive structured user profile with review sampling.
    
    Args:
        llm: LLM instance to use for profile generation
        user_reviews: List of user's past reviews
        user_profile_raw: Raw user profile data
    
    Returns:
        JSON string containing structured user profile
    """
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
    sampled = sample_diverse_reviews(user_reviews, max_samples=15)
    reviews_text = "\n\n".join([
        f"Rating: {r.get('stars', 'N/A')} stars\nReview: {r.get('text', '')}"
        for r in sampled
    ])

    if source == "amazon":
        prompt = f"""Analyze this Amazon user's review patterns and create a STRUCTURED PROFILE.

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
  "purchase_patterns": "[categories/themes/uses shared between reviewed products, if any]",
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
        prompt = f"""Analyze this Goodreads user's review patterns and create a STRUCTURED PROFILE.

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
  "reading_preferences": ["genre", "themes", "writing styles they gravitate toward"],
  "literary_focus": ["plot", "character development", "writing style", "themes", "world-building", "etc."],
  "critical_depth": "[surface-level reactions / in-depth literary analysis / balanced]",
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
        prompt = f"""Analyze this Yelp user's review patterns and create a STRUCTURED PROFILE.

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
    "restaurants": ["service", "food quality", "ambiance", "etc."],
    "general": ["what they typically comment on"]
  }},
  "dining_preferences": "[type of cuisine, dining atmosphere, price sensitivity, etc.]",
  "service_sensitivity": "[how much they care about service quality vs food quality]",
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
        
    else:
        # Generic prompt for unknown sources
        prompt = f"""Analyze this user's review patterns and create a STRUCTURED PROFILE.

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
  "focus_aspects": ["what they typically comment on"],
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
    response = llm(messages=messages, temperature=0.0, max_tokens=800).strip()
    
    # Try to parse as JSON, if it fails return the raw response
    try:
        json.loads(response)
        return response
    except:
        logger.warning("User profile not valid JSON, using raw response")
        return response


def build_item_profile(llm, item_reviews: List[Dict], item_info: Dict) -> str:
    """Build structured item aspect profile with liberal review sampling.
    
    Args:
        llm: LLM instance to use for profile generation
        item_reviews: List of reviews for the item
        item_info: Raw item information
    
    Returns:
        JSON string containing structured item profile
    """

    if not item_reviews:
        return json.dumps({
            "common_themes": [],
            "sentiment_distribution": {},
            "pros": [],
            "cons": [],
            "polarizing_aspects": []
        })

    source = None
    try:
        source = item_info.get("source", None)
        if not source and item_reviews:
            source = item_reviews[0].get("source", None)
    except:
        source = None
    source = (source or "").lower()
    
    # Sample generously - up to 15 reviews for comprehensive profiling
    sampled = sample_diverse_reviews(item_reviews, max_samples=15)
    reviews_text = "\n\n".join([
        f"Rating: {r.get('stars', 'N/A')} stars\nReview: {r.get('text', '')}"
        for r in sampled
    ])

    if source == "amazon":
        prompt = f"""Analyze existing reviews for this Amazon product and create a STRUCTURED ASPECT PROFILE.

Item details:
{json.dumps(item_info, ensure_ascii=False)}

Item reviews ({len(sampled)} reviews):
{reviews_text}

Create a comprehensive structured profile with the following schema:

OUTPUT SCHEMA (JSON format):
{{
  "item_name": "[name of the item/business from the item details]",
  "item_type": "[product type]",
  "common_themes": ["list", "of", "aspects", "people", "discuss"],
  "theme_details": {
    "theme1": "what people say about this theme",
    "theme2": "what people say about this theme"
  },
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
  "specific_mentions": ["concrete items/features people name"],

  "usage_patterns": ["how customers typically use the product"],
  "durability_feedback": ["comments about longevity, build quality"],
  "value_for_money": "[perceived value: great/decent/poor]",
  "comparison_points": ["how reviewers compare this item to alternatives, if relevant"],
  "ease_of_use": "[easy/moderate/difficult]",
  "feature_performance": ["specific product features and how well they work"]
}}

Provide ONLY the JSON object, no additional text.
"""
    
    elif source == "goodreads":
        prompt = f"""Analyze existing reviews for this Goodreads-listed book and create a STRUCTURED ASPECT PROFILE.

Item details:
{json.dumps(item_info, ensure_ascii=False)}

Item reviews ({len(sampled)} reviews):
{reviews_text}

Create a comprehensive structured profile with the following schema:

OUTPUT SCHEMA (JSON format):
{{
  "item_name": "[name of the item/business from the item details]",
  "common_themes": ["list", "of", "aspects", "people", "discuss"],
  "theme_details": {
    "theme1": "what people say about this theme",
    "theme2": "what people say about this theme"
  },
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
  "specific_mentions": ["concrete items/features people name"],

  "genre_classification": "[genre(s) based on item info and reviews]",
  "writing_style_feedback": ["comments on prose, pacing, clarity, complexity"],
  "plot_feedback": ["what reviewers say about plot quality"],
  "character_feedback": ["opinions on character depth, relatability"],
  "reading_experience": "[immersive/slow/dense/engaging/etc.]",
  "comparisons_to_other_works": ["authors or works reviewers compare this to, if relevant"]
}}

Provide ONLY the JSON object, no additional text.
"""

    elif source == "yelp":
        prompt = f"""Analyze existing reviews for this Yelp-listed business and create a STRUCTURED ASPECT PROFILE.

Item details:
{json.dumps(item_info, ensure_ascii=False)}

Item reviews ({len(sampled)} reviews):
{reviews_text}

Create a comprehensive structured profile with the following schema:

OUTPUT SCHEMA (JSON format):
{{
  "item_name": "[name of the item/business from the item details]",
  "item_type": "[business type]",
  "common_themes": ["list", "of", "aspects", "people", "discuss"],
  "theme_details": {
    "theme1": "what people say about this theme",
    "theme2": "what people say about this theme"
  },
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
  "specific_mentions": ["concrete items/features people name"],

  "food_quality": ["comments on taste, freshness, portion size OR 'N/A' if irrelevant to the business"],
  "service_quality": ["attentiveness, speed, friendliness OR 'N/A' if irrelevant to the business"],
  "ambiance_feedback": ["vibe, decor, noise OR 'N/A' if irrelevant to the business"],
  "price_sensitivity": "[expensive/moderate/cheap/value-focused/etc.]",
  "wait_time_feedback": ["comments on wait times OR 'N/A' if irrelevant to the business]",
  "popular_dishes_or_services": ["menu items OR key services offered for non-food businesses"],
  "recurring_issues": ["recurring complaints (service, cleanliness, reliability, etc.)"]
}}

Provide ONLY the JSON object, no additional text.
"""

    else: 
        prompt = f"""Analyze existing reviews for this item/business and create a STRUCTURED ASPECT PROFILE.

NOTE: The 'item' can be either a physical product OR a service/business (restaurant, hotel, etc.) OR a book.

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
    response = llm(messages=messages, temperature=0.0, max_tokens=800).strip()
    
    try:
        json.loads(response)
        return response
    except:
        logger.warning("Item profile not valid JSON, using raw response")
        return response


def refine_review(llm, draft: Dict[str, Any], user_profile: str, item_profile: str,
                  user_reviews: List[Dict], item_reviews: List[Dict]) -> Dict[str, Any]:
    """Self-critique and refinement to ensure alignment.
    
    Args:
        llm: LLM instance to use for refinement
        draft: Draft review with 'stars' and 'review' keys
        user_profile: Structured user profile (JSON string)
        item_profile: Structured item profile (JSON string)
        user_reviews: List of user's past reviews
        item_reviews: List of item's reviews
    
    Returns:
        Refined review dict with 'stars' and 'review' keys
    """
    review_text = draft.get("review", "")
    stars = draft.get("stars", 3.0)
    
    # Sample reviews for comparison
    user_review_samples = format_review_samples(user_reviews, max_samples=3)
    item_review_samples = format_review_samples(item_reviews, max_samples=3)
    
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
   ✓ Does it match user's vocabulary level and sentence structure?
   ✓ Is formality/casualness consistent with user's style?
   ✓ Are punctuation patterns similar?
   ✓ Does length match user's typical length?
   ✓ Is grammar style consistent?

2. SENTIMENT/EMOTION ALIGNMENT:
   ✓ Does emotional tone match the rating intensity?
   ✓ Are the intensity markers appropriate (amazing/great/okay/bad/terrible)?
   ✓ Does the sentiment strength match the user's typical expression level?
   ✓ Is sentiment consistent throughout?
   ✓ Does it convey appropriate emotion for this rating?
   ✓ Are we using extreme words (amazing/terrible) vs moderate words (good/bad) correctly?

3. TOPIC/SEMANTIC ALIGNMENT:
   ✓ Does it mention specific, concrete features of this product/book/business?
   ✓ Does it reflect the personality of the user and highlight their priorities and preferences?
   ✓ Does it use relevant vocabulary from item reviews?
   ✓ Are topics semantically similar to what others mention?
   ✓ Does it avoid generic phrases?

4. COHERENCE:
   ✓ Is it the right length?
   ✓ Does it provide meaningful information?
   ✓ Is it coherent and well-structured?

REFINEMENT STRATEGY:
- If personality and preferences do not fit: increase the user's identity highlighted in user_profile
- If style mismatches: adjust vocabulary, structure, formality
- If sentiment intensity is off: use stronger/weaker intensity markers (amazing→great→good→okay)
- If sentiment is off: strengthen or soften emotional expression to match rating
- If topics are generic: add specific item details
- If length is wrong: condense to match user's typical 2-5 sentence reviews
- Maintain the rating unless clearly misaligned

OUTPUT FORMAT (OUTPUT ONLY THE FOLLOWING):
Critique: [What needs refinement, if anything - be specific]
Refined Review: [The improved review, or original if no changes needed]
"""
    
    messages = [{"role": "user", "content": prompt}]
    response = llm(messages=messages, temperature=0.1, max_tokens=350)
    
    refined_review = parse_refinement_output(response, original_review=review_text)
    logger.info(f"Refinement: {refined_review[:100]}...")
    draft["review"] = refined_review
    
    return draft


def sample_diverse_reviews(reviews: List[Dict], max_samples: int = 5) -> List[Dict]:
    """Sample reviews to get diverse ratings.
    
    Args:
        reviews: List of review dicts
        max_samples: Maximum number of samples to return
    
    Returns:
        List of sampled reviews with diverse ratings
    """
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


def format_review_samples(reviews: List[Dict], max_samples: int = 3) -> str:
    """Format sample reviews for style reference.
    
    Args:
        reviews: List of review dicts
        max_samples: Maximum number of samples to format
    
    Returns:
        Formatted string with review examples
    """
    if not reviews:
        return "No past reviews available."
    
    sampled = sample_diverse_reviews(reviews, max_samples=max_samples)
    formatted = []
    for i, review in enumerate(sampled, 1):
        stars = review.get("stars", "N/A")
        text = review.get("text", "")
        formatted.append(f"Example {i}:\nRating: {stars} stars\nReview: {text}")
    
    return "\n\n".join(formatted)


def parse_review_output(text: str) -> Dict[str, Any]:
    """Parse LLM output into stars and review.
    
    Args:
        text: Raw LLM output text
    
    Returns:
        Dict with 'stars' and 'review' keys
    """
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


def parse_refinement_output(text: str, original_review: str) -> str:
    """Parse refinement output to extract the refined review.
    
    Args:
        text: Raw LLM refinement output
        original_review: Original review text to use as fallback
    
    Returns:
        Refined review text
    """
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
