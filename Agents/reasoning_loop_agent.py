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
from Util import format_llm_logs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reasoning_loop_agent")


class ReasoningLoopAgent(SimulationAgent):
    def __init__(self, llm: LLMBase, max_reasoning_steps: int = 4):
        super().__init__(llm=llm)
        self.max_reasoning_steps = max_reasoning_steps
        self.memory = MemoryDILU(llm=llm)

    def workflow(self) -> Dict[str, Any]:
        try:
            user_id = self.task.get("user_id")
            item_id = self.task.get("item_id")

            user_profile = self.interaction_tool.get_user(user_id=user_id)
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id) or []
            item_info = self.interaction_tool.get_item(item_id=item_id)
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id) or []
            
            logger.info(f"Fetched {len(user_reviews)} user reviews, {len(item_reviews)} item reviews")

            user_analysis = self._analyze_user_reviews(user_reviews)
            logger.info(f"User analysis: {user_analysis[:100]}...")

            item_analysis = self._analyze_item_reviews(item_reviews)
            logger.info(f"Item analysis: {item_analysis[:100]}...")

            reasoning_insights = self._reasoning_loop(user_analysis, item_analysis, user_profile, item_info)
            logger.info(f"Completed {len(reasoning_insights)} reasoning steps")

            draft = self._generate_draft(reasoning_insights, user_analysis, item_analysis)
            logger.info(f"Draft: stars={draft['stars']}, review={draft['review'][:50]}...")

            final = self._simple_refinement(draft, user_analysis)
            
            return final

        except Exception as e:
            logger.exception("ReasoningLoopAgent workflow failed: %s", e)
            return {"stars": 0.0, "review": ""}

    def _analyze_user_reviews(self, user_reviews: List[Dict]) -> str:
        if not user_reviews:
            return "No user review history available."
        
        sampled = self._sample_diverse_reviews(user_reviews, max_samples=5)
        reviews_text = "\n\n".join([
            f"Rating: {r.get('stars', 'N/A')}\nReview: {r.get('text', '')[:500]}"
            for r in sampled
        ])
        
        prompt = f"""Analyze this user's review patterns. Provide a concise summary covering:
1. Rating tendency (harsh/generous/moderate, average rating)
2. Review style (formal/casual, length, tone, linguistic patterns)
3. Lifestyle/preferences (are they a professional, student, parent? any lifestyle clues?)
4. Item preferences (what do they value most? what do they criticize?)
5. Personality traits evident in reviews

User's review samples:
{reviews_text}

Keep response under 200 words, be specific and data-driven.
"""
        
        messages = [{"role": "user", "content": prompt}]
        return self.llm(messages=messages, temperature=0.0, max_tokens=300).strip()

    def _analyze_item_reviews(self, item_reviews: List[Dict]) -> str:
        if not item_reviews:
            return "No existing reviews for this item."
        
        # Sample diverse reviews
        sampled = self._sample_diverse_reviews(item_reviews, max_samples=5)
        reviews_text = "\n\n".join([
            f"Rating: {r.get('stars', 'N/A')}\nReview: {r.get('text', '')[:300]}"
            for r in sampled
        ])
        
        prompt = f"""Analyze existing reviews for this item. Provide a concise summary covering:
1. Overall sentiment and average rating
2. Most commonly mentioned benefits/pros (with specific vocabulary used)
3. Most commonly mentioned issues/cons (with specific vocabulary used)
4. Emotional tone patterns (enthusiastic, disappointed, neutral, etc.)
5. Common phrases and vocabulary that appear across reviews
6. Notable thematic patterns

Item reviews:
{reviews_text}

Keep response under 250 words, be specific about patterns and exact terminology used.
"""
        
        messages = [{"role": "user", "content": prompt}]
        return self.llm(messages=messages, temperature=0.0, max_tokens=300).strip()

    def _reasoning_loop(self, user_analysis: str, item_analysis: str, 
                       user_profile: Dict, item_info: Dict) -> List[Dict[str, str]]:
        reasoning_insights = []
        
        # Context that persists across all reasoning steps
        persistent_context = f"""USER ANALYSIS:
{user_analysis}

ITEM ANALYSIS:
{item_analysis}

User Profile Summary: {json.dumps(user_profile, ensure_ascii=False)[:500]}
Item Info Summary: {json.dumps(item_info, ensure_ascii=False)[:500]}"""
        
        for step in range(self.max_reasoning_steps):
            logger.info(f"Reasoning step {step + 1}/{self.max_reasoning_steps}")
            
            # Build summary of previous insights
            previous_insights = self._format_previous_insights(reasoning_insights)
            
            # Generate next reasoning insight
            insight = self._generate_reasoning_step(
                persistent_context, 
                previous_insights, 
                step + 1,
                self.max_reasoning_steps
            )
            
            # Check if reasoning should continue
            if insight.get("action") == "DONE":
                logger.info("Reasoning complete - sufficient insights gathered")
                break
            
            reasoning_insights.append(insight)
            logger.info(f"Insight {step + 1}: {insight.get('insight', '')[:100]}...")
        
        return reasoning_insights

    def _generate_reasoning_step(self, persistent_context: str, previous_insights: str,
                                 step_num: int, max_steps: int) -> Dict[str, str]:
                
        prompt = f"""You are in a REASONING LOOP to determine what a review should contain.
Your task is to provide INSIGHTS and INSTRUCTIONS for the review writer, NOT to write the review.

{persistent_context}

Previous reasoning insights:
{previous_insights if previous_insights else "This is your first reasoning step."}

Current step: {step_num}/{max_steps}

CRITICAL INSTRUCTIONS:
1. Your output should be INSIGHTS about what the review should contain
2. Think of yourself as giving instructions to the review writer
3. Be specific about: rating prediction, key points to mention, tone/style to use
4. If your reasoning is complex, indicate that MORE STEPS are needed
5. If you have sufficient insights for a complete review, output DONE

GUIDELINES FOR COMPARTMENTALIZATION:
- Break complex reasoning into focused steps
- Each step should address ONE specific aspect (e.g., "rating prediction", "key topics", "style matching")
- Don't try to reason about everything at once
- BUT: Stop when you have enough insights - avoid unnecessary steps

(IMPORTANT) AVOID REPETITION:
- Review your previous insights carefully
- Do NOT repeat information already covered in previous steps
- Focus on NEW, complementary aspects not yet addressed
- If you find yourself repeating previous insights, choose DONE instead
- Each step should add UNIQUE value to the reasoning chain

Suggested progression (choose what's NOT covered yet):
Step 1: Rating prediction and justification
Step 2: SPECIFIC TOPICS/ASPECTS - What concrete business features to discuss (e.g., "food quality", "service speed", "ambiance", "price") - be specific!
Step 3: SEMANTIC CONTENT - What specific words, phrases, and concepts should appear? What topics from item reviews should be echoed?
Step 4: Tone, style, and linguistic patterns to match
 
OUTPUT FORMAT (exactly as below):
Thought: [What NEW aspect will you reason about? Why hasn't this been covered yet?]
Insight: [Your specific NEW insight/instruction - must be different from previous steps]
Action: [CONTINUE if more reasoning needed, DONE if sufficient insights gathered]

EXAMPLES:

Example 1 (first step - rating focus):
Thought: I need to predict what rating this user would give based on their tendency and item characteristics.
Insight: User rates generously (avg 4.3) and this item has strong qualities in areas the user values (service, ambiance). Predict 4-5 stars. The review should reflect positive experience but mention any minor service issues noted in item reviews.
Action: CONTINUE
 
Example 2 (second step - CONCRETE TOPICS focus, DIFFERENT from step 1):
Thought: Step 1 covered rating prediction. Now I need to identify SPECIFIC CONCRETE ASPECTS of THIS business that the review should discuss.
Insight: Based on item reviews, this restaurant's standout features are: (1) handmade pasta with authentic Italian recipes, (2) attentive waitstaff who explain menu items. User typically discusses food quality and service. Review MUST mention: the pasta quality specifically, the helpful service experience. Avoid: generic comments about "good food" - be specific to THIS business.
Action: CONTINUE

Example 3 (third step - SEMANTIC CONTENT, building on steps 1-2):
Thought: I know WHAT topics to cover. Now I need to specify the SEMANTIC CONTENT - what concepts and vocabulary should be included for topic similarity.
Insight: Key semantic elements to include: "pasta" (or "noodles"), "fresh"/"homemade", "service"/"staff"/"waiter", "attentive"/"helpful". Mirror terminology from real reviews: if they say "authentic Italian" use similar phrases. These words ensure semantic/topic alignment with actual reviews of this business.
Action: CONTINUE

Example 4 (fourth step - style focus, DIFFERENT from steps 1-3):
Thought: Steps 1-3 covered what to say and semantic content. Now I need to specify HOW to say it - tone and style.
Insight: User writes in casual enthusiastic tone with conversational language ("me and my husband", "love this place"). Review should be 2-3 sentences, use first-person, include exclamation points, and feel spontaneous rather than formal. Match user's energetic vocabulary.
Action: DONE

Example of BAD repetition (what NOT to do):
Step 1: Predict 4 stars based on user tendency
Step 2: User rates positively, so give 4-5 stars <- REPETITIVE! Already covered rating.
Instead, Step 2 should cover content, style, or another NEW dimension.

NOW PROVIDE YOUR REASONING STEP (output 3 lines: Thought, Insight, Action):
"""
        
        messages = [
            {
                "role": "system",
                "content": "You provide reasoning insights for review generation. Output EXACTLY 3 lines: Thought, Insight, Action. Focus on insights, not generating the actual review."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = self.llm(
            messages=messages,
            temperature=0.2,
            max_tokens=250,
            stop_strs=["\n\n", "Example", "Step"]
        )
        
        return self._parse_reasoning_output(response)

    def _parse_reasoning_output(self, text: str) -> Dict[str, str]:
        result = {"thought": "", "insight": "", "action": "CONTINUE"}
        
        try:
            # Extract thought
            thought_match = re.search(r"Thought:\s*(.+?)(?=\nInsight:|\Z)", text, re.IGNORECASE | re.DOTALL)
            if thought_match:
                result["thought"] = thought_match.group(1).strip()
            
            # Extract insight
            insight_match = re.search(r"Insight:\s*(.+?)(?=\nAction:|\Z)", text, re.IGNORECASE | re.DOTALL)
            if insight_match:
                result["insight"] = insight_match.group(1).strip()
            
            # Extract action
            action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
            if action_match:
                action = action_match.group(1).strip().upper()
                if "DONE" in action:
                    result["action"] = "DONE"
                else:
                    result["action"] = "CONTINUE"
        
        except Exception as e:
            logger.warning(f"Failed to parse reasoning output: {e}")
        
        return result

    def _format_previous_insights(self, insights: List[Dict[str, str]]) -> str:
        if not insights:
            return "No previous insights yet."
        
        formatted = []
        for i, insight in enumerate(insights, 1):
            formatted.append(f"Step {i}: {insight.get('insight', '')}")
        
        return "\n".join(formatted)

    def _generate_draft(self, reasoning_insights: List[Dict[str, str]], 
                       user_analysis: str, item_analysis: str) -> Dict[str, Any]:
        
        # Compile all insights
        insights_summary = "\n".join([
            f"- {ins.get('insight', '')}" for ins in reasoning_insights
        ])
        
        # Get sample past user reviews for style reference
        user_id = self.task.get("user_id")
        user_reviews = self.interaction_tool.get_reviews(user_id=user_id) or []
        user_review_samples = self._format_review_samples(user_reviews, max_samples=3)
 
        # Get concrete item details for topic alignment
        item_id = self.task.get("item_id")
        item_info = self.interaction_tool.get_item(item_id=item_id)
        item_details = json.dumps(item_info, ensure_ascii=False)[:500]
        
        # Use memory-based similarity search like baseline to find most relevant item review
        item_reviews = self.interaction_tool.get_reviews(item_id=item_id) or []
        for review in item_reviews:
            review_text = review.get('text', '')
            if review_text:
                self.memory(f'review: {review_text}')
        
        similar_item_review = ""
        if user_reviews and self.memory.scenario_memory._collection.count() > 0:
            # Find item review most similar to user's style
            similar_item_review = self.memory(f'{user_reviews[0]["text"]}')
            logger.info(f"Found similar item review (length: {len(similar_item_review)} chars)")
 
        prompt = f"""Generate a review based on the following reasoning insights.

PAST USER REVIEW EXAMPLES (study these carefully for authentic style):
{user_review_samples}

SIMILAR ITEM REVIEW (reference for semantic/emotional alignment):
{similar_item_review if similar_item_review else "No similar review found."}

REASONING INSIGHTS (what to say - the content/message):
{insights_summary}

USER ANALYSIS (additional patterns):
{user_analysis}

ITEM ANALYSIS (background context - note vocabulary and emotional patterns):
{item_analysis}
 
CONCRETE BUSINESS DETAILS (use these for specific mentions):
{item_details}
 
CRITICAL INSTRUCTIONS - You must balance TWO requirements:

1. STYLE MIMICRY (HOW to write):
   - Study the example reviews above and MIMIC their exact writing style
   - Copy their vocabulary level, sentence structure, and phrasing patterns
   - Match their formality level (casual vs formal)
   - Use similar punctuation patterns (exclamation marks, ellipses, etc.)
   - Match their typical review length (count words in examples)
   - Adopt their grammar patterns (fragments, run-ons, perfect grammar, etc.)
   - Mirror their emotional expression style (enthusiastic, reserved, dramatic, etc.)
   - MATCH THE EMOTIONAL TONE: if examples are positive/negative/neutral, maintain that sentiment
   - Use vocabulary that appears in the Similar Item Review when discussing this business
   - If they use specific phrases or expressions, incorporate similar ones

2. CONTENT GUIDANCE (WHAT to say):
   - Follow the rating prediction from the reasoning insights
   - Include the SPECIFIC CONCRETE topics/aspects mentioned in the insights
   - Use SPECIFIC BUSINESS DETAILS (actual menu items, specific services, concrete features)
   - EXPRESS SEMANTICALLY SIMILAR CONCEPTS to what real reviewers mention for this business
   - Include KEY VOCABULARY and concepts from the insights (specific words that ensure topic alignment)
   - Express the sentiment and opinions indicated by the insights
   - Cover the points the insights say to emphasize or avoid
   - AVOID generic comments ("good food", "nice place") - be business-specific!
 
THINK OF IT THIS WAY:
- The EXAMPLES show you the user's authentic "voice" and writing patterns → copy this style
- The INSIGHTS tell you the message/content this specific review should convey → use these facts
- Your job: deliver the insights' message while sounding EXACTLY like the user in the examples

(IMPORTANT) COMMON MISTAKE TO AVOID:
Do NOT write in a generic, polished, or formal style if the examples are casual/spontaneous.
Do NOT use vocabulary or sentence structures that don't appear in the examples.
If the examples are brief and casual, your output should be too - even if the insights mention many topics, weave them into the user's natural style.

Output format:
stars: [1.0|2.0|3.0|4.0|5.0]
review: [your review text - written in the USER'S style from examples, containing the message from insights]
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages, temperature=0.1, max_tokens=400)  # Lower temperature for more deterministic, aligned output
        
        return self._parse_review_output(response)

    def _simple_refinement(self, draft: Dict[str, Any], user_analysis: str) -> Dict[str, Any]:        
        review_text = draft.get("review", "")
        stars = draft.get("stars", 3.0) # Default to 3 stars if missing
        
        # Get user and item context for alignment checking
        user_id = self.task.get("user_id")
        item_id = self.task.get("item_id")
        
        user_reviews = self.interaction_tool.get_reviews(user_id=user_id) or []
        item_reviews = self.interaction_tool.get_reviews(item_id=item_id) or []
        item_info = self.interaction_tool.get_item(item_id=item_id)
        
        # Sample user and item reviews for comparison
        user_review_samples = self._format_review_samples(user_reviews, max_samples=3)
        item_review_samples = self._format_review_samples(item_reviews, max_samples=3)
        
        # Get the similar item review from memory for additional context
        similar_item_review = ""
        if user_reviews and self.memory.scenario_memory._collection.count() > 0:
            similar_item_review = self.memory(f'{user_reviews[0]["text"]}')
        
        item_name = item_info.get("name", "this business")
        item_details = json.dumps(item_info, ensure_ascii=False)[:400]
        
        prompt = f"""You are refining a generated review to ensure it meets quality standards.

GENERATED REVIEW TO REFINE:
Rating: {stars} stars
Review: "{review_text}"

USER'S PAST REVIEW EXAMPLES (for style/tone reference):
{user_review_samples}

ITEM'S PAST REVIEW EXAMPLES (for topic/vocabulary reference):
{item_review_samples}

MOST SIMILAR ITEM REVIEW (semantic/emotional reference):
{similar_item_review if similar_item_review else "No similar review available."}

BUSINESS DETAILS:
{item_details}

USER ANALYSIS SUMMARY:
{user_analysis[:300]}

YOUR TASK: Analyze and refine the generated review to ensure:

1. STYLE CONSISTENCY (compare with user's examples):
   - Does it match the user's vocabulary level and sentence structure?
   - Is the formality/casualness consistent with user's style?
   - Are punctuation patterns similar (exclamations, ellipses, etc.)?
   - Does review length match user's typical length?
   - Is the grammar style consistent (fragments vs. complete sentences)?

2. SENTIMENT/EMOTION ALIGNMENT:
   - Does the emotional tone match the rating? (e.g., 5 stars = enthusiastic, 1 star = disappointed)
   - Is the sentiment consistent throughout the review?
   - Does it convey appropriate emotional intensity for this rating?

3. TOPIC/SEMANTIC ALIGNMENT (compare with item reviews):
   - Does it mention specific, concrete features of THIS business?
   - Does it use relevant vocabulary that appears in other reviews of this item?
   - Are the topics discussed semantically similar to what others mention?
   - Does it avoid generic phrases in favor of business-specific details?

4. COHERENCE AND COMPLETENESS:
   - Is the review too brief or too verbose compared to user's style?
   - Does it provide meaningful information about the experience?
   - Is it coherent and well-structured?

REFINEMENT STRATEGY:
- If style doesn't match user's examples: adjust vocabulary, sentence structure, formality
- If sentiment/emotion is off: strengthen or soften emotional expression to match rating
- If topics are too generic: add specific business details and vocabulary from item reviews
- If length is wrong: expand with specifics or condense while keeping key points
- Maintain the original rating ({stars} stars) unless it's clearly misaligned

OUTPUT FORMAT:
Analysis: [Brief analysis of what needs refinement - 1-2 sentences]
Refined Review: [The improved review text that addresses alignment issues]

If the review is already excellent and needs no changes, output:
Analysis: Review is well-aligned with user style, appropriate sentiment, and specific topics.
Refined Review: {review_text}
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages, temperature=0.1, max_tokens=500)
        
        # Parse the refinement response
        refined_review = self._parse_refinement_output(response, original_review=review_text)
        
        logger.info(f"Refinement: {refined_review[:100]}...")
        draft["review"] = refined_review
        
        return draft
    
    def _parse_refinement_output(self, text: str, original_review: str) -> str:
        try:
            # Look for "Refined Review:" section
            refined_match = re.search(r"Refined Review:\s*(.+?)(?=\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
            if refined_match:
                refined = refined_match.group(1).strip()
                # Remove quotes if present
                refined = refined.strip('"\'')
                return refined if refined else original_review
            
            # Fallback: look for lines starting with the review
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if line.lower().startswith('refined review:'):
                    # Get everything after "Refined Review:"
                    review_start = i
                    review_parts = [line.split(':', 1)[1].strip() if ':' in line else '']
                    # Collect subsequent lines that are part of the review
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip() and not lines[j].lower().startswith('analysis'):
                            review_parts.append(lines[j].strip())
                        elif lines[j].strip():
                            break
                    refined = ' '.join(review_parts).strip('"\'')
                    return refined if refined else original_review
            
            # If parsing fails, return original
            logger.warning("Could not parse refinement output, using original review")
            return original_review
            
        except Exception as e:
            logger.warning(f"Refinement parse error: {e}, using original review")
            return original_review

    def _parse_review_output(self, text: str) -> Dict[str, Any]:
        stars = 3.0  # Default fallback
        review = ""
        
        try:
            # Parse stars
            stars_match = re.search(r"stars\s*[:=]\s*\[?\s*([0-9]+(?:\.[0-9]+)?)\s*\]?", text, re.IGNORECASE)
            if stars_match:
                stars = float(stars_match.group(1))
                stars = max(1.0, min(5.0, stars))
            
            # Parse review
            review_match = re.search(r"review\s*[:=]\s*(.+?)(?=\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
            if review_match:
                review = review_match.group(1).strip()
            elif not review_match:
                # Fallback: look line by line
                for line in text.split("\n"):
                    if line.lower().startswith("review:"):
                        review = line.split(":", 1)[1].strip()
                        break
            
            # Final fallback
            if not review:
                review = text.strip()
        
        except Exception as e:
            logger.warning(f"Parse error: {e}")
            review = text if text else ""
        
        return {"stars": stars, "review": review}

    def _sample_diverse_reviews(self, reviews: List[Dict], max_samples: int = 5) -> List[Dict]:
        if len(reviews) <= max_samples:
            return reviews
        
        # Try to get diverse ratings
        by_rating = {}
        for r in reviews:
            rating = r.get("stars", 3.0)
            if rating not in by_rating:
                by_rating[rating] = []
            by_rating[rating].append(r)
        
        # Sample from each rating bucket
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
            text = review.get("text", "")[:700]  # Limit length
            formatted.append(f"Example {i}:\nRating: {stars} stars\nReview: {text}")
        
        return "\n\n".join(formatted)


if __name__ == "__main__":
    from Util import format_llm_logs
    import os


    print(os.pardir)
    sim = Simulator(data_dir="dataset", device="gpu", cache=True)
    here = os.path.dirname(__file__)

    task_set = "yelp" # "goodreads" or "yelp"
    task_dir = os.path.join("example", "track1", task_set, "tasks")
    groundtruth_dir = os.path.join("example", "track1", task_set, "groundtruth")
    sim.set_task_and_groundtruth(task_dir=task_dir, groundtruth_dir=groundtruth_dir)

    llm = OllamaLLM(model="mistral")
    sim.set_agent(ReasoningLoopAgent)
    sim.set_llm(llm)


    # Set to True for single tasked debugging
    if True:
        print("Running reasoning loop agent for task 0...")
        res = sim.run_single_task(task_index=0, wrap_llm_with_logger=True)
        
        # Prepare structured log data including the LLM calls so the formatter can consume it
        output_data = {
            "output": res.get("output"),
            "llm_calls": res.get("llm_calls", [])
        }

        # Print a short summary to stdout
        print(json.dumps({"output_present": bool(output_data["output"]), "llm_call_count": len(output_data["llm_calls"])}, indent=2))

        # Save JSON log
        log_json_path = f'./Outputs/reasoning_loop_output.json' 
        with open(log_json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Saved detailed logs to {log_json_path}")

        # Also create a human-readable formatted text file using format_llm_logs.py
        formatted_txt_path = f'./Outputs/reasoning_loop_output.txt'
        try:
            format_llm_logs(log_json_path, formatted_txt_path)
            print(f"Saved formatted LLM logs to {formatted_txt_path}")
        except Exception as e:
            logger.exception("Failed to format LLM logs: %s", e)
    else:
        outputs = sim.run_simulation(number_of_tasks=80)
        evaluation_results = sim.evaluate()
        with open(f'./Outputs/evaluation_results_track1_{task_set}.json', 'w') as f:
            json.dump(evaluation_results, f, indent=4)

        print(f"The evaluation_results is :{evaluation_results}")

