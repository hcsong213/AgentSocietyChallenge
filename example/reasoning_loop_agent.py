"""Simplified agent focused on the Reasoning Loop component.

This agent implements a compartmentalized reasoning loop that:
1. Analyzes past user reviews (provided by teammate)
2. Analyzes past item reviews (provided by teammate)
3. Uses reasoning loop to determine what the review should contain
4. Generates initial draft based on reasoning insights
5. Simple refinement (barebones placeholder for teammate's work)

The reasoning loop provides instructions/insights rather than generating the review directly.
"""
from __future__ import annotations

import json
import re
import logging
from typing import Dict, Any, List

from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import OllamaLLM, LLMBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reasoning_loop_agent")


class ReasoningLoopAgent(SimulationAgent):
    """Agent with focused reasoning loop for review generation."""

    def __init__(self, llm: LLMBase, max_reasoning_steps: int = 4):
        super().__init__(llm=llm)
        self.max_reasoning_steps = max_reasoning_steps

    def workflow(self) -> Dict[str, Any]:
        """Execute the workflow: analyze → reason → draft → refine."""
        if not self.interaction_tool:
            raise RuntimeError("interaction_tool is required")

        try:
            user_id = self.task.get("user_id")
            item_id = self.task.get("item_id")

            # Fetch data
            user_profile = self.interaction_tool.get_user(user_id=user_id)
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id) or []
            item_info = self.interaction_tool.get_item(item_id=item_id)
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id) or []
            
            logger.info(f"Fetched {len(user_reviews)} user reviews, {len(item_reviews)} item reviews")

            # Step 1: Analyze past user reviews (teammate's component - simplified version)
            user_analysis = self._analyze_user_reviews(user_reviews)
            logger.info(f"User analysis: {user_analysis[:100]}...")

            # Step 2: Analyze past item reviews (teammate's component - simplified version)
            item_analysis = self._analyze_item_reviews(item_reviews)
            logger.info(f"Item analysis: {item_analysis[:100]}...")

            # Step 3: REASONING LOOP (our focus)
            reasoning_insights = self._reasoning_loop(user_analysis, item_analysis, user_profile, item_info)
            logger.info(f"Completed {len(reasoning_insights)} reasoning steps")

            # Step 4: Generate draft based on reasoning insights
            draft = self._generate_draft(reasoning_insights, user_analysis, item_analysis)
            logger.info(f"Draft: stars={draft['stars']}, review={draft['review'][:50]}...")

            # Step 5: Simple refinement (teammate's component - barebones placeholder)
            final = self._simple_refinement(draft, user_analysis)
            
            return final

        except Exception as e:
            logger.exception("ReasoningLoopAgent workflow failed: %s", e)
            return {"stars": 0.0, "review": ""}

    def _analyze_user_reviews(self, user_reviews: List[Dict]) -> str:
        """Analyze user's review history (simplified - will be teammate's work).
        
        Extracts: rating tendency, style, preferences, personality.
        """
        if not user_reviews:
            return "No user review history available."
        
        # Sample diverse reviews
        sampled = self._sample_diverse_reviews(user_reviews, max_samples=5)
        reviews_text = "\n\n".join([
            f"Rating: {r.get('stars', 'N/A')}\nReview: {r.get('text', '')[:300]}"
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
        """Analyze item's existing reviews (simplified - will be teammate's work).
        
        Extracts: common issues, benefits, overall sentiment.
        """
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
2. Most commonly mentioned benefits/pros
3. Most commonly mentioned issues/cons
4. Notable patterns or themes

Item reviews:
{reviews_text}

Keep response under 200 words, be specific about patterns.
"""
        
        messages = [{"role": "user", "content": prompt}]
        return self.llm(messages=messages, temperature=0.0, max_tokens=300).strip()

    def _reasoning_loop(self, user_analysis: str, item_analysis: str, 
                       user_profile: Dict, item_info: Dict) -> List[Dict[str, str]]:
        """REASONING LOOP - Core component (our focus).
        
        The loop generates insights about what the review SHOULD contain,
        rather than generating the review directly. These insights guide
        the draft generation.
        
        Returns list of reasoning insights.
        """
        reasoning_insights = []
        
        # Context that persists across all reasoning steps
        persistent_context = f"""USER ANALYSIS:
{user_analysis}

ITEM ANALYSIS:
{item_analysis}

User Profile Summary: {json.dumps(user_profile, ensure_ascii=False)[:300]}
Item Info Summary: {json.dumps(item_info, ensure_ascii=False)[:300]}"""
        
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
        """Generate one reasoning step that provides insight about what review should contain."""
        
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

⚠️ AVOID REPETITION:
- Review your previous insights carefully
- Do NOT repeat information already covered in previous steps
- Focus on NEW, complementary aspects not yet addressed
- If you find yourself repeating previous insights, choose DONE instead
- Each step should add UNIQUE value to the reasoning chain

Suggested progression (choose what's NOT covered yet):
Step 1: Rating prediction and justification
Step 2: Specific content/topics to emphasize or avoid
Step 3: Tone, style, and linguistic patterns to match
Step 4: Additional nuances (length, specific phrases, emotional tone)

OUTPUT FORMAT (exactly as below):
Thought: [What NEW aspect will you reason about? Why hasn't this been covered yet?]
Insight: [Your specific NEW insight/instruction - must be different from previous steps]
Action: [CONTINUE if more reasoning needed, DONE if sufficient insights gathered]

EXAMPLES:

Example 1 (first step - rating focus):
Thought: I need to predict what rating this user would give based on their tendency and item characteristics.
Insight: User rates generously (avg 4.3) and this item has strong qualities in areas the user values (service, ambiance). Predict 4-5 stars. The review should reflect positive experience but mention any minor service issues noted in item reviews.
Action: CONTINUE

Example 2 (second step - content focus, DIFFERENT from step 1):
Thought: Step 1 covered rating prediction. Now I need to determine what SPECIFIC aspects the review should discuss.
Insight: User always discusses food quality and service in detail. Item reviews highlight excellent food but inconsistent service. Review should lead with food quality praise (2-3 specific positive phrases), then briefly acknowledge service variability without being negative. Avoid mentioning price since user rarely discusses it.
Action: CONTINUE

Example 3 (third step - style focus, DIFFERENT from steps 1-2):
Thought: Steps 1-2 covered what to say. Now I need to specify HOW to say it - tone and style.
Insight: User writes in casual enthusiastic tone with conversational language ("me and my husband", "love this place"). Review should be 2-3 sentences, use first-person, include exclamation points, and feel spontaneous rather than formal. Match user's energetic vocabulary.
Action: DONE

Example of BAD repetition (what NOT to do):
Step 1: Predict 4 stars based on user tendency
Step 2: User rates positively, so give 4-5 stars ← REPETITIVE! Already covered rating.
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
            temperature=0.2,  # Slightly higher for creative reasoning
            max_tokens=250,
            stop_strs=["\n\n", "Example", "Step"]
        )
        
        return self._parse_reasoning_output(response)

    def _parse_reasoning_output(self, text: str) -> Dict[str, str]:
        """Parse reasoning output into thought, insight, and action."""
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
        """Format previous reasoning insights for context."""
        if not insights:
            return "No previous insights yet."
        
        formatted = []
        for i, insight in enumerate(insights, 1):
            formatted.append(f"Step {i}: {insight.get('insight', '')}")
        
        return "\n".join(formatted)

    def _generate_draft(self, reasoning_insights: List[Dict[str, str]], 
                       user_analysis: str, item_analysis: str) -> Dict[str, Any]:
        """Generate review draft based on reasoning insights."""
        
        # Compile all insights
        insights_summary = "\n".join([
            f"- {ins.get('insight', '')}" for ins in reasoning_insights
        ])
        
        prompt = f"""Generate a review based on the following reasoning insights.

REASONING INSIGHTS (instructions for you):
{insights_summary}

USER ANALYSIS:
{user_analysis}

ITEM ANALYSIS:
{item_analysis}

Your task: Write the actual review following the insights above.
- Match the predicted rating
- Cover the key topics mentioned in insights
- Use the tone and style specified in insights
- Keep length as suggested in insights (typically 2-4 sentences)

Output format:
stars: [1.0|2.0|3.0|4.0|5.0]
review: [your review text]
"""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages, temperature=0.3, max_tokens=400)
        
        return self._parse_review_output(response)

    def _simple_refinement(self, draft: Dict[str, Any], user_analysis: str) -> Dict[str, Any]:
        """Simple refinement (barebones - teammate's component)."""
        
        # For now, just check if review seems too short/long and adjust
        review_text = draft.get("review", "")
        
        # Simple length check
        if len(review_text.split()) < 10:
            # Too short, try to expand slightly
            prompt = f"""This review is too brief: "{review_text}"

User typically writes: {user_analysis[:200]}

Expand it slightly to be more complete while maintaining the same rating and tone. Output only the revised review text.
"""
            messages = [{"role": "user", "content": prompt}]
            revised = self.llm(messages=messages, temperature=0.2, max_tokens=200)
            draft["review"] = revised.strip()
        
        return draft

    def _parse_review_output(self, text: str) -> Dict[str, Any]:
        """Parse LLM output into stars and review."""
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
                review = text.strip()[:2048]
            
            # Trim if too long
            if len(review) > 2048:
                review = review[:2048]
        
        except Exception as e:
            logger.warning(f"Parse error: {e}")
            review = text[:2048] if text else ""
        
        return {"stars": stars, "review": review}

    def _sample_diverse_reviews(self, reviews: List[Dict], max_samples: int = 5) -> List[Dict]:
        """Sample reviews to get diverse ratings."""
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


if __name__ == "__main__":
    # Demo run
    sim = Simulator(data_dir="dataset", device="gpu", cache=True)
    import os
    here = os.path.dirname(__file__)
    task_dir = os.path.join(here, "track1", "amazon", "tasks")
    groundtruth_dir = os.path.join(here, "track1", "amazon", "groundtruth")
    sim.set_task_and_groundtruth(task_dir=task_dir, groundtruth_dir=groundtruth_dir)

    llm = OllamaLLM(model="mistral")
    sim.set_agent(ReasoningLoopAgent)
    sim.set_llm(llm)

    print("Running reasoning loop agent for task 0...")
    res = sim.run_single_task(task_index=0, wrap_llm_with_logger=True)
    
    output_data = {
        "output": res.get("output"),
        "llm_call_count": len(res.get("llm_calls", []))
    }

    
    print(json.dumps(output_data, indent=2))
    

    # Save just the output and LLM calls.
    output_data = {
        "output": res.get("output"),
        "llm_call_count": len(res.get("llm_calls", []))
    }
    with open("reasoning_loop_output.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("Saved detailed logs to reasoning_loop_run.json")
