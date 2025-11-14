"""Advanced agent with ReAct-style plan-execute loop and iterative refinement.

Architecture:
  1) Plan-Execute Loop: Interleaved thinking and acting with step limit awareness
  2) Smart context retrieval: Select most relevant user/item reviews
  3) Iterative refinement: Generate draft → critique → revise
  
The agent uses a Thought-Action-Observation cycle with a hard step limit.
As the limit approaches, the agent is prompted to work faster and prioritize synthesis.
"""
from __future__ import annotations

import json
import re
import logging
from typing import Dict, Any, List, Optional

from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import OllamaLLM, LLMBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("react_planning_agent")


class ReActPlanningAgent(SimulationAgent):
    """Agent with ReAct-style plan-execute loop and iterative refinement."""

    def __init__(self, llm: LLMBase, max_steps: int = 8):
        super().__init__(llm=llm)
        self.max_steps = max_steps
        self.action_history = []  # Track actions to prevent loops

    def workflow(self) -> Dict[str, Any]:
        """Execute the ReAct plan-execute loop followed by iterative refinement."""
        if not self.interaction_tool:
            raise RuntimeError("interaction_tool is required")

        try:
            user_id = self.task.get("user_id")
            item_id = self.task.get("item_id")

            # === PHASE 1: Plan-Execute Loop ===
            logger.info("Phase 1: Starting ReAct plan-execute loop...")
            context = self._react_loop(user_id, item_id)
            
            # === PHASE 2: Iterative Refinement ===
            logger.info("Phase 2: Generating review with iterative refinement...")
            final_output = self._iterative_refinement(context)

            return final_output

        except Exception as e:
            logger.exception("ReActPlanningAgent workflow failed: %s", e)
            return {"stars": 0.0, "review": ""}

    def _react_loop(self, user_id: str, item_id: str) -> Dict[str, Any]:
        """Execute ReAct-style Thought-Action-Observation loop.
        
        The agent interleaves thinking and acting, gathering context until it has
        enough information or reaches the step limit.
        
        All data is pre-fetched; agent focuses on reasoning about it.
        
        Returns accumulated context for review generation.
        """
        # Pre-fetch all data so agent can focus on reasoning
        user_profile = self.interaction_tool.get_user(user_id=user_id)
        user_reviews = self.interaction_tool.get_reviews(user_id=user_id) or []
        item_info = self.interaction_tool.get_item(item_id=item_id)
        item_reviews = self.interaction_tool.get_reviews(item_id=item_id) or []
        
        context = {
            "user_profile": user_profile,
            "user_reviews": user_reviews,
            "user_analysis": "",
            "item_info": item_info,
            "item_reviews": item_reviews,
            "item_analysis": "",
            "reasoning_steps": [],  # Track all reasoning outputs
            "observations": []
        }
        
        logger.info(f"Pre-fetched context: {len(user_reviews)} user reviews, {len(item_reviews)} item reviews")
        
        self.action_history = []
        
        for step in range(self.max_steps):
            steps_remaining = self.max_steps - step
            
            # Generate thought and action
            thought_action = self._generate_thought_action(
                context, user_id, item_id, steps_remaining
            )
            
            logger.info(f"Step {step + 1}/{self.max_steps}: {thought_action.get('thought', '')[:100]}")
            logger.info(f"Action: {thought_action.get('action', '')}")
            
            # Check if agent signals completion
            if thought_action.get("action") == "DONE":
                logger.info("Agent signaled completion")
                break
            
            # Execute action and get observation
            observation = self._execute_action(
                thought_action.get("action", ""),
                thought_action.get("action_input", ""),
                context,
                user_id,
                item_id
            )
            
            # Record observation
            context["observations"].append({
                "step": step + 1,
                "thought": thought_action.get("thought", ""),
                "action": thought_action.get("action", ""),
                "observation": observation
            })
            
            logger.info(f"Observation: {observation[:200]}...")
            
            # Track action to prevent loops
            action_key = f"{thought_action.get('action', '')}:{thought_action.get('action_input', '')}"
            self.action_history.append(action_key)
        
        return context

    def _generate_thought_action(self, context: Dict[str, Any], user_id: str, 
                                 item_id: str, steps_remaining: int) -> Dict[str, str]:
        """Generate next thought and action based on current context.
        
        Uses ReAct prompting with step limit awareness to create urgency.
        """
        # Build context summary
        context_summary = self._build_context_summary(context)
        
        # Build urgency message based on remaining steps
        if steps_remaining <= 2:
            urgency = f"⚠️ URGENT: Only {steps_remaining} steps left! Prioritize final synthesis."
        elif steps_remaining <= 4:
            urgency = f"You have {steps_remaining} steps remaining. Start wrapping up soon."
        else:
            urgency = f"You have {steps_remaining} steps remaining."
        
        # Available actions with examples
        available_actions = """
Available Actions:
1. reason: Perform focused reasoning about the data. Break down your thinking into clear, systematic steps.
   - Use this to analyze user patterns, item characteristics, rating predictions, style matching, etc.
   - If the reasoning is complex, split it across multiple reason actions
   - Each reasoning step should build on previous observations

2. analyze_user_reviews: Use LLM to extract structured patterns from user's review history
   - Only use once, after initial reasoning about what patterns to look for

3. analyze_item_reviews: Use LLM to extract structured themes from item's existing reviews  
   - Only use once, after initial reasoning about what themes matter

4. DONE: Signal that you have completed sufficient reasoning to generate the review

Guidelines for 'reason' action:
- Break complex reasoning into smaller, focused steps
- Each step should address ONE specific aspect (e.g., "analyze user's rating tendency", "identify user's style preferences", "predict likely rating")
- Build on previous reasoning - reference what you learned in earlier steps
- Be systematic: don't jump around between topics randomly

Example reasoning progression:
  Step 1: reason → "Analyze user's rating distribution to understand if they rate harshly or generously"
  Step 2: reason → "Based on user's generous rating pattern, examine what aspects they value most"  
  Step 3: reason → "Compare item's strengths to user's valued aspects to predict compatibility"
  Step 4: analyze_user_reviews → Get structured LLM analysis
  Step 5: reason → "Synthesize findings to determine appropriate rating and review angle"
  Step 6: DONE
"""
        
        # Check what's already been done
        done_actions = []
        done_actions.append(f"✓ User profile available: {len(str(context.get('user_profile', {}))[:100])}... chars")
        done_actions.append(f"✓ User reviews available: {len(context['user_reviews'])} reviews")
        done_actions.append(f"✓ Item info available: {len(str(context.get('item_info', {}))[:100])}... chars")
        done_actions.append(f"✓ Item reviews available: {len(context['item_reviews'])} reviews")
        
        if context.get("user_analysis"):
            done_actions.append("✓ User analysis completed (via analyze_user_reviews)")
        if context.get("item_analysis"):
            done_actions.append("✓ Item analysis completed (via analyze_item_reviews)")
        if context.get("reasoning_steps"):
            done_actions.append(f"✓ Completed {len(context['reasoning_steps'])} reasoning steps")
        
        done_summary = "\n".join(done_actions)
        
        prompt = f"""You are generating a review for user_id={user_id} reviewing item_id={item_id}.

{urgency}

Available Data (already loaded):
{done_summary}

Recent observations and reasoning:
{context_summary}

{available_actions}

Your task: Perform systematic reasoning to understand how this user would review this item.

CRITICAL FORMATTING RULES:
- Output EXACTLY THREE LINES in this format:
  Line 1: Thought: [your reasoning]
  Line 2: Action: [ONE action only]
  Line 3: Action Input: [input or "none"]
- STOP IMMEDIATELY after Action Input line
- DO NOT write multiple Thought/Action pairs
- DO NOT write explanations after Action Input
- DO NOT write what you will do next
- Choose ONE action per step

STRATEGY:
- All data is already available - focus on REASONING about it
- Use 'reason' action for your own analysis and thinking
- Use 'analyze_user_reviews' or 'analyze_item_reviews' to get LLM-powered structured analysis (use sparingly, only once each)
- Break complex reasoning into multiple steps - don't try to solve everything in one step
- Build on previous reasoning steps - reference what you learned earlier
- When you have enough reasoning to generate an authentic review, choose DONE

CORRECT OUTPUT FORMAT (choose ONE):

Example 1 (reasoning step):
Thought: I need to understand if this user rates harshly or generously to calibrate my rating prediction.
Action: reason
Action Input: Analyze user's rating distribution across their review history to determine rating tendency

Example 2 (LLM analysis):
Thought: I've identified key patterns; now I need structured analysis of user's linguistic style.
Action: analyze_user_reviews
Action Input: none

Example 3 (completion):
Thought: I have sufficient reasoning about user patterns, item characteristics, and likely rating. Ready to generate review.
Action: DONE
Action Input: none

NOW OUTPUT YOUR SINGLE ACTION (3 lines only):
"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a reasoning agent that outputs EXACTLY 3 lines per step: Thought, Action, Action Input. Never output multiple actions. Stop immediately after Action Input."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = self.llm(
            messages=messages, 
            temperature=0.1, 
            max_tokens=200,  # Reduced to prevent long outputs
            stop_strs=["\n\n", "Observation:", "Thought:", "Step", "Next"]  # Stop if trying to continue
        )
        
        return self._parse_thought_action(response)

    def _parse_thought_action(self, text: str) -> Dict[str, str]:
        """Parse LLM output into thought, action, and action_input.
        
        Stops parsing after Action Input to ignore any extra text the LLM generates.
        """
        result = {"thought": "", "action": "", "action_input": ""}
        
        try:
            # Extract thought (everything between "Thought:" and "Action:")
            thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.IGNORECASE | re.DOTALL)
            if thought_match:
                result["thought"] = thought_match.group(1).strip()
            
            # Extract action (first word/phrase after "Action:")
            action_match = re.search(r"Action:\s*([^\n]+)", text, re.IGNORECASE)
            if action_match:
                action_text = action_match.group(1).strip()
                # Clean up: remove any trailing annotations or explanations
                # Take only the action name (first word or underscore-connected phrase)
                action_clean = re.match(r"^(\w+)", action_text)
                if action_clean:
                    result["action"] = action_clean.group(1)
                else:
                    result["action"] = action_text.split()[0] if action_text else ""
            
            # Extract action input (stop at newline to prevent capturing extra text)
            input_match = re.search(r"Action Input:\s*([^\n]+)", text, re.IGNORECASE)
            if input_match:
                action_input = input_match.group(1).strip()
                # Remove common annotations in parentheses
                action_input = re.sub(r'\s*\([^)]*\)\s*', '', action_input)
                # Clean up extra explanations after the value
                if '=' in action_input:
                    # If format is "key=value", take just the value
                    action_input = action_input.split('=')[-1].strip()
                result["action_input"] = action_input
        
        except Exception as e:
            logger.warning(f"Failed to parse thought-action: {e}")
        
        return result

    def _build_context_summary(self, context: Dict[str, Any]) -> str:
        """Build a concise summary of recent observations for the next step."""
        observations = context.get("observations", [])
        if not observations:
            return "No observations yet."
        
        # Show last 2-3 observations
        recent = observations[-3:]
        summary_lines = []
        for obs in recent:
            summary_lines.append(
                f"Step {obs['step']}: {obs['action']} → {obs['observation'][:150]}..."
            )
        
        return "\n".join(summary_lines)

    def _execute_action(self, action: str, action_input: str, context: Dict[str, Any],
                       user_id: str, item_id: str) -> str:
        """Execute the chosen action and return observation."""
        action_lower = action.lower()
        
        try:
            if action_lower == "reason" or "reason" in action_lower:
                # Agent performs its own reasoning about the data
                reasoning_focus = action_input if action_input and action_input.lower() != "none" else "general analysis"
                reasoning_result = self._perform_reasoning(reasoning_focus, context)
                context["reasoning_steps"].append({
                    "focus": reasoning_focus,
                    "result": reasoning_result
                })
                return f"Reasoning completed: {reasoning_result[:300]}..."
            
            elif "analyze_user" in action_lower or action == "analyze_user_reviews":
                if context.get("user_analysis"):
                    return "User analysis already completed."
                if not context.get("user_reviews"):
                    return "No user reviews available to analyze."
                context["user_analysis"] = self._analyze_user_reviews(context["user_reviews"])
                return f"User analysis completed: {context['user_analysis'][:200]}..."
            
            elif "analyze_item" in action_lower or action == "analyze_item_reviews":
                if context.get("item_analysis"):
                    return "Item analysis already completed."
                if not context.get("item_reviews"):
                    return "No item reviews available to analyze."
                context["item_analysis"] = self._analyze_item_reviews(context["item_reviews"])
                return f"Item analysis completed: {context['item_analysis'][:200]}..."
            
            elif action == "DONE" or action_lower == "done":
                return "Agent signaled completion."
            
            else:
                return f"Unknown action: {action}. Available actions: reason, analyze_user_reviews, analyze_item_reviews, DONE"
        
        except Exception as e:
            logger.exception(f"Error executing action {action}: {e}")
            return f"Error executing {action}: {str(e)}"

    def _perform_reasoning(self, focus: str, context: Dict[str, Any]) -> str:
        """Perform focused reasoning about the available data.
        
        This is where the agent does its own analysis and thinking.
        """
        # Build data summary for reasoning
        user_profile_str = json.dumps(context.get("user_profile", {}), ensure_ascii=False)[:500]
        user_reviews_sample = self._sample_diverse_reviews(context.get("user_reviews", []), max_samples=3)
        user_reviews_str = "\n".join([
            f"Rating: {r.get('stars', 'N/A')}, Text: {r.get('text', '')[:200]}"
            for r in user_reviews_sample
        ])
        
        item_info_str = json.dumps(context.get("item_info", {}), ensure_ascii=False)[:500]
        item_reviews_sample = self._sample_diverse_reviews(context.get("item_reviews", []), max_samples=3)
        item_reviews_str = "\n".join([
            f"Rating: {r.get('stars', 'N/A')}, Text: {r.get('text', '')[:200]}"
            for r in item_reviews_sample
        ])
        
        # Include previous reasoning steps for continuity
        previous_reasoning = ""
        if context.get("reasoning_steps"):
            recent_steps = context["reasoning_steps"][-2:]  # Last 2 steps
            previous_reasoning = "\n".join([
                f"Previous reasoning ({s['focus']}): {s['result'][:150]}..."
                for s in recent_steps
            ])
        
        prompt = f"""You are analyzing data to generate an authentic review prediction.

Focus for this reasoning step: {focus}

User Profile:
{user_profile_str}

Sample User Reviews:
{user_reviews_str}

Item Info:
{item_info_str}

Sample Item Reviews:
{item_reviews_str}

Previous Reasoning (if any):
{previous_reasoning if previous_reasoning else "This is your first reasoning step."}

Task: Perform focused analysis on "{focus}". Be specific and data-driven.
- Reference specific patterns you see in the data
- Draw concrete conclusions
- Keep it concise (2-4 sentences)
- Build on previous reasoning if applicable

Your reasoning:
"""
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_output = self.llm(messages=messages, temperature=0.1, max_tokens=300)
        
        return reasoning_output.strip()

    def _analyze_user_reviews(self, user_reviews: List[Dict]) -> str:
        """Analyze user's review history to extract patterns.
        
        Uses smart sampling to select most informative reviews.
        """
        if not user_reviews:
            return "No user review history available."
        
        # Sample reviews intelligently: get diverse ratings
        sampled = self._sample_diverse_reviews(user_reviews, max_samples=5)
        
        reviews_text = "\n\n".join([
            f"Rating: {r.get('stars', 'N/A')}\nReview: {r.get('text', '')[:300]}"
            for r in sampled
        ])
        
        prompt = f"""Analyze this user's review patterns and extract key characteristics.

User's review samples:
{reviews_text}

Provide a concise analysis covering:
1. Rating tendency (harsh/generous/moderate)
2. Review style (formal/casual, length, tone)
3. Key aspects they focus on (service, quality, value, etc.)
4. Linguistic patterns

Keep response under 150 words.
"""
        messages = [{"role": "user", "content": prompt}]
        return self.llm(messages=messages, temperature=0.0, max_tokens=250)

    def _analyze_item_reviews(self, item_reviews: List[Dict]) -> str:
        """Analyze existing reviews for this item to understand consensus.
        
        Identifies common themes and overall sentiment.
        """
        if not item_reviews:
            return "No existing reviews for this item."
        
        # Sample reviews
        sampled = self._sample_diverse_reviews(item_reviews, max_samples=5)
        
        reviews_text = "\n\n".join([
            f"Rating: {r.get('stars', 'N/A')}\nReview: {r.get('text', '')[:300]}"
            for r in sampled
        ])
        
        prompt = f"""Analyze the existing reviews for this item and identify key themes.

Item reviews:
{reviews_text}

Provide a concise summary covering:
1. Overall sentiment (positive/mixed/negative)
2. Most commonly mentioned aspects (pros and cons)
3. Average rating tendency
4. Notable patterns

Keep response under 150 words.
"""
        messages = [{"role": "user", "content": prompt}]
        return self.llm(messages=messages, temperature=0.0, max_tokens=250)

    def _sample_diverse_reviews(self, reviews: List[Dict], max_samples: int = 5) -> List[Dict]:
        """Sample reviews to get diverse ratings and perspectives."""
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

    def _iterative_refinement(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate review using iterative refinement: draft → critique → revise.
        
        Returns final output with stars and review.
        """
        # Convert context to text summaries
        user_summary = self._format_context_for_prompt(context, focus="user")
        item_summary = self._format_context_for_prompt(context, focus="item")
        
        # === Step 1: Generate initial draft ===
        draft = self._generate_draft(user_summary, item_summary)
        logger.info(f"Initial draft: {draft}")
        
        # === Step 2: Critique the draft ===
        critique = self._critique_draft(draft, user_summary, item_summary)
        logger.info(f"Critique: {critique}")
        
        # === Step 3: Revise based on critique ===
        final = self._revise_draft(draft, critique, user_summary, item_summary)
        logger.info(f"Final output: {final}")
        
        return final

    def _format_context_for_prompt(self, context: Dict[str, Any], focus: str = "both") -> str:
        """Format context dictionary into readable text for prompts."""
        parts = []
        
        if focus in ["user", "both"]:
            if context.get("user_profile"):
                parts.append(f"User Profile: {json.dumps(context['user_profile'], ensure_ascii=False)[:300]}")
            if context.get("user_analysis"):
                parts.append(f"User Analysis: {context['user_analysis']}")
        
        if focus in ["item", "both"]:
            if context.get("item_info"):
                parts.append(f"Item Info: {json.dumps(context['item_info'], ensure_ascii=False)[:300]}")
            if context.get("item_analysis"):
                parts.append(f"Item Analysis: {context['item_analysis']}")
        
        return "\n\n".join(parts)

    def _generate_draft(self, user_summary: str, item_summary: str) -> Dict[str, Any]:
        """Generate initial draft review."""
        prompt = f"""You are writing a review as this user for this item.

{user_summary}

{item_summary}

Generate a review that:
1. Matches the user's typical rating tendency and style
2. Reflects the item's characteristics and existing review patterns
3. Is authentic and specific (2-4 sentences)

Output format:
stars: [1.0|2.0|3.0|4.0|5.0]
review: [your review text]
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages, temperature=0.3, max_tokens=400)
        
        return self._parse_review_output(response)

    def _critique_draft(self, draft: Dict[str, Any], user_summary: str, item_summary: str) -> str:
        """Critique the draft review for consistency and quality."""
        prompt = f"""You are a review quality critic. Evaluate this draft review for consistency and authenticity.

Draft Review:
Stars: {draft.get('stars', 0.0)}
Review: {draft.get('review', '')}

Context:
{user_summary}

{item_summary}

Critique the draft on:
1. Rating consistency with user's typical patterns
2. Style matching (tone, length, formality)
3. Specificity and authenticity
4. Alignment with item characteristics

Provide 2-3 specific improvement suggestions. Be concise (under 100 words).
"""
        messages = [{"role": "user", "content": prompt}]
        return self.llm(messages=messages, temperature=0.0, max_tokens=200)

    def _revise_draft(self, draft: Dict[str, Any], critique: str, 
                     user_summary: str, item_summary: str) -> Dict[str, Any]:
        """Revise the draft based on critique."""
        prompt = f"""Revise the draft review based on the critique.

Original Draft:
Stars: {draft.get('stars', 0.0)}
Review: {draft.get('review', '')}

Critique:
{critique}

Context:
{user_summary}
{item_summary}

Produce an improved version addressing the critique points.

Output format:
stars: [1.0|2.0|3.0|4.0|5.0]
review: [revised review text]
"""
        messages = [{"role": "user", "content": prompt}]
        response = self.llm(messages=messages, temperature=0.1, max_tokens=400)
        
        return self._parse_review_output(response)

    def _parse_review_output(self, text: str) -> Dict[str, Any]:
        """Parse LLM output into structured review format."""
        stars = 0.0
        review = ""
        
        try:
            # Parse stars (handle both "stars: 5.0" and "stars: [5.0]")
            m = re.search(r"stars\s*[:=]\s*\[?\s*([0-9]+(?:\.[0-9]+)?)\s*\]?", text, flags=re.IGNORECASE)
            if m:
                stars = float(m.group(1))
                # Clamp to valid range
                stars = max(1.0, min(5.0, stars))
            
            # Parse review text
            m2 = re.search(r"review\s*[:=]\s*(.+?)(?=\n\n|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
            if m2:
                review = m2.group(1).strip()
            else:
                # Fallback: try line-by-line
                for line in text.split("\n"):
                    if line.lower().startswith("review:"):
                        review = line.split(":", 1)[1].strip()
                        break
            
            # Ensure review isn't too long
            if len(review) > 512:
                review = review[:512]
            
            # Final fallback
            if not review:
                review = text[:512]
        
        except Exception as e:
            logger.warning(f"Parse error: {e}")
            review = text[:512] if text else ""
        
        return {"stars": stars, "review": review}


if __name__ == "__main__":
    # Demo: run with Simulator.run_single_task
    sim = Simulator(data_dir="dataset", device="gpu", cache=True)
    import os
    here = os.path.dirname(__file__)
    task_dir = os.path.join(here, "track1", "goodreads", "tasks")
    groundtruth_dir = os.path.join(here, "track1", "goodreads", "groundtruth")
    sim.set_task_and_groundtruth(task_dir=task_dir, groundtruth_dir=groundtruth_dir)

    llm = OllamaLLM(model="mistral")
    sim.set_agent(ReActPlanningAgent)
    sim.set_llm(llm)

    print("Running ReAct planning agent for task 0 (with LLM logging)...")
    res = sim.run_single_task(task_index=1, wrap_llm_with_logger=True)
    
    output_data = {
        "task": res.get("task"),
        "output": res.get("output"),
        "llm_call_count": len(res.get("llm_calls", [])),
        "steps_taken": len([obs for obs in res.get("output", {}).get("observations", []) if obs])
    }
    print(json.dumps(output_data, indent=2))
    
    # Optionally save detailed logs
    with open("react_planning_run.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Saved detailed logs to react_planning_run.json")
