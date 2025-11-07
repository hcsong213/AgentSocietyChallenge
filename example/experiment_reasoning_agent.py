"""Example test agent implementing a 3-step reasoning workflow.

Steps:
  1) Fetch user profile & history, item info, and similar reviews via interaction tool.
  2) Call LLM to extract key characteristics of the user from their past reviews.
  3) Call LLM to generate the final review using characteristics + product info + similar reviews.

Run as a script to exercise the agent with the local `Simulator.run_single_task` helper.
"""
from __future__ import annotations

import json
import re
import logging
from typing import Dict, Any

from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import OllamaLLM, LLMBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_reasoning_agent")


class TestReasoningAgent(SimulationAgent):
    """A lightweight test agent that demonstrates stepwise reasoning.

    The agent assumes an `interaction_tool` is set (via `Simulator`) and that
    the task is available in `self.task` (Simulator will call insert_task).
    """

    def __init__(self, llm: LLMBase):
        super().__init__(llm=llm)

    def workflow(self) -> Dict[str, Any]:
        """Perform the 3-step workflow and return the generated review.

        Returns a dict with keys: 'stars', 'review', 'characteristics'.
        """
        if not self.interaction_tool:
            raise RuntimeError("interaction_tool is required for TestReasoningAgent")

        try:
            user_id = self.task.get("user_id")
            item_id = self.task.get("item_id")

            # Step 1: fetch context
            user_profile = self.interaction_tool.get_user(user_id=user_id)
            user_reviews = self.interaction_tool.get_reviews(user_id=user_id)
            item_info = self.interaction_tool.get_item(item_id=item_id)
            item_reviews = self.interaction_tool.get_reviews(item_id=item_id)

            # Convert to compact strings for LLM prompts (keep lengths reasonable)
            user_reviews_snippet = "\n".join([r.get("text", "")[:400] for r in (user_reviews or [])][:5])
            item_reviews_snippet = "\n".join([r.get("text", "")[:400] for r in (item_reviews or [])][:5])
            user_profile_text = json.dumps(user_profile or {}, ensure_ascii=False)
            item_info_text = json.dumps(item_info or {}, ensure_ascii=False)

            # Step 2: ask LLM to extract key user characteristics
            extract_prompt = (
                "You are an assistant that extracts concise user review characteristics.\n"
                "Given the following user profile and recent review snippets, list 3-5 characteristics about the user's style, typical topics, sentiment tendency, and rating habits."
                "\n\nUser profile:\n{profile}\n\nUser reviews (snippets):\n{reviews}\n\n"
            ).format(profile=user_profile_text, reviews=user_reviews_snippet)

            messages = [
                {"role": "system", "content": "You summarize review authors' characteristics."},
                {"role": "user", "content": extract_prompt},
            ]

            characteristics_text = self.llm(messages=messages, temperature=0.0, max_tokens=300)

            # Step 3: generate the actual review using characteristics + item info + similar reviews
            generate_prompt = (
                "You are a Yelp user writing a review. Use the following user characteristics to write a short, personal review for the business.\n\n"
                "User characteristics:\n{chars}\n\nItem info:\n{item}\n\nSimilar reviews for this business (snippets):\n{irevs}\n\n"
                "Instructions:\n- Produce output exactly in the format:\n  stars: [1.0|2.0|3.0|4.0|5.0]\n  review: [2-4 sentences]\n- Keep the review consistent with the user's style.\n"
            ).format(chars=characteristics_text, item=item_info_text, irevs=item_reviews_snippet)

            messages2 = [
                {"role": "system", "content": "You will produce a rating and a short review consistent with the user's voice."},
                {"role": "user", "content": generate_prompt},
            ]

            final_text = self.llm(messages=messages2, temperature=0.0, max_tokens=400)

            # Parse simple structured response (robust to formats like 'stars: 5.0' or 'stars: [5.0]')
            stars = 0.0
            review_out = ""
            try:
                # Try a regex over the whole text to find a numeric stars value, allowing optional brackets
                m = re.search(r"stars\s*[:=]\s*\[?\s*([0-9]+(?:\.[0-9]+)?)\s*\]?", final_text, flags=re.IGNORECASE)
                if m:
                    try:
                        stars = float(m.group(1))
                    except Exception:
                        stars = 0.0

                # Find review line via regex (captures rest of line after 'review:')
                m2 = re.search(r"review\s*[:=]\s*(.+)", final_text, flags=re.IGNORECASE)
                if m2:
                    review_out = m2.group(1).strip()

                # Fallback: if review not found, try per-line parsing
                if not review_out:
                    lines = [ln.strip() for ln in final_text.split("\n") if ln.strip()]
                    for ln in lines:
                        if ln.lower().startswith("review:"):
                            review_out = ln.split(":", 1)[1].strip()
                            break

                # Final fallback: use full text clipped
                if not review_out:
                    review_out = final_text.strip()[:512]
            except Exception:
                review_out = final_text.strip()[:512]

            return {"stars": stars, "review": review_out, "characteristics": characteristics_text}

        except Exception as e:
            logger.exception("TestReasoningAgent workflow failed: %s", e)
            return {"stars": 0.0, "review": "", "characteristics": ""}


if __name__ == "__main__":
    # Demo: run with Simulator.run_single_task and LLM logging
    sim = Simulator(data_dir="dataset", device="gpu", cache=True)
    import os
    here = os.path.dirname(__file__)
    task_dir = os.path.join(here, "track1", "amazon", "tasks")
    groundtruth_dir = os.path.join(here, "track1", "amazon", "groundtruth")
    sim.set_task_and_groundtruth(task_dir=task_dir, groundtruth_dir=groundtruth_dir)

    # Replace model name with your local model if needed
    llm = OllamaLLM(model="mistral")
    sim.set_agent(TestReasoningAgent)
    sim.set_llm(llm)

    print("Running test reasoning agent for task 0 (with LLM logging)...")
    res = sim.run_single_task(task_index=0, wrap_llm_with_logger=True)
    print(json.dumps({"task": res.get("task"), "output": res.get("output")}, indent=2))
