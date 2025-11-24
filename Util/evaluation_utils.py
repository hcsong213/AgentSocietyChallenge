"""Evaluation and debugging utilities for agent testing.

This module provides functions for:
1. Single task debugging with LLM logging
2. Full simulation evaluation with organized output
"""
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Handle imports for both module usage and direct execution
if __name__ == "__main__":
    # Add parent directory to path when running directly
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from Util.format_llm_logs import format_llm_logs
    from Util.debug_utils import run_single_task
else:
    # Normal imports when used as a module - import directly from files
    from Util.format_llm_logs import format_llm_logs
    from Util.debug_utils import run_single_task

logger = logging.getLogger(__name__)


def debug_single_task(
    simulator,
    agent_name: str,
    task_index: int = 0,
    task_set: str = "yelp",
    wrap_llm_with_logger: bool = True,
    output_base_dir: str = "./Outputs"
) -> Dict[str, Any]:
    """
    Run a single task for debugging and save outputs with timestamps.
    
    Args:
        simulator: The Simulator instance (already configured with agent and LLM)
        agent_name: Name of the agent (used for subfolder organization)
        task_index: Index of the task to run (default: 0)
        task_set: Name of the task set (e.g., "yelp", "goodreads")
        wrap_llm_with_logger: Whether to log LLM calls (default: True)
        output_base_dir: Base directory for outputs (default: "./Outputs")
    
    Returns:
        Dict containing the task result and output paths
    """
    print(f"Running {agent_name} for task {task_index}...")
    
    # Run the single task
    res = run_single_task(simulator, task_index, wrap_llm_with_logger)
    
    # Prepare output directory with agent name subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_output_dir = Path(output_base_dir) / agent_name
    agent_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare structured log data
    output_data = {
        "task_index": task_index,
        "task_set": task_set,
        "timestamp": timestamp,
        "output": res.get("output"),
        "llm_calls": res.get("llm_calls", [])
    }
    
    # Print summary
    print(json.dumps({
        "output_present": bool(output_data["output"]),
        "llm_call_count": len(output_data["llm_calls"])
    }, indent=2))
    
    # Save JSON log with timestamp
    log_json_path = agent_output_dir / f"debug_task{task_index}_{timestamp}.json"
    with open(log_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed logs to {log_json_path}")
    
    # Create human-readable formatted text file
    formatted_txt_path = agent_output_dir / f"debug_task{task_index}_{timestamp}.txt"
    try:
        format_llm_logs(str(log_json_path), str(formatted_txt_path))
        print(f"Saved formatted LLM logs to {formatted_txt_path}")
    except Exception as e:
        logger.exception("Failed to format LLM logs: %s", e)
    
    return {
        "result": res,
        "json_path": str(log_json_path),
        "txt_path": str(formatted_txt_path)
    }


def run_evaluation(
    simulator,
    agent_name: str,
    task_set: str = "yelp",
    number_of_tasks: int = 50,
    output_base_dir: str = "./Outputs"
) -> Dict[str, Any]:
    """
    Run full simulation evaluation and save results with timestamps.
    
    Args:
        simulator: The Simulator instance (already configured with agent and LLM)
        agent_name: Name of the agent (used for subfolder organization)
        task_set: Name of the task set (e.g., "yelp", "goodreads")
        number_of_tasks: Number of tasks to run (default: 50)
        output_base_dir: Base directory for outputs (default: "./Outputs")
    
    Returns:
        Dict containing evaluation results and output path
    """
    print(f"Running {agent_name} evaluation on {number_of_tasks} tasks from {task_set}...")
    
    # Run simulation
    outputs = simulator.run_simulation(number_of_tasks=number_of_tasks)
    
    # Evaluate
    evaluation_results = simulator.evaluate()
    
    # Prepare output directory with agent name subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_output_dir = Path(output_base_dir) / agent_name
    agent_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    evaluation_results["metadata"] = {
        "agent_name": agent_name,
        "task_set": task_set,
        "number_of_tasks": number_of_tasks,
        "timestamp": timestamp
    }
    
    # Save evaluation results with timestamp
    eval_path = agent_output_dir / f"evaluation_{task_set}_{timestamp}.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=4, ensure_ascii=False)
    
    print(f"Saved evaluation results to {eval_path}")
    print(f"\nEvaluation Summary:")
    print(f"  Agent: {agent_name}")
    print(f"  Task Set: {task_set}")
    print(f"  Tasks: {number_of_tasks}")
    
    # Print key metrics if available
    if "metrics" in evaluation_results:
        print(f"\nMetrics:")
        for key, value in evaluation_results["metrics"].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    return {
        "evaluation_results": evaluation_results,
        "output_path": str(eval_path),
        "outputs": outputs
    }


if __name__ == "__main__":
    from websocietysimulator import Simulator
    from Agents.structured_profile_agent import StructuredProfileAgent
    from Agents.baseline_agent import BaselineAgent
    from Agents.ensemble_reviews_agent import EnsembleReviewsAgent
    from Util.ollama_llm import OllamaLLM
    
    # Configuration
    DEBUG_MODE = False  # Set to True for single task debugging, False for full evaluation
    TASK_INDEX = 4  # Task index for debugging mode
    TASK_SET = "yelp"  # "goodreads" or "yelp"
    NUMBER_OF_TASKS = 20  # Number of tasks for evaluation mode
    MODEL = "mistral"  # Ollama model to use
    AGENT = StructuredProfileAgent
    
    print("=" * 60)
    print(f"{AGENT.__name__.upper()} EVALUATION")
    print("=" * 60)
    
    # Setup simulator
    sim = Simulator(data_dir="dataset", device="gpu", cache=True)
    
    task_dir = os.path.join("example", "track1", TASK_SET, "tasks")
    groundtruth_dir = os.path.join("example", "track1", TASK_SET, "groundtruth")
    sim.set_task_and_groundtruth(task_dir=task_dir, groundtruth_dir=groundtruth_dir)
    
    # Setup LLM and agentSTRUCTUre
    llm = OllamaLLM(model=MODEL)
    sim.set_agent(AGENT)
    sim.set_llm(llm)
    
    print(f"\nConfiguration:")
    print(f"  Mode: {'DEBUG (single task)' if DEBUG_MODE else 'EVALUATION (full run)'}")
    print(f"  Task Set: {TASK_SET}")
    print(f"  Model: {MODEL}")
    
    if DEBUG_MODE:
        print(f"  Task Index: {TASK_INDEX}")
        print("\n" + "=" * 60)
        result = debug_single_task(
            simulator=sim,
            agent_name=AGENT.__name__,
            task_index=TASK_INDEX,
            task_set=TASK_SET,
            wrap_llm_with_logger=True
        )
        print("\n" + "=" * 60)
        print("DEBUG COMPLETE")
        print(f"JSON output: {result['json_path']}")
        print(f"TXT output: {result['txt_path']}")
    else:
        print(f"  Number of Tasks: {NUMBER_OF_TASKS}")
        print("\n" + "=" * 60)
        result = run_evaluation(
            simulator=sim,
            agent_name=AGENT.__name__,
            task_set=TASK_SET,
            number_of_tasks=NUMBER_OF_TASKS
        )
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print(f"Results saved to: {result['output_path']}")
        print("\nFull Results:")
        print(json.dumps(result['evaluation_results'], indent=2))
