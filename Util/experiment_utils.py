"""Experiment utilities for comprehensive agent evaluation across datasets.

This module provides functions for:
1. Multi-dataset evaluation (Yelp, Amazon, Goodreads)
2. Agent comparison experiments
3. LLM comparison experiments
4. Ablation studies
"""
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Type
from pathlib import Path

# Handle imports for both module usage and direct execution
import sys
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import *
from Util.ollama_llm import OllamaLLM

logger = logging.getLogger(__name__)


def evaluate_across_datasets(
    simulator: Simulator,
    agent_class: Type[SimulationAgent],
    llm: LLMBase,
    datasets: List[str] = ["yelp", "amazon", "goodreads"],
    number_of_tasks: int = 75,
    agent_kwargs: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Core evaluation function across multiple datasets.
    
    Args:
        simulator: Simulator instance (will be reconfigured for each dataset)
        agent_class: Agent class to evaluate
        llm: LLM instance to use
        datasets: List of dataset names (default: ["yelp", "amazon", "goodreads"])
        number_of_tasks: Number of tasks per dataset (default: 75)
        agent_kwargs: Optional kwargs to pass to agent constructor
    
    Returns:
        Dict containing results per dataset and aggregated metrics
    """
    agent_kwargs = agent_kwargs or {}
    results = {
        "agent": agent_class.__name__,
        "llm_model": getattr(llm, 'model', 'unknown'),
        "datasets": {},
        "aggregated_metrics": {}
    }
    
    # Metrics to aggregate
    metric_sums = {
        "preference_estimation": 0.0,
        "review_generation": 0.0,
        "overall_quality": 0.0
    }
    metric_counts = {key: 0 for key in metric_sums.keys()}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset.upper()} dataset...")
        print(f"{'='*60}")
        
        # Configure simulator for this dataset
        task_dir = os.path.join("example", "track1", dataset, "tasks")
        groundtruth_dir = os.path.join("example", "track1", dataset, "groundtruth")
        
        simulator.set_task_and_groundtruth(
            task_dir=task_dir,
            groundtruth_dir=groundtruth_dir
        )
        
        # Set agent with optional kwargs
        if agent_kwargs:
            # Create a wrapper that applies kwargs
            class ConfiguredAgent(agent_class):
                def __init__(self, llm):
                    super().__init__(llm=llm, **agent_kwargs)
            simulator.set_agent(ConfiguredAgent)
        else:
            simulator.set_agent(agent_class)
        
        simulator.set_llm(llm)
        
        # Run evaluation
        print(f"Running {number_of_tasks} tasks on {dataset}...")
        outputs = simulator.run_simulation(number_of_tasks=number_of_tasks)
        evaluation_results = simulator.evaluate()
        
        # Store results
        results["datasets"][dataset] = {
            "metrics": evaluation_results.get("metrics", {}),
            "data_info": evaluation_results.get("data_info", {})
        }
        
        # Aggregate metrics
        if "metrics" in evaluation_results:
            for key in metric_sums.keys():
                if key in evaluation_results["metrics"]:
                    metric_sums[key] += evaluation_results["metrics"][key]
                    metric_counts[key] += 1
        
        print(f"\n{dataset.upper()} Results:")
        if "metrics" in evaluation_results:
            for key, value in evaluation_results["metrics"].items():
                print(f"  {key}: {value:.4f}")
    
    # Calculate aggregated metrics
    for key in metric_sums.keys():
        if metric_counts[key] > 0:
            results["aggregated_metrics"][key] = metric_sums[key] / metric_counts[key]
    
    print(f"\n{'='*60}")
    print("AGGREGATED METRICS ACROSS ALL DATASETS:")
    print(f"{'='*60}")
    for key, value in results["aggregated_metrics"].items():
        print(f"  {key}: {value:.4f}")
    
    return results


def experiment_agent_comparison(
    agent_classes: List[Type[SimulationAgent]],
    llm_model: str = "mistral",
    datasets: List[str] = ["yelp", "amazon", "goodreads"],
    number_of_tasks: int = 75,
    output_base_dir: str = "./Outputs"
) -> Dict[str, Any]:
    """
    Compare different agent architectures using the same LLM.
    
    Args:
        agent_classes: List of agent classes to compare
        llm_model: Ollama model to use (default: "mistral")
        datasets: List of dataset names
        number_of_tasks: Number of tasks per dataset
        output_base_dir: Base directory for outputs
    
    Returns:
        Dict containing comparison results
    """
    print("="*60)
    print("AGENT COMPARISON EXPERIMENT")
    print("="*60)
    print(f"LLM Model: {llm_model}")
    print(f"Agents: {[cls.__name__ for cls in agent_classes]}")
    print(f"Datasets: {datasets}")
    print(f"Tasks per dataset: {number_of_tasks}")
    
    # Setup
    simulator = Simulator(data_dir="dataset", device="gpu", cache=True)
    llm = OllamaLLM(model=llm_model)
    
    comparison_results = {
        "experiment_type": "agent_comparison",
        "llm_model": llm_model,
        "datasets": datasets,
        "number_of_tasks_per_dataset": number_of_tasks,
        "agents": {}
    }
    
    # Evaluate each agent
    for agent_class in agent_classes:
        print(f"\n{'='*60}")
        print(f"Evaluating {agent_class.__name__}")
        print(f"{'='*60}")
        
        results = evaluate_across_datasets(
            simulator=simulator,
            agent_class=agent_class,
            llm=llm,
            datasets=datasets,
            number_of_tasks=number_of_tasks
        )
        
        comparison_results["agents"][agent_class.__name__] = results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / "AgentComparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"agent_comparison_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("AGENT COMPARISON COMPLETE")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print summary comparison
    print("\nSUMMARY COMPARISON (Aggregated Metrics):")
    print("-" * 60)
    for agent_name, agent_results in comparison_results["agents"].items():
        print(f"\n{agent_name}:")
        for metric, value in agent_results["aggregated_metrics"].items():
            print(f"  {metric}: {value:.4f}")
    
    return comparison_results


def experiment_llm_comparison(
    agent_class: Type[SimulationAgent],
    llm_configs: List[Dict[str, Any]] = None,
    datasets: List[str] = ["yelp", "amazon", "goodreads"],
    number_of_tasks: int = 75,
    output_base_dir: str = "./Outputs"
) -> Dict[str, Any]:
    """
    Compare different LLMs using the same agent architecture.
    
    Args:
        agent_class: Agent class to use for comparison
        llm_configs: List of LLM configurations. Each config should have:
                    - 'type': 'ollama', 'openai', or 'google'
                    - 'model': model name
                    - 'api_key': (optional, for OpenAI)
                    If None, defaults to common Ollama models + GPT-3.5-turbo
        datasets: List of dataset names
        number_of_tasks: Number of tasks per dataset
        output_base_dir: Base directory for outputs
    
    Returns:
        Dict containing comparison results
    """
    # Default LLM configurations if none provided
    if llm_configs is None:
        llm_configs = [
            {"type": "ollama", "model": "qwen2.5:3b"},
            {"type": "ollama", "model": "mistral"},
            {"type": "ollama", "model": "llama3.1:8b"},
            {"type": "google", "model": "gemini-2.5-flash"}
        ]

    print("="*60)
    print("LLM COMPARISON EXPERIMENT")
    print("="*60)
    print(f"Agent: {agent_class.__name__}")
    print(f"LLMs: {[cfg['model'] for cfg in llm_configs]}")
    print(f"Datasets: {datasets}")
    print(f"Tasks per dataset: {number_of_tasks}")

    # Setup
    simulator = Simulator(data_dir="dataset", device="gpu", cache=True)

    comparison_results = {
        "experiment_type": "llm_comparison",
        "agent": agent_class.__name__,
        "datasets": datasets,
        "number_of_tasks_per_dataset": number_of_tasks,
        "llms": {}
    }

    # Evaluate each LLM
    for llm_config in llm_configs:
        llm_type = llm_config["type"]
        model_name = llm_config["model"]

        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} ({llm_type})")
        print(f"{'='*60}")

        # Initialize LLM
        try:
            if llm_type == "ollama":
                llm = OllamaLLM(model=model_name)
            elif llm_type == "openai":
                api_key = llm_config.get("api_key") or os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    print(f"WARNING: No OpenAI API key found. Skipping {model_name}")
                    continue
                llm = OpenAILLM(api_key=api_key, model=model_name)
            elif llm_type == "google":
                api_key = (
                    llm_config.get("api_key")
                    or os.environ.get("GEMINI_API_KEY")
                    or os.environ.get("GOOGLE_API_KEY")
                )
                if not api_key:
                    print(f"WARNING: No Google API key found. Skipping {model_name}")
                    continue
                llm = GoogleLLM(api_key=api_key, model=model_name)
            else:
                print(f"WARNING: Unknown LLM type '{llm_type}'. Skipping {model_name}")
                continue

            results = evaluate_across_datasets(
                simulator=simulator,
                agent_class=agent_class,
                llm=llm,
                datasets=datasets,
                number_of_tasks=number_of_tasks
            )

            comparison_results["llms"][f"{model_name}_{llm_type}"] = results

        except Exception as e:
            print(f"ERROR evaluating {model_name}: {e}")
            logger.exception(f"Failed to evaluate {model_name}")
            continue

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / "LLMComparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"llm_comparison_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("LLM COMPARISON COMPLETE")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # Print summary comparison
    print("\nSUMMARY COMPARISON (Aggregated Metrics):")
    print("-" * 60)
    for llm_name, llm_results in comparison_results["llms"].items():
        print(f"\n{llm_name}:")
        for metric, value in llm_results["aggregated_metrics"].items():
            print(f"  {metric}: {value:.4f}")

    return comparison_results


def experiment_ablation_study(
    agent_class: Type[SimulationAgent],
    llm_model: str = "mistral",
    datasets: List[str] = ["yelp", "amazon", "goodreads"],
    number_of_tasks: int = 75,
    output_base_dir: str = "./Outputs"
) -> Dict[str, Any]:
    """
    Perform ablation study on agent components.
    
    Tests four configurations:
    1. Full pipeline (profiling + refinement)
    2. No profiling, with refinement
    3. With profiling, no refinement
    4. Minimal (no profiling, no refinement)
    
    Args:
        agent_class: Agent class to study (must support enable_profiling and enable_refinement)
        llm_model: Ollama model to use (default: "mistral")
        datasets: List of dataset names
        number_of_tasks: Number of tasks per dataset
        output_base_dir: Base directory for outputs
    
    Returns:
        Dict containing ablation study results
    """
    print("="*60)
    print("ABLATION STUDY EXPERIMENT")
    print("="*60)
    print(f"Agent: {agent_class.__name__}")
    print(f"LLM Model: {llm_model}")
    print(f"Datasets: {datasets}")
    print(f"Tasks per dataset: {number_of_tasks}")
    
    # Setup
    simulator = Simulator(data_dir="dataset", device="gpu", cache=True)
    llm = OllamaLLM(model=llm_model)
    
    ablation_configs = [
        {
            "name": "full_pipeline",
            "description": "Profiling + Refinement",
            "kwargs": {"enable_profiling": True, "enable_refinement": True}
        },
        {
            "name": "no_profiling",
            "description": "No Profiling, With Refinement",
            "kwargs": {"enable_profiling": False, "enable_refinement": True}
        },
        {
            "name": "no_refinement",
            "description": "With Profiling, No Refinement",
            "kwargs": {"enable_profiling": True, "enable_refinement": False}
        },
        {
            "name": "minimal",
            "description": "No Profiling, No Refinement",
            "kwargs": {"enable_profiling": False, "enable_refinement": False}
        }
    ]
    
    comparison_results = {
        "experiment_type": "ablation_study",
        "agent": agent_class.__name__,
        "llm_model": llm_model,
        "datasets": datasets,
        "number_of_tasks_per_dataset": number_of_tasks,
        "configurations": {}
    }
    
    # Evaluate each configuration
    for config in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {config['description']}")
        print(f"{'='*60}")
        
        try:
            results = evaluate_across_datasets(
                simulator=simulator,
                agent_class=agent_class,
                llm=llm,
                datasets=datasets,
                number_of_tasks=number_of_tasks,
                agent_kwargs=config["kwargs"]
            )
            
            comparison_results["configurations"][config["name"]] = {
                "description": config["description"],
                "kwargs": config["kwargs"],
                "results": results
            }
            
        except Exception as e:
            print(f"ERROR evaluating configuration {config['name']}: {e}")
            logger.exception(f"Failed to evaluate {config['name']}")
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base_dir) / "AblationComparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"ablation_study_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # Print summary comparison
    print("\nSUMMARY COMPARISON (Aggregated Metrics):")
    print("-" * 60)
    for config_name, config_data in comparison_results["configurations"].items():
        print(f"\n{config_data['description']}:")
        for metric, value in config_data["results"]["aggregated_metrics"].items():
            print(f"  {metric}: {value:.4f}")
    
    return comparison_results


if __name__ == "__main__":
    from Agents.baseline_agent import BaselineAgent
    from Agents.structured_profile_agent import StructuredProfileAgent
    from Agents.reasoning_loop_agent import ReasoningLoopAgent
    from Agents.ensemble_reviews_agent import EnsembleReviewsAgent
    
    # # Example: Run agent comparison
    # print("Running Agent Comparison Experiment...")
    # agent_comparison_results = experiment_agent_comparison(
    #     agent_classes=[BaselineAgent, StructuredProfileAgent, ReasoningLoopAgent, EnsembleReviewsAgent],
    #     llm_model="mistral",
    #     datasets=["yelp", "amazon", "goodreads"],  # Start with one dataset for testing
    #     number_of_tasks=1  # Small number for testing
    # )
    
    # Example: Run LLM comparison
    print("\nRunning LLM Comparison Experiment...")
    llm_comparison_results = experiment_llm_comparison(
        agent_class=StructuredProfileAgent,
        datasets=["yelp"],
        number_of_tasks=1,
        llm_configs = [
            # {"type": "ollama", "model": "qwen2.5:3b"},
            # {"type": "ollama", "model": "mistral"},
            # {"type": "ollama", "model": "llama3.1:8b"},
            {"type": "google", "model": "gemini-2.5-flash"}
        ]
    )
    
    # # Example: Run ablation study
    # print("\nRunning Ablation Study...")
    # ablation_results = experiment_ablation_study(
    #     agent_class=StructuredProfileAgent,
    #     llm_model="mistral",
    #     datasets=["yelp"],
    #     number_of_tasks=10
    # )
