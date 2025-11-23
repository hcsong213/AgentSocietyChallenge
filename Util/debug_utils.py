from typing import Dict, Any
import logging

logger = logging.getLogger("websocietysimulator")


class LoggingLLMWrapper:    
    def __init__(self, inner_llm):
        self._inner = inner_llm
        self.calls = []

    def __call__(
        self,
        messages,
        model=None,
        temperature=0.0,
        max_tokens=500,
        stop_strs=None,
        n=1
    ):
        self.calls.append({"request": messages})
        
        try:
            resp = self._inner(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_strs=stop_strs,
                n=n
            )
            self.calls.append({"response": resp})
            
            print("LLM request:\n", messages)
            print("LLM response:\n", resp)
            
            return resp
        except Exception as e:
            self.calls.append({"error": str(e)})
            raise

    def get_embedding_model(self):
        return getattr(self._inner, "get_embedding_model", lambda: None)()


def run_single_task(
    simulator,
    task_index: int = 0,
    wrap_llm_with_logger: bool = False
) -> Dict[str, Any]:
    """
    Run a single task immediately and return its result.
    
    Args:
        simulator: The Simulator instance to run the task on
        task_index: Index of the task to run
        wrap_llm_with_logger: Whether to wrap the LLM with logging
    
    Returns:
        Dict with keys: 'task', 'output', 'agent', and optionally 'llm_calls'
        when logging is enabled.
    """
    if not simulator.tasks:
        raise RuntimeError(
            "No tasks loaded. Call set_task_and_groundtruth() first."
        )

    if task_index < 0 or task_index >= len(simulator.tasks):
        raise IndexError("task_index out of range")

    if not simulator.agent_class:
        raise RuntimeError("Agent class is not set. Use set_agent() to set it.")

    task = simulator.tasks[task_index]

    llm_to_use = simulator.llm
    llm_calls = None
    
    if wrap_llm_with_logger:
        llm_to_use = LoggingLLMWrapper(simulator.llm)
        llm_calls = llm_to_use.calls

    if isinstance(llm_to_use, list):
        agent = simulator.agent_class(llm=llm_to_use[task_index % len(llm_to_use)])
    else:
        agent = simulator.agent_class(llm=llm_to_use)

    agent.set_interaction_tool(simulator.interaction_tool)
    agent.insert_task(task)

    try:
        output = agent.workflow()
    except Exception as e:
        logger.exception("Error running single task: %s", e)
        raise

    result = {"task": task.to_dict(), "output": output}
    
    if llm_calls is not None:
        result["llm_calls"] = llm_calls
        
    result["agent"] = agent
    
    print("Task:", task.to_dict())
    print("Output:", output)
    
    return result
