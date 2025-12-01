# Group Documentation

## Prerequisites

**IMPORTANT**: Before proceeding, ensure you have completed the original project setup:

1. Followed all setup instructions from the WWW'25 AgentSociety Challenge starter code
2. Installed all prerequisite packages specified in `pyproject.toml`
3. Downloaded and properly installed the dataset in the `dataset/` directory
4. Verified the base environment works with the provided baseline agents

This documentation assumes you have a **fully working base environment** and describes only the additional components we've added.

---

## Environment Setup

### Additional Packages Installed

To support our implementation, the following packages were installed on top of the base environment:

#### 1. Ollama Support
```bash
pip install ollama
```
- **Package**: `ollama==0.6.0`
- **Purpose**: Official Ollama Python client for running local LLMs

#### 2. Google Gemini Support
```bash
pip install google-genai langchain-google-genai
```
- **Package**: `google-genai==1.52.0`
- **Purpose**: Google's generative AI SDK for Gemini models

- **Package**: `langchain-google-genai==2.1.12`
- **Purpose**: LangChain integration for Google Gemini
- **Note**: Version 2.1.12 is used (not 3.x) to maintain compatibility with `langchain-core` 0.3.x from the starter code

### Installation Instructions

1. **Set up base environment** (from starter code):
   ```bash
   # Create conda environment
   conda create -n websocietysimulator python=3.11
   conda activate websocietysimulator
   
   # Install base dependencies
   pip install .
   ```

2. **Install additional packages**:
   ```bash
   # Install Ollama support
   pip install ollama
   
   # Install Google Gemini support (with compatible versions)
   pip install google-genai "langchain-google-genai>=2.0,<3.0"
   ```

### Package Compatibility Notes

- **langchain-google-genai**: Must use version 2.x (not 3.x) to avoid conflicts with `langchain-core` 0.3.x
- **langchain-core**: Maintained at 0.3.x as specified in the original `pyproject.toml`
- All transitive dependencies are automatically installed by pip

---

## Project Structure

### Agents Directory (`Agents/`)

#### `agent_utils.py`
Shared utility functions used across all agent implementations. Contains common functionality for user/item profiling, review refinement, sampling, and parsing.

#### `structured_profile_agent.py`
5-stage structured workflow agent: builds user profile -> builds item profile -> cross-reasoning -> generates review -> refinement.

#### `reasoning_loop_agent.py`
Multi-step iterative reasoning agent with configurable reasoning steps (default: 4). Uses persistent context across reasoning iterations.

#### `ensemble_reviews_agent.py`
Generates three stylistically different reviews (concise/analytical/emotional), then uses a critic to synthesize them into a final review.

### Utilities Directory (`Util/`)

#### `ollama_llm.py`
Wrapper for local Ollama models. Provides `OllamaLLM` class for running local LLMs and `OllamaEmbeddings` for generating embeddings.

#### `gemini_llm.py`
Wrapper for Google Gemini models. Provides `GoogleLLM` class compatible with the project's LLM interface.

#### `debug_utils.py`
Core debugging utilities. Contains `LoggingLLMWrapper` for tracking LLM calls and `run_single_task()` for testing individual tasks.

#### `evaluation_utils.py`
Evaluation and testing functions. Includes `debug_single_task()` for single-task debugging with organized output and `run_evaluation()` for full simulation runs.

#### `experiment_utils.py`
Comprehensive experiment runner. Provides functions for multi-dataset evaluation, agent comparison, LLM comparison, and ablation studies.

#### `format_llm_logs.py`
Formats JSON LLM logs into human-readable text files with proper newlines and wrapping.

---

## Usage Examples

### Running a Single Task (Debugging)

```python
from websocietysimulator import Simulator
from Agents.structured_profile_agent import StructuredProfileAgent
from Util.ollama_llm import OllamaLLM
from Util.evaluation_utils import debug_single_task

# Initialize simulator and LLM
simulator = Simulator(data_dir="dataset", device="gpu", cache=True)
llm = OllamaLLM(model="mistral")

# Set up task
simulator.set_task_and_groundtruth(
    task_file="example/track1/yelp/tasks/tasks.json",
    groundtruth_file="example/track1/yelp/groundtruth/groundtruth.json"
)
simulator.set_agent(StructuredProfileAgent)

# Debug single task with LLM logging
result = debug_single_task(
    simulator=simulator,
    agent_name="StructuredProfileAgent",
    task_index=0,
    task_set="yelp",
    wrap_llm_with_logger=True,
    output_base_dir="./Outputs"
)

# Outputs saved to: ./Outputs/StructuredProfileAgent/debug_task0_TIMESTAMP.json
```

### Running Full Evaluation

```python
from websocietysimulator import Simulator
from Agents.reasoning_loop_agent import ReasoningLoopAgent
from Util.gemini_llm import GoogleLLM
from Util.evaluation_utils import run_evaluation

# Initialize with Gemini
simulator = Simulator(data_dir="dataset", device="gpu", cache=True)
llm = GoogleLLM(api_key="your-api-key", model="gemini-2.5-flash")

# Set up task
simulator.set_task_and_groundtruth(
    task_file="example/track1/amazon/tasks/tasks.json",
    groundtruth_file="example/track1/amazon/groundtruth/groundtruth.json"
)
simulator.set_agent(ReasoningLoopAgent)

# Run evaluation on 50 tasks
result = run_evaluation(
    simulator=simulator,
    agent_name="ReasoningLoopAgent",
    task_set="amazon",
    number_of_tasks=50,
    output_base_dir="./Outputs"
)

# Outputs saved to: ./Outputs/ReasoningLoopAgent/evaluation_amazon_TIMESTAMP.json
```

### Running Multi-Dataset Experiments

```python
from Util.experiment_utils import experiment_agent_comparison
from Agents.structured_profile_agent import StructuredProfileAgent
from Agents.reasoning_loop_agent import ReasoningLoopAgent
from Agents.ensemble_reviews_agent import EnsembleReviewsAgent

# Compare all three agents across all datasets
result = experiment_agent_comparison(
    agent_classes=[
        StructuredProfileAgent,
        ReasoningLoopAgent,
        EnsembleReviewsAgent
    ],
    llm_model="mistral",  # Ollama model
    datasets=["yelp", "amazon", "goodreads"],
    number_of_tasks=75,
    output_base_dir="./Outputs"
)

# Outputs saved to: ./Outputs/AgentComparison/agent_comparison_TIMESTAMP.json
```

### Running LLM Comparison

```python
from Util.experiment_utils import experiment_llm_comparison
from Agents.structured_profile_agent import StructuredProfileAgent

# Compare different LLMs with same agent
result = experiment_llm_comparison(
    agent_class=StructuredProfileAgent,
    llm_configs=[
        {"type": "ollama", "model": "mistral"},
        {"type": "ollama", "model": "llama3"},
        {"type": "gemini", "model": "gemini-2.5-flash", "api_key": "your-key"}
    ],
    datasets=["yelp", "amazon"],
    number_of_tasks=50,
    output_base_dir="./Outputs"
)

# Outputs saved to: ./Outputs/LLMComparison/llm_comparison_TIMESTAMP.json
```

### Formatting LLM Logs

```python
from Util.format_llm_logs import format_llm_logs

# Convert JSON debug logs to readable text
format_llm_logs(
    json_file="./Outputs/StructuredProfileAgent/debug_task0_20251130_120000.json",
    output_file="./Outputs/StructuredProfileAgent/debug_task0_20251130_120000_formatted.txt"
)
```

---

## Output Organization

All outputs are organized by agent name and timestamped:

```
Outputs/
├── StructuredProfileAgent/
│   ├── debug_task0_20251130_120000.json
│   ├── debug_task0_20251130_120000.txt
│   └── evaluation_yelp_20251130_120500.json
├── ReasoningLoopAgent/
│   └── evaluation_amazon_20251130_121000.json
├── AgentComparison/
│   └── agent_comparison_20251130_122000.json
└── LLMComparison/
    └── llm_comparison_20251130_123000.json
```
