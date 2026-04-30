## Project Overview

The project uses LLM-as-a-judge methods to analyze model rationales for clinical trial eligibility decisions. .

## Repository Structure

```text
clinical_trial_matching_LLM_explanation/
├── trialgpt_LLM_as_a_judge.py
├── n2c2_LLM_as_a_judge_prompt_1.py
├── n2c2_LLM_as_a_judge_prompt_2.py
├── results/
└── README.md
```

## Installation

Create and activate a conda environment:

```bash
conda create -n llm-as-a-judge python=3.10 -y
conda activate llm-as-a-judge
pip install -r requirements.txt
```

## Usage/Examples

Before running any script, activate the environment and set the OpenAI API key:

```bash
conda activate llm-as-a-judge
export OPENAI_API_KEY="your_api_key_here"
```

### n2c2 prompt 1 

This script should run top-5 retrieved evidence chunk:

```bash
python n2c2_LLM_as_a_judge_prompt_1.py path/to/input.csv --k 5
```

Run only a test:

```bash
python n2c2_LLM_as_a_judge_prompt_1.py path/to/input.csv --head 10
```

Run using the same input data file
```bash
python n2c2_LLM_as_a_judge_prompt_1.py "n2c2_results/gpt-4o-mini|sentence-transformers_all-MiniLM-L6-v2|5|each_criteria_all_notes|test|chunk|criteria-all.csv" --k 5
```
Save output to a specific file:

```bash
python n2c2_LLM_as_a_judge_prompt_1.py path/to/input.csv --out results/n2c2_results.csv
```
### n2c2 Prompt 2

This script should run top-5 retrieved evidence chunk:

```bash
python n2c2_LLM_as_a_judge_prompt_2.py "n2c2_results/gpt-4o-mini|sentence-transformers_all-MiniLM-L6-v2|5|each_criteria_all_notes|test|chunk|criteria-all.csv" --k 5
```

Run only a small test sample:

```bash
python n2c2_LLM_as_a_judge_prompt_2.py "n2c2_results/gpt-4o-mini|sentence-transformers_all-MiniLM-L6-v2|5|each_criteria_all_notes|test|chunk|criteria-all.csv" --k 5 --head 10
```

Save output to a specific file:

```bash
python n2c2_LLM_as_a_judge_prompt_2.py "n2c2_results/gpt-4o-mini|sentence-transformers_all-MiniLM-L6-v2|5|each_criteria_all_notes|test|chunk|criteria-all.csv" --k 5 --out results/n2c2_prompt2_judge_results.csv
```
## TrialGPT

Run the TrialGPT script:

```bash
python trialgpt_LLM_as_a_judge.py
```

Run only a test:

```bash
python trialgpt_LLM_as_a_judge.py --head 10
```
