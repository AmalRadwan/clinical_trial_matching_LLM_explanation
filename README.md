
This repository contains scripts and resluts for evaluating LLM-based explanations in clinical trial matching tasks.

## Project Overview

The project uses LLM-as-a-judge methods to analyze model rationales for clinical trial eligibility decisions. .

# Repository Structure

```text
clinical_trial_matching_LLM_expaination/
├── trialgpt_LLM_as_a_judge.py
├── n2c2_LLM_as_a_judge_prompt_1.py
├── n2c2_LLM_as_a_judge_prompt_2.py
├── results/
└── README.md

## Installation

Create and activate a conda environment:

```bash
conda create -n llm-as-a-judge python=3.10 -y
conda activate llm-as-a-judge
pip install -r requirements.txt


## Command-line Usage

Before running any script, activate the environment and set the OpenAI API key:

```bash
conda activate clinical_trial_matching
export OPENAI_API_KEY="your_api_key_here"

python trialgpt_LLM_as_a_judge.py

python n2c2_LLM_as_a_judge_prompt_1.py path/to/input.csv --k 5
