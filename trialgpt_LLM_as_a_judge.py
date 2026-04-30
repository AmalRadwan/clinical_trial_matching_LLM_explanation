import pandas as pd
import argparse
import csv
import json
import os
import time
import sys  

from openai import OpenAI
from datasets import load_dataset, concatenate_datasets
import datetime


# Basic configuration
HF_DATASET_NAME = "ncbi/TrialGPT-Criterion-Annotations"
DEFAULT_MODEL = "gpt-4o-mini"
MAX_RETRIES = 3

INCORRECT_CATS = [
    "complete",
    "partially_complete",
    "extraction_error",
    "assumption",
    "logical_error",
    "other",
]

INCORRECT_DEFS = {
    "complete": "The rationale is logically complete.",
    "partially_complete": "The rationale only applies partially to the criterion definition.",
    "extraction_error": "The rationale relies on incorrect facts about the patient or trial, or misses crucial information.",
    "assumption": "The rationale relies on facts about the patient or trial not mentioned explicitly or missing.",
    "logical_error": "The rationale has a logical error.",
    "other": "Does not fit any of the five categories.",
}

PARTIAL_CATS = [
    "assumption",
    "extraction_error",
    "criteria_misinterpretation",
    "rational_correct",
    "other",
]

PARTIAL_DEFS = {
    "assumption": "The rationale relies on facts about the patient or trial not mentioned explicitly or missing.",
    "extraction_error": "The rationale relies on incorrect facts about the patient or trial, or misses crucial information.",
    "rational_correct": "The rationale is logical but the label chosen by the model was not correct.",
    "criteria_misinterpretation": "Misreads/misunderstands the criterion text or type.",
    "other": "Does not fit any category above.",
}

ALLOWED_CONF = {"low", "medium", "high"}
REQUIRED_KEYS = {"category", "confidence", "reasons", "eligibility_notes"}

SYSTEM_MSG = (
    "You are a clinical assessor who reviews a model's incorrect predictions, mismatches, "
    "and errors in selecting eligible patients for a clinical trial. "
    "Your task is to assess the model's rationale, which can be partially correct or incorrect, "
    "and classify the error into the most appropriate category. "
    "Use the evidence provided in the INPUT JSON as the primary source. "
    "You MAY use widely accepted standard clinical reference knowledge, such as typical lab reference ranges, "
    "only when needed to interpret a value. "
    "If the evidence is insufficient, state that clearly in the reasons and choose the best-fitting category anyway. "
    "Return ONLY valid JSON."
)

# source "https://developers.openai.com/api/docs/models/gpt-4o-mini"
# source "https://developers.openai.com/api/docs/models/gpt-4o"
MODEL_PRICING = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.00060},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.0100},
}


# output csv
#write the csv header
def write_csv_header (path,column_names):
    with open (path, "w", newline ="", encoding="utf-8")as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames= column_names)
        writer.writeheader()

# Append rows to the csv
def append_csv_rows(path, column_names, rows):
    with open (path, "a", newline="", encoding="utf-8")as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames= column_names)
        writer.writerows(rows)
 

# Load data 
def load_huggingface_data():
    print("Loading the whole TrialGPT dataset from Hugging Face...")

    dataset_obj = load_dataset(HF_DATASET_NAME)

    if isinstance(dataset_obj, dict):
        all_splits = []
        for split_name in dataset_obj.keys():
            print(f"Loading split: {split_name} | rows: {len(dataset_obj[split_name])}")
            all_splits.append(dataset_obj[split_name])

        dataset_obj = concatenate_datasets(all_splits)

    df = dataset_obj.to_pandas()

    print(f"Dataset loaded: {HF_DATASET_NAME} | total rows: {len(df)}")
    return df


# choose the rational correctness 
def classify_correctness(value):
    if not isinstance(value, str):
        return ""
    value = value.strip().lower()
    if "incorrect" in value:
        return "incorrect"
    if "partial" in value:
        return "partial"
    return ""

# choose allowed categories and their definitions for each 
def get_labels_for_correctness_group(correctness_group):
    if correctness_group == "partial":
        return PARTIAL_CATS, PARTIAL_DEFS

    if correctness_group == "incorrect":
        return INCORRECT_CATS, INCORRECT_DEFS

    raise ValueError(f"Unknown correctness_group: {correctness_group}")

# Creat a prompt
def build_prompt(row, correctness_group):
    cats, defs = get_labels_for_correctness_group(correctness_group)


    trial_title = strip_text(row.get("trial_title", row.get("trial_name", "")))
    criterion_type = strip_text(row.get("criterion_type", ""))
    criterion_text = strip_text(row.get("criterion_text", ""))
    gpt4_explanation = strip_text(row.get("gpt4_explanation", ""))
    explanation_correctness = strip_text(row.get("explanation_correctness", ""))
    gpt4_sentences = row.get("gpt4_sentences", "")
    expert_sentences = row.get("expert_sentences", "")
    gpt4_eligibility = strip_text(row.get("gpt4_eligibility", row.get("gpt_eligibility", "")))
    expert_eligibility = strip_text(row.get("expert_eligibility", ""))
    note = strip_text(row.get("note_text", ""))


    input_data = {}

    input_data["task"] = "Assess the model rationale and classify the reasoning error to a category using the provided TrialGPT evidence."
    input_data["correctness_group"] =correctness_group
  
    input_data["trial_title"] = trial_title
    
    input_data["criterion_type"] =criterion_type
    input_data["criterion_text"] =criterion_text
    input_data["patient_note"] =note

    input_data["gpt4_explanation"] = gpt4_explanation
    input_data["explanation_correctness"] =explanation_correctness
    input_data["gpt4_sentences"] = gpt4_sentences
    input_data["expert_sentences"] = expert_sentences
    input_data["gpt4_eligibility"] = gpt4_eligibility
    input_data["expert_eligibility"] = expert_eligibility

    input_data["allowed_categories"] = cats
    input_data["category_explanations"] = defs
    input_data["confidence_allowed"] = sorted(list(ALLOWED_CONF))

    #  output description 
    input_data["output_schema"] = {
    "category": f"one of {cats}",
    "confidence": f"one of {ALLOWED_CONF}",
    "reasons": "1 to 2 sentences explaining why you chose this category.",
    "eligibility_notes": "short explanation connecting chosen category and reasons to the patient note.",}

    input_data["rules"] = [
    "Assess the correctness of the model rationale with respect to the criterion definition.",
    "Choose exactly one 'category' from allowed_categories for this correctness_group.",
    "Use trial_title, criterion_type, criterion_text, gpt4_explanation, gpt4_sentences, expert_sentences, gpt4_eligibility, and expert_eligibility as the primary source.",
    "Use gpt4_sentences and expert_sentences to compare what evidence GPT-4 used versus what the expert used.",
    "If the evidence is insufficient, state that clearly in the reasons.",
    "Return ONLY valid JSON with exactly these keys: category, confidence, reasons, eligibility_notes."]

    json_string = json.dumps(input_data, ensure_ascii=False)
    user_prompt = "INPUT JSON:\n" + json_string
    #Debugging
    #print(user_prompt)
    return user_prompt


#remove whitespace at the beginning and end
def strip_text(text):
    if not text:
        return ""
    return str(text).strip()


# calculate the usage of tokens for each model
def get_tokens_usage(response):
    prompt_tokens = 0
    completion_tokens = 0

    usage= getattr(response,"usage", None)
    if usage is None:
        return prompt_tokens, completion_tokens
    raw_prompt_tokens= getattr(usage, "prompt_tokens",None)
    raw_completion_tokens = getattr(usage, "completion_tokens",None)
    if raw_prompt_tokens:
        prompt_tokens = int(raw_prompt_tokens)
    else:
        prompt_tokens = 0
    if raw_completion_tokens:
        completion_tokens = int(raw_completion_tokens)
    else:
        completion_tokens =0
    return prompt_tokens, completion_tokens

# validate the ouptput to avoid errors 
def validate_output(data, correctness_group):
    
    required_cat= {"category", "confidence", "reasons", "eligibility_notes"}
    actual_keys = set(data.keys())
    # added for debugging
    missing = required_cat- actual_keys

    if missing:
        raise ValueError(f"missing_key:{','.join(missing)}")
    raw_category=data.get("category", "")
    raw_confidence = data.get("confidence", "")
   
    clean_category = str(raw_category).strip().lower()
    clean_confidence = str(raw_confidence).strip().lower()

    allowed_cats, defs =get_labels_for_correctness_group(correctness_group)
    if clean_category not in allowed_cats:
         raise ValueError(f"invalid_category:{clean_category}")

    allowed_confs = {"low", "medium", "high"}
    if clean_confidence  not in allowed_confs:
        raise ValueError(f"invalid_confidence:{clean_confidence}")


    
    data["category"] = clean_category
    data["confidence"] = clean_confidence
    data["reasons"] = str(data.get("reasons", "")).strip()
    data["eligibility_notes"] = str(data.get("eligibility_notes", "")).strip()

   
    return data


def call_llm(client, model, initial_prompt, correctness_group, max_retries=3):
  
    last_output = ""
    # save the error so the tiunderstand why the run failed
    error_kind = None

    # prompt change across retries if call fails
    current_prompt =initial_prompt

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content":current_prompt},
                ],
                # deterministic setting
                temperature=0.0,
                # reduces variation across multiple runs
                seed=42,
                response_format={"type": "json_object"},
            )

            raw_text = response.choices[0].message.content or ""
            last_output =raw_text.strip()
            # convert model's JSON string into a dict to inspect and validate 
            parsed_output = json.loads(last_output)

            # Validation to avoid failures
            validated_output = validate_output(parsed_output, correctness_group)

            prompt_tokens, completion_tokens = get_tokens_usage(response)
            return {
                "status": "ok",
                "data":validated_output,
                "attempts": attempt,
                "error_type":None,
                "raw_text": last_output,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
       
        except json.JSONDecodeError:
            error_kind = "json_parse"
         # keep the exception text to make debugging easier
        except Exception as e:
            error_kind = str(e)

        allowed_cats, defs = get_labels_for_correctness_group(correctness_group)
        # If the first attempt was invalid, the next prompt becomes a repair prompt
        repair_prompt = f"""{initial_prompt}

IMPORTANT CORRECTION:
Your previous answer did not match the required output format.

Return ONLY valid JSON with these keys:
category, confidence, reasons, eligibility_notes

Rules:
- category must be one of: {allowed_cats}
- confidence must be one of: {sorted(list(ALLOWED_CONF))}
"""
        # replace the current prompt with repair prompt so the etries should focus on fixing
        current_prompt = repair_prompt
    return {
        "status": "llm_failure",
        "data": None,
        "attempts": max_retries,
        "error_type": error_kind or "unknown_error",
        "raw_text": last_output,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
########### reporting ############
def estimate_cost(model, prompt_tokens, completion_tokens):
    if model not in MODEL_PRICING:
        return 0.0
    pricing = MODEL_PRICING[model]

    # Prompt and completion tokens are billed at different rates,
    prompt_rate = pricing["prompt"]
    completion_rate =pricing["completion"]

    prompt_cost =(prompt_tokens / 1000) * prompt_rate
    completion_cost =(completion_tokens / 1000) * completion_rate
    total_cost = prompt_cost + completion_cost
    return total_cost
# Cost and time output report 
def write_report(path, model, stats):
  
    end_time = time.time()
    run_time_seconds = end_time - stats["start_time"]

    # count tokens
    prompt_tokens = stats["prompt_tokens"]
    completion_tokens = stats["completion_tokens"]
    total_tokens = prompt_tokens + completion_tokens

    #calculate average latency 
    llm_calls = stats["llm_calls"]
    total_latency_ms = stats["total_latency_ms"]
    if llm_calls > 0:
        average_latency_ms =total_latency_ms /llm_calls
    else:
        average_latency_ms =0

    #cost estimated in USD
    estimated_cost_usd = estimate_cost(model, prompt_tokens, completion_tokens)

    #text report 
    with open(path, "w", encoding="utf-8") as txt_report:
        txt_report.write("LLM price and cost summary")
        txt_report.write("*" * 40 + "\n")
        txt_report.write(f"Model: {model}\n")
        txt_report.write(f"Total rows processed: {stats['total_rows']}\n")
        txt_report.write(f"Number of LLM Calls:  {llm_calls}\n")
        txt_report.write(f"Successful calls: {stats['llm_success']}\n")
        txt_report.write(f"Failed calls: {stats['llm_failures']}\n")
        txt_report.write(f"Total attempts (incling retries): {stats['total_attempts']}\n")
        txt_report.write("\n")
        txt_report.write("*" * 40 + "\n")
        txt_report.write(f"Prompt tokens: {prompt_tokens}\n")
        txt_report.write(f"Completion tokens: {completion_tokens}\n")
        txt_report.write(f"Total tokens: {total_tokens}\n")
        txt_report.write("\n")
        txt_report.write("*" * 40 + "\n")
        txt_report.write(f"Total run time (s): {run_time_seconds:.2f}\n")
        txt_report.write(f"Average latency per call (ms): {average_latency_ms:.1f}\n")
        txt_report.write("\n")
        txt_report.write("*" * 40 + "\n")
        txt_report.write(f"Estimated cost(USD): ${estimated_cost_usd:.4f}\n")
#sanity check
def sanity_check():
    assert classify_correctness("Incorrect") == "incorrect"
    assert classify_correctness("Partially Correct") == "partial"
    assert classify_correctness("Correct") == ""
    assert classify_correctness(None) == ""
    assert classify_correctness(123) == ""

# Main
############
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--head", type=int, default=0, help="Process only first N head after filtering. 0 means all")
    parser.add_argument("--correctness_group", type=str, default="both", choices=["incorrect", "partial", "both"], help="Which TrialGPT category correctness_group to process")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--out", type=str, default="", help="Output CSV path (optional)")
    args = parser.parse_args()

    # API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("ERROR: set OPENAI_API_KEY environment")
    client = OpenAI(api_key=api_key)

    # Load TrialGPT from Hugging Face
    df = load_huggingface_data()

    if "explanation_correctness" not in df.columns:
        raise RuntimeError("ERROR: 'explanation_correctness' column is missing")

    # Reset index
    df = df.reset_index().rename(columns={'index': 'original_index'})

    # keep only incorrect and partially correct rationals 
    df["correctness_group"] = df["explanation_correctness"].map(classify_correctness)
    if args.correctness_group == "incorrect":
        df = df[df["correctness_group"] == "incorrect"]
    elif args.correctness_group == "partial":
        df = df[df["correctness_group"] == "partial"]
    else:
        df = df[df["correctness_group"].isin(["incorrect", "partial"])]

    trial_rows = df.to_dict(orient="records")


     
    if args.head> 0:
        print(f"Only processing first {args.head} TrialGPT rows")
        trial_rows = trial_rows[:args.head]

    # Output file
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.out:
        output_csv_path =args.out

    else:
        output_csv_path = f"trialgpt_results__{args.correctness_group}_{run_stamp}.csv"
     
    fieldnames = []
    if trial_rows:
        fieldnames = list(trial_rows[0].keys())
    else:
        fieldnames = list(df.columns)

    added_fields = [
        "model",
        "rationale_category", "rationale_confidence","rationale_reasons",
        "rationale_notes",
        "assessor_evidence_ids", "retrieved_count", "latency_ms","error",
        "llm_status", "llm_error_type", "llm_attempts",
        "prompt_tokens", "completion_tokens",
    ]
    for col in added_fields:
        if col not in fieldnames:
            fieldnames.append(col)

    write_csv_header(output_csv_path, fieldnames)

    # Statistics
    stats = {
        "start_time":time.time(),
        "total_rows": 0,
        "llm_calls": 0,
        "llm_success":0,
        "llm_failures":0,
        "total_attempts":0,
        "prompt_tokens":  0,
        "completion_tokens": 0,
        "total_latency_ms": 0,
    }
    # temporarily store results
    output_buffer = []
  

    for idx, one_row in enumerate(trial_rows):
        correctness_group = str(one_row.get("correctness_group", "")).strip()

        prompt = build_prompt(
            one_row,
            correctness_group,
        )


        # print prompt (debug)
        ''''
        print("\n" + "="*80)
        print("PROMPT SENT TO LLM")
        print(f"correctness_group: {correctness_group}")
        print("-"*80)
        print(prompt)
        print("="*80 + "\n", flush=True)
        '''
        # Call the LLM
        start_time =time.time()
        result = call_llm(client, args.model, prompt, correctness_group, MAX_RETRIES)
        end_time =time.time()
        latency_ms=int((end_time - start_time) * 1000)

        # update stats
        stats["total_rows"] += 1
        stats["llm_calls"] += 1
        stats["total_attempts"] += int(result.get("attempts", 0))
        stats["total_latency_ms"] += latency_ms
        stats["prompt_tokens"] += int(result.get("prompt_tokens", 0))
        stats["completion_tokens"] += int(result.get("completion_tokens", 0))

        if result["status"] == "ok":
            stats["llm_success"] += 1
            data = result["data"]
            category = data["category"]
            confidence = data["confidence"]
            reason_text = str(data.get("reasons", "")).strip()
           
            notes = str(data.get("eligibility_notes", "")).strip()
            error_message = ""
        else:
            stats["llm_failures"] += 1
            category = ""
            confidence = ""
            reason_text = "LLM failed to produce valid structured output after retries."
            
            notes = ""
            error_message = f"llm_failure:{result.get('error_type','unknown_error')}"

        output_row = dict(one_row)
        output_row["model"] = args.model
        output_row["rationale_category"] = category
        output_row["rationale_confidence"] = confidence
        output_row["rationale_reasons"] = reason_text
     
        output_row["rationale_notes"] = notes
        output_row["assessor_evidence_ids"] = "[]"
        output_row["retrieved_count"] = 0
        output_row["latency_ms"] = latency_ms
        output_row["error"] = error_message
        output_row["llm_status"] = result["status"]
        output_row["llm_error_type"] = result.get("error_type", "")
        output_row["llm_attempts"] = result.get("attempts", 0)
        output_row["prompt_tokens"] = int(result.get("prompt_tokens", 0))
        output_row["completion_tokens"] = int(result.get("completion_tokens", 0))

        output_buffer.append(output_row)
       
        if len(output_buffer) >= 50:
            append_csv_rows(output_csv_path, fieldnames, output_buffer)
            output_buffer = []


    # write output 
    if output_buffer:
        append_csv_rows(output_csv_path, fieldnames, output_buffer)

    # Report
    report_path = output_csv_path.replace(".csv", "_report.txt")
    write_report(report_path, args.model, stats)

    print(f"[done] results written to {output_csv_path}")
    print(f"[done] report written to {report_path}")

if __name__ == "__main__":
    sanity_check()
    print("Starting TrialGPT LLM-as-a-judge analysis ....")
    main()
