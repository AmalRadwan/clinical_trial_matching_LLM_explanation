import pandas as pd
import argparse
import csv
import json
import os
import time
import sys  
from typing import Any, Dict, List, Optional, Tuple
import torch
import chromadb
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI

import datetime

# Basic configuration
CHROMA_PATH = "/home/user/ctpm-main/data/chroma"
 # use the same match the embedding model used in the chromadb
COLLECTION_NAME = "all-MiniLM-L6-v2"         
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_K = 3
MAX_RETRIES = 3

CRITERIA = {
    "ABDOMINAL": "History of intra-abdominal surgery. This could include any form of intra-abdominal surgery, including but not limited to small/large intestine resection or small bowel obstruction",
    "ADVANCED-CAD": "Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define “advanced” as having 2 or more of the following: (a) Taking 2 or more medications to treat CAD (b) History of myocardial infarction (MI) (c) Currently experiencing angina (d) Ischemia, past or present. The patient must have at least 2 of these categories (a,b,c,d) to meet this criterion, otherwise the patient does not meet this criterion. For ADVANCED-CAD, be strict in your evaluation of the patient – if they just have cardiovascular disease, then they do not meet this criterion.",
    "ALCOHOL-ABUSE": "Current alcohol use over weekly recommended limits",
    "ASP-FOR-MI": "Use of aspirin for preventing myocardial infarction (MI)",
    "CREATININE": "Serum creatinine level above the upper normal limit",
    "DIETSUPP-2MOS": "Consumption of a dietary supplement (excluding vitamin D) in the past 2 months. To assess this criterion, go through the list of medications and supplements taken from the note. If a substance could potentially be used as a dietary supplement (i.e. it is commonly used as a dietary supplement, even if it is not explicitly stated as being used as a dietary supplement),then the patient meets this criterion. Be lenient and broad in what is considered a dietary supplement. For example, a 'multivitamin' and 'calcium carbonate' should always be considered a dietary supplement if they are included in this list.",
    "DRUG-ABUSE": "Current or past history of drug abuse",
    "ENGLISH": "Patient speaks English. Assume that the patient speaks English, unless otherwise explicitly noted. If the patient's language is not mentioned in the note, then assume they speak English and thus meet this criteria.",
    "HBA1C": "Any hemoglobin A1c (HbA1c) value between \"6.5%\" and 9.5%",
    "KETO-1YR": "Diagnosis of ketoacidosis within the past year",
    "MAJOR-DIABETES": "Major diabetes-related complication. Examples of “major complication” (as opposed to “minor complication”) include, but are not limited to, any of the following that are a result of (or strongly correlated with) uncontrolled diabetes: Amputation, Kidney damage, Skin coditionconditions, Retinopathy, nephropathy and neuropathy. Additionally, if multiple conditions together imply a severe case of diabetes, then count that as a major complication.",
    "MAKES-DECISIONS": "Patient must make their own medical decisions. Assume that the patient makes their own medical decisions, unless otherwise explicitly noted. There is no information provided about the patientś ability to make their own medical decisions, then assume they do make their own decisions and therefore meet this criteria",
    "MI-6MOS": "Myocardial infarction (MI) within the past 6 months"
}

ALLOWED_CATEGORIES = [
    "complete",
    "partially_complete",
    "extraction_error",
    "assumption",
    "logical_error",
    "other",
]

CATEGORY_EXPLANATIONS = {
    "complete": "The rationale is logically complete.",
    "partially_complete": "The rationale only applies partially to the criterion definition.",
    "extraction_error": "The rationale relies on incorrect facts about the patient or trial, or it misses crucial information.",
    "assumption": "The rationale relies on facts about the patient or trial not mentioned explicitly or missing.",
    "logical_error": "The rationale has a logical error.",
    "other": "Does not fit any of the five categories.",
}

ALLOWED_CONF = {"low", "medium", "high"}
REQUIRED_KEYS = {"category", "confidence", "reason", "evidence_ids"}


SYSTEM_MSG = (
    "You are a clinical trial assessor that reviews a model's wrong predictions or mismatches and errors in selecting eligibile patients and assess the model rationales which could be faulty and classify them into categories. "
    "Use the evidence provided in the INPUT JSON as the primary source. "
    "You MAY use widely accepted general medical knowledge (e.g., typical lab reference ranges) only when necessary to interpret values; "
    "if you do, explicitly label those parts of your reasoning as 'General medical knowledge:' and do not claim they come from the evidence. "
    "Return ONLY valid JSON."
)

# source "https://developers.openai.com/api/docs/models/gpt-4o-mini"
# source "https://developers.openai.com/api/docs/models/gpt-4o"
MODEL_PRICING = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.00060},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.0100},
}

# Read the CSV file 
def read_csv (path :str) -> List [Dict [str, str]]:
    rows = []
    with open (path, "r", encoding = "utf-8", newline="")as file_obj:
        reader_obj = csv.DictReader (file_obj)
        for row in reader_obj:
            rows.append (row)
        return rows

# output csv
#write the csv header
def csv_header (path,column_names):
    with open (path, "w", newline ="", encoding="utf-8")as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames= column_names)
        writer.writeheader()

# Append rows to the csv
def append_rows(path, column_names, rows):
    with open (path, "a", newline="", encoding="utf-8")as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames= column_names)
        writer.writerows(rows)
 


# Creat a prompt
def build_prompt(patient_id, criterion_id, criteria_def, rationale, meds, docs, ids, dists, k):
    evidence_list =[]

    # get the chunk text
    for idx in range(k):
        if idx < len(docs):
            doc_text = docs[idx]
        else:
            doc_text = ""
        # get the chunk_ id for each chunk
    
        if idx < len(ids):
            chroma_id = ids[idx]
        else:
            chroma_id = "MISSING"
        # Get similarity distance
        if idx < len(dists):
            dist_value = dists[idx]
        else:
            dist_value = None
    
        # clean the chunk text before adding it to the prompt
        cleaned_text = strip_text(doc_text)

        evidence_list.append ({
            "rank": idx,
            "chroma_id": chroma_id,
            "distance": dist_value,
            "text": cleaned_text
        })

    input_data = {}

    input_data["task"] = "Assess the model rationale and classify the reasoning error to a category using the provided evidence."
    input_data["patient_id"] = patient_id
    input_data["criterion"] = {"id": criterion_id,"definition": criteria_def}
    input_data["model_rationale"] = rationale
    input_data["medications_and_supplements"] = meds
    input_data["allowed_categories"] = ALLOWED_CATEGORIES
    input_data["category_explanations"]= CATEGORY_EXPLANATIONS
    input_data["confidence_allowed"] = sorted(list(ALLOWED_CONF))
    input_data["retrieval"] = {
    "k_requested": k,
    "retrieved_count": len(docs)
}
    input_data["evidence"] = evidence_list

    #  output description 
    input_data["output_schema"] = {
    "category": f"one of {ALLOWED_CATEGORIES}",
    "confidence": f"one of {ALLOWED_CONF}",
    "reason": "2-4 sentences. Must reference evidence by chroma_id or rank when possible.",
    "evidence_ids": "JSON list of evidence chroma_id strings used. Can be empty.",}

    input_data["rules"] = [
        "Assess the MODEL RATIONAL relative to the criterion.",
        "Choose a single 'category' from allowed_categories.",
        "Use the evidence field as primary source.",
        "You MAY use widely accepted standard clinical reference knowledge (e.g., typical lab reference ranges) ONLY when needed to interpret a value. "
        "When you use such knowledge, explicitly label it in the reason as 'General medical knowledge:' and do NOT claim it came from evidence.",
        "If evidence is insufficient, state that clearly in 'reason' and choose the best-fitting category anyway.",
        "Return ONLY valid JSON with exactly these keys: category, confidence, reason, evidence_ids. "
        "Set evidence_ids to the IDs you used from evidence; it can be an empty list if none apply.",
    ]

    json_string = json.dumps(input_data, ensure_ascii=False)
    user_prompt = "INPUT JSON:\n" + json_string
    #Debugging
    #print(user_prompt)
    return user_prompt


#remove whitespace at the beginning and end
def strip_text(text):
    if not text:
        return ""
    return text.strip()


# Chroma retrieval using the same logic as eval.py and helprt.py in n2c2
# load model and tokenizer for embedding
def load_model(model_name: str, device: Optional[str] = None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer, device

#Embedding function to embed the criteria
def embed(text: str, model, tokenizer, device) -> list:
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        outputs = model(**tokens)
        embedding = outputs[0][0].mean(dim=0).cpu()
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
    return embedding.tolist()

def retrieve_chunks(
    criterion_definition: str,
    patient_id: str,
    collection: chromadb.Collection,
    model,
    tokenizer,
    device,
    n_chunks: Optional[int] = None,
    threshold: Optional[float] = None,
    keep_extra: bool = False,          #  to keep all the notes when needed 
    debug: bool = False,
) -> Tuple[List[str], List[str], List[float], List[str]]:
    """
    Given a patient and criterion, query Chroma db for relevant documents for that criterion.
        Return the top-n_chunks documents.
    Returns:
        docs_sorted: list of chunk texts sorted by note_idx/chunk_idx
        ids_sorted: corresponding chunk IDs
        dists_sorted: corresponding distances
        similarity_ids: chunk IDs in original similarity order (as returned by Chroma)
    """
    # Special case: return all chunks for this patient and not use similarity 
    if n_chunks == 9999:
        results = collection.get(
            where={"patient_id": patient_id},
            include=["metadatas", "documents"]
        )
        ids = results["ids"]
        docs = results["documents"]
        metas = results["metadatas"]
        #get fake smilarity (no needed here)
        distances = [0.0] * len(ids)
       
        results = {
            "ids": [ids],
            "distances": [distances],
            "documents": [docs],
            "metadatas": [metas]
        }
        # get how many raw chunks were found
        raw_count = len(ids)
    else:
        #vget the chunks for the criteria 
        # defualt is 
        n_results = n_chunks if n_chunks is not None else 10
        # embed the criteria
        query_emb = embed(criterion_definition, model, tokenizer, device)
        results = collection.query(
            query_embeddings=[query_emb],
            where={"patient_id": patient_id},
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        raw_count = len(results["ids"][0])

    # Build list of records (in similarity order)

    records = []
    for i in range(raw_count):
        distance = results["distances"][0][i]
        similarity = 1 -distance
        if threshold is not None and similarity < threshold:
            continue
        text = results["documents"][0][i]
        #remove very short chunks (rule in n2c2)
        if len(text) < 40:          
            continue
        records.append({
            "id": results["ids"][0][i],
            "metadata": results["metadatas"][0][i],
            "similarity":similarity,
            "text":text,
            "distance":distance,
        })

    # If not keep_extra and n_chunks given, truncate to n_chunks
    if n_chunks is not None and n_chunks != 9999 :#and not keep_extra and len(records) > n_chunks:
        records = records[:n_chunks]

    # Original similarity order 
    similarity_ids = [r["id"] for r in records]
    # Sort by note_idx, chunk_idx
    records_sorted = sorted(records, key=lambda x: (
        int(x["metadata"].get("note_idx", 0)),
        int(x["metadata"].get("chunk_idx", 0))
    ))

    docs_sorted = [r["text"] for r in records_sorted]
    ids_sorted = [r["id"] for r in records_sorted]
    dists_sorted = [r["distance"] for r in records_sorted]

    return docs_sorted, ids_sorted, dists_sorted, similarity_ids



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
def validate_output(data):
    
    required_cat= {"category", "confidence", "reason", "evidence_ids"}
    actual_keys = set(data.keys())
    # added for debugging
    missing = required_cat- actual_keys

    if missing:
        raise ValueError(f"missing_key:{','.join(missing)}")

    raw_category= data.get("category", "")
    raw_confidence = data.get("confidence", "")
    raw_evidence_ids= data.get("evidence_ids", [])

    clean_category = str(raw_category).strip().lower()
    clean_confidence = str(raw_confidence).strip().lower()

    allowed_cats = ["complete","partially_complete" ,"extraction_error", "assumption", "logical_error","other"]
    if clean_category not in allowed_cats:
         raise ValueError(f"invalid_category:{clean_category}")


    allowed_confs = {"low", "medium", "high"}
    if clean_confidence  not in allowed_confs:
        raise ValueError(f"invalid_confidence:{clean_confidence}")
    

    if not isinstance(raw_evidence_ids, list):
        raise ValueError("invalid_evidence_ids:not_list")

    data["category"] = clean_category
    data["confidence"] = clean_confidence
    return data

# call the LLM-as-a-judge 
def call_llm(client, model, initial_prompt, max_retries=3):
   
    #  keep the last output for debugging when the call does not succeed 
    last_output = ""

    # save the error so the tiunderstand why the run failed
    error_kind = None

    # The prompt change across retries if call fails
    current_prompt = initial_prompt

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content":current_prompt},
                ],
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
            validated_output = validate_output(parsed_output)

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

        # If the first attempt was invalid, the next prompt becomes a repair prompt
        repair_prompt = f"""{initial_prompt}

IMPORTANT CORRECTION:
Your previous answer did not match the required output format.

Return ONLY valid JSON with these keys:
category, confidence, reason, evidence_ids
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



# Main
############
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("input_csv", type=str, help="path to input CSV file")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-K evidence to retrieve")
    # to test and try few rows 
    parser.add_argument("--head", type=int, default=None, help="Process only first N rows (after filtering)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument("--out", type=str, default="", help="Output CSV path (optional)")
    parser.add_argument("--device", type=str, default=None, help="Device for embedding model (cuda/cpu/mps)")
    #simlilar to n2c2 study
    parser.add_argument("--keep-extra", action="store_true", default= False,
                        help="If False, then only use specific chunks that meet threshold. If True, then use full note if any chunk within that note meets threshold")
    args = parser.parse_args()

    # API key
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("ERROR: set OPENAI_API_KEY environment")
    client = OpenAI(api_key=api_key)
    # Load embedding model 
    print("Loading embedding model...")
    embed_model, embed_tokenizer, device = load_model(EMBED_MODEL_NAME, args.device)
    print(f"Embedding model loaded on {device}")

    # Connect to ChromaDB
    print("Connecting to Chroma...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' loaded (size: {collection.count()})")

    # read and filter CSV (only mismatches)
    df = pd.read_csv(args.input_csv)

    # Reset index
    df = df.reset_index().rename(columns={'index': 'original_index'})

    # keep only mismatches
    df = df[df["is_met"] != df["true_label"]]
    mismatch_rows = df.to_dict(orient="records")

     
    if args.head:
        print(f"Only processing first {args.head} mismatch rows")
        mismatch_rows = mismatch_rows[:args.head]

    # Output file
    base_name = os.path.splitext(os.path.basename(args.input_csv))[0]
     # add a timestamp so each run gets its own file
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.out:
        output_csv_path =args.out

    else:
        output_csv_path = f"{base_name}_results_k{args.k}_{run_stamp}.csv"
     
        fieldnames = [
            "index", "note","patient_id", "criterion", "rationale", "medications",
            "is_met", "true_label",
            "assessed_category","assessor_confidence","assessor_reason",
            "assessor_evidence_ids",
            "used_k", "retrieved_count","latency_ms", "error",
            "llm_status","llm_error_type","llm_attempts",
        ]
        csv_header(output_csv_path, fieldnames)

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

    for idx, one_row in enumerate(mismatch_rows):
        patient_id = str(one_row.get("patient_id", "")).strip()
        criterion_id = str(one_row.get("criterion", "")).strip()
        rationale_text = str(one_row.get("rationale", "")).strip()
        medications_text = str(one_row.get("medications_and_supplements", "")).strip()
        is_met_value = str(one_row.get("is_met", "")).strip()
        true_label_value = str(one_row.get("true_label", "")).strip()

        definition = CRITERIA.get(criterion_id, "").strip()
        query_text = "\n".join([criterion_id, definition]).strip()

        # Retrieve using the retrival function
        try:
            docs, ids, dists, similarity_ids = retrieve_chunks(
                criterion_definition=query_text,
                patient_id=patient_id,
                collection=collection,
                model=embed_model,
                tokenizer=embed_tokenizer,
                device=device,
                n_chunks=args.k,
                threshold=None,
                keep_extra=args.keep_extra
            )
            
            retrieved_count = len(docs)


            #  print retrieved chunks to compare to torignal n2c2 
            '''
            for i, doc in enumerate(docs, 1):
              print(f"\n--- ordered chunk {i} ---")
              print(doc[:500])

            # deugging 
            #print chunks
            print(f"\n{'='*80}")
            print(f"PATIENT: {patient_id} | CRITERION: {criterion_id}")
            print(f"Retrieved {retrieved_count} chunks:")
            for i, (doc_id, doc_text, dist) in enumerate(zip(ids, docs, dists)):
                print(f"\n--- Chunk {i+1} ---")
                print(f"ID: {doc_id}")
                print(f"Distance: {dist:.4f}")
                # Print first 300 chars of the text
                preview = doc_text[:300] + "..." if len(doc_text) > 300 else doc_text
                print(f"Text: {preview}")
            print(f"{'='*80}\n")
            '''
            # ===== end print =====


        except Exception as e:
             error_row = {
                "index": one_row.get("original_index", idx),
                "note": one_row.get("note", ""),
                "patient_id":patient_id,
                "criterion": criterion_id,
                "rationale": rationale_text,
                "medications":medications_text,
                "is_met":is_met_value,
                "true_label": true_label_value,
                "assessed_category": "",
                "assessor_confidence": "",
                "assessor_reason": "",
                "assessor_evidence_ids": "[]",
                "used_k": args.k,
                "retrieved_count": 0,
                "latency_ms": 0,
                "error": f"retrieval_error:{e}",
                "llm_status": "llm_failure",
                "llm_error_type":"retrieval_error",
                "llm_attempts": 0,}
             
             output_buffer.append(error_row)
             if len(output_buffer) >= 50:
                    append_rows(output_csv_path, fieldnames, output_buffer)
                    output_buffer = []
             continue

        # Build prompt
        prompt = build_prompt(
            patient_id,
            criterion_id,
            definition,
            rationale_text,
            medications_text,
            docs,
            ids,
            dists,
            args.k,
        )

        # print prompt (debug)
        ''''
        print("\n" + "="*80)
        print("PROMPT SENT TO LLM")
        print(f"patient_id: {patient_id}")
        print(f"criterion: {criterion_id}")
        print("-"*80)
        print(prompt)
        print("="*80 + "\n", flush=True)
        '''
        # Call the LLM
        start_time =time.time()
        result = call_llm(client, args.model, prompt, MAX_RETRIES)
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
            reason_text = str(data.get("reason", "")).strip()
            evidence_ids_used = data.get("evidence_ids", [])
            error_message = ""
        else:
            stats["llm_failures"] += 1
            category = ""
            confidence = ""
            reason_text = "LLM failed to produce valid structured output after retries."
            evidence_ids_used = []
            error_message = f"llm_failure:{result.get('error_type','unknown_error')}"

        output_row = {
            "index": one_row.get("original_index", idx),
            "note": one_row.get("note", ""),
            "patient_id": patient_id,
            "criterion": criterion_id,
            "rationale":rationale_text,
            "medications":medications_text,
            "is_met": is_met_value,
            "true_label":true_label_value,
            "assessed_category": category,
            "assessor_confidence": confidence,
            "assessor_reason": reason_text,
            "assessor_evidence_ids": json.dumps(evidence_ids_used, ensure_ascii=False),
            "used_k": args.k,
            "retrieved_count": retrieved_count,
            "latency_ms": latency_ms,
            "error": error_message,
            "llm_status": result["status"],
            "llm_error_type": result.get("error_type", ""),
            "llm_attempts": result.get("attempts", 0),
        }
        output_buffer.append(output_row)
        if len(output_buffer) >= 50:
            append_rows(output_csv_path, fieldnames, output_buffer)
            output_buffer = []


    # write output 
    if output_buffer:
        append_rows(output_csv_path, fieldnames, output_buffer)

    # Report
    report_path = output_csv_path.replace(".csv", "_report.txt")
    write_report(report_path, args.model, stats)

    print(f"[done] results written to {output_csv_path}")
    print(f"[done] report written to {report_path}")

if __name__ == "__main__":
    print("Starting LLM-as-a-judge analysis ....")

    main()