import tiktoken
from typing import List, Dict
from openai import OpenAI
import os
import numpy as np
from numpy.linalg import norm
import pandas as pd
import json


def count_tokens(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

# find cosine similarity of every chunk to a given embedding
def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

# Prompt builder
def userMsg(*args) -> dict:
    return {"role": "user", "content": "\n".join(args)}
def assistantMsg(*args) -> dict:
    return {"role": "assistant", "content": "\n".join(args)}
def systemMsg(*args) -> dict:
    return {"role": "system", "content": "\n".join(args)}


def calc_MAE(prediction: pd.Series, target: pd.Series):
    return (prediction - target).abs().sum() / len(prediction)

def unclutterSignature(baseSign: str):
    remove_list = ('-29', '_V7')
    for remove_item in remove_list:
        baseSign = baseSign.replace(remove_item, '')
    return baseSign



def describe_prompts(final_prompts: List[List[Dict]]) -> Dict:
    total_all_prompt_tokens = 0 #used elsewhere too
    prompt_tokens_min = 0
    prompt_tokens_max = 0
    for p in final_prompts:
        pt = 0 # Prompt tokens
        for msg in p:
            pt += count_tokens(msg["content"])
        if prompt_tokens_min == 0 or pt < prompt_tokens_min:
            prompt_tokens_min = pt
        if pt > prompt_tokens_max:
            prompt_tokens_max = pt

        total_all_prompt_tokens += pt

    return {
        'average_prompt_tokens': round(total_all_prompt_tokens/len(final_prompts)), 
        "total_all_prompt_tokens": total_all_prompt_tokens,
        "prompt_tokens_min": prompt_tokens_min,
        "prompt_tokens_max": prompt_tokens_max
    }

def describe_prompts_and_print(final_prompts: List[List[Dict]]) -> Dict:
    info = describe_prompts(final_prompts)

    print(f"Created {len(final_prompts)} prompts.")
    print(f"Average prompt size: {round(info['total_all_prompt_tokens']/len(final_prompts))} tokens.")
    print(f"Min prompt size: {info['prompt_tokens_min']}, Max prompt size: {info['prompt_tokens_max']}")
    return info

def bring_to_front_important_columns(df: pd.DataFrame, cols_to_front: List[str]):
    # Reorder columns by notna counts
    non_nan_counts = df.notna().sum()
    sorted_columns = non_nan_counts.sort_values(ascending=False).index
    df = df[sorted_columns]

    # Bring to front the columns that are important
    new_order = [col for col in cols_to_front if col in df.columns] + [col for col in df.columns if col not in cols_to_front]
    df = df[new_order]
    return df

def load_sim(path) -> Dict:
    """Simply loads a simulation .json file to a dictionary."""
    with open(path, 'r') as f:
        sim = json.load(f)
    return sim

def dataframe_from_QA(qa) -> pd.DataFrame:
    """Loads a simulation .json file to a pandas DataFrame. Does not include info data."""
    dfs = pd.DataFrame(qa)
    dfs.columns = dfs.iloc[0]  # Set the first row as the header
    dfs = dfs[1:]  

    return dfs

BLACKLIST_CHAT_REGEX_FILTERS = [
    {
        "id": "link-filter",
        "pattern": r"https.*"
    },
    {
        "id": "react-filter",
        "pattern": r"Reacted\s.*\sto\syour\smessage"
    },
    {
        "id": "danish-react-filter",
        "pattern": r"Har\sreageret\smed\s.*\spå\sdin\sbesked"
    },
    {
        "id": "cookie-data-filter",
        "pattern": r"\b\w{101,}\b"
    }
]

BLACKLIST_ANSWER_SUBSTRINGS = [
    r"\.", r"\!"
]

