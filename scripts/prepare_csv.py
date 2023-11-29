import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import random_split
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
logger = logging.getLogger(__name__)
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer
from chat.base import prompt_config

# COLUMNS = ("instruction", "input", "output")
COLUMNS = ("input", "output")


def csv2dataset(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if not (df.columns.values == COLUMNS).all():
        raise ValueError(f"CSV columns must be {COLUMNS}, found {df.columns.values}")
    return json.loads(df.to_json(orient="records", indent=4))


def prepare(
    csv_path: Path,
    destination_path: Path = Path("data/csv"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    test_split_fraction: float = 0.1,
    seed: int = 42,
    mask_inputs: bool = False,
    to_lower: bool = False,
    ignore_index: int = -1,
    test_csv_path: Optional[Path] = None,
    max_seq_length: Optional[int] = None,
) -> None:
    """Prepare a CSV dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    if max_seq_length is None:
        with open(checkpoint_dir / "lit_config.json", "r") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)
    logger.info("Loading data file ...")
    
    data = csv2dataset(csv_path)

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    system_prompt, _ = prompt_config(checkpoint_dir, tokenizer)
    print(system_prompt.format(prompt="Just a prompt example."))

    if test_csv_path is None:
        # Partition the dataset into train and test
        train_set, test_set = random_split(
            data, [1.0 - test_split_fraction, test_split_fraction], generator=torch.Generator().manual_seed(seed)
        )
    else:
        train_set = random_split(data, [1.0], generator=torch.Generator().manual_seed(seed))[0]  # include shuffling training data.
        test_set = random_split(csv2dataset(test_csv_path), [1.0], generator=torch.Generator().manual_seed(seed))[0]

    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"test has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_sample(
            system_prompt=system_prompt,
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
            to_lower=to_lower,
        )
        for sample in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_sample(
            system_prompt=system_prompt,
            example=sample,
            tokenizer=tokenizer,
            max_length=max_seq_length,
            mask_inputs=mask_inputs,
            ignore_index=ignore_index,
            to_lower=to_lower,
        )
        for sample in tqdm(test_set)
    ]
    torch.save(test_set, destination_path / "test.pt")


def prepare_sample(
    system_prompt: str,
    example: dict, tokenizer: Tokenizer, max_length: int,
    mask_inputs: bool, ignore_index: int, to_lower: bool
) -> dict:
    """Processes a single sample.

    Each sample in the dataset consists of:
    - instruction: A string describing the task
    - input: A string holding a special input value for the instruction.
        This only applies to some samples, and in others this is empty.
    - output: The response string

    This function processes this data to produce a prompt text and a label for
    supervised training. The prompt text is formed as a single message including both
    the instruction and the input. The label/target is the same message but with the
    response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens
    in the label that correspond to the original input prompt get masked out (default).
    """
    if to_lower:
        example['input'] = example['input'].lower()
        example['output'] = example['output'].lower()
    full_prompt = generate_prompt(system_prompt, example)
    full_prompt_and_response = full_prompt + example["output"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


def generate_prompt(system_prompt:str, example: dict) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return system_prompt.format(prompt=example['input'])
    else:
        return system_prompt


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare, as_positional=False)

    # prompt = 'a photo of a ML engineer'
    # print(system_prompt.format(prompt=prompt))
    # print(generate_prompt({'input': prompt}))
