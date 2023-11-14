# TODO: Clean this mess up.
import argparse
import json
import os
from typing import Dict

import kaggle as kg
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def arguments():
    ap = argparse.ArgumentParser()
    # Start exclusive argument group
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "-d",
        "--dataset",
        default="./car_evaluation.csv",
        help="path to input dataset",
    )
    g.add_argument(
        "-k",
        "--kaggle_dataset",
        default="elikplim/car-evaluation-data-set",
        help="path to input dataset",
    )
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("-m", "--method", default="entropy", help="path to input dataset")
    return ap.parse_args()


def calc_entropy(y: pd.Series):
    # Calculate Distributions of items in Y
    dist = y.value_counts(normalize=True)
    # Calculate entropy
    entropy = -np.sum(dist * np.log2(dist))
    # Calculate Entropy After every slipt
    return entropy


def create_tree(df: pd.DataFrame):
    # Get Label and Features
    y = df["class"]
    x = df.drop(["class"], axis=1)

    # Feature ranges
    value_ranges = {}
    cols = x.columns
    for col in cols:
        value_ranges[col] = x[col].unique().tolist()

    # Split em
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, random_state=420)
    # Start the tree creation
    tree = {}
    print("Creating tree")
    # Build Tree
    tree_dict = split_tree(xtr, ytr, 0, "root", value_ranges)  # type:ignore
    # Make it to json
    # tree_json = json.dumps(tree_dict, sort_keys=True, indent=4, separators=(",", ": "))
    # print(tree_json) #  🐞
    # Evaluate
    print("Evaluating Tree")
    evaluator(xte, yte, tree_dict)


def eval(tree_dict: Dict, sample: pd.Series, label: str):
    cur_key = list(tree_dict.keys())[0]
    sample_val = sample[cur_key]
    tree_or_val = tree_dict[cur_key][sample_val]
    if isinstance(tree_or_val, str):
        return label == tree_or_val
    # Get Trees first key:
    return eval(tree_or_val, sample, label)


def evaluator(x_te, y_te, tree_dict: Dict[str, Dict]):
    correct = 0
    # JOintly iterate through x_te dataframe and y_te series:
    for (_, x), y in zip(x_te.iterrows(), y_te):
        correct += eval(tree_dict, x, y)
    print(f"Final evaluation is {correct/len(x_te)}")


def split_tree(x: pd.DataFrame, y: pd.Series, depth, parent="root", value_ranges={}):
    # Terminating Condition

    # Non Terminating
    overall_entropy = calc_entropy(y)
    avg_split_entropy = {}
    if overall_entropy < 0.1:
        max_val = y.value_counts().idxmax()
        # print(str(" " * 4 * (depth + 2)) + "`-> " + max_val)  # 🪲
        return max_val

    # Calculate entropy in split
    for col in x.columns:
        unique_vals = value_ranges[col]
        y_idxs = {branch: y[(x[col] == branch).values] for branch in unique_vals}
        avg_split_entropy[col] = np.mean(
            np.array([calc_entropy(yidx) for _, yidx in y_idxs.items()])
        )

    reductions = {k: overall_entropy - v for k, v in avg_split_entropy.items()}
    # Select key with max reduction from reductions
    # Get max reduction and its key
    max_val = max(list(reductions.values()))
    max_reduction_feat = max(reductions, key=reductions.get)  # type: ignore
    cur_dict = {max_reduction_feat: {}}  # 💫

    branches = value_ranges[max_reduction_feat]

    # Non Terminating Condition
    for branch in branches:
        relevant_ys = y[(x[max_reduction_feat] == branch).values]
        if len(x.columns) == 1:
            # Just check majority per branch
            if len(relevant_ys) > 0:
                cur_dict[max_reduction_feat][
                    branch
                ] = relevant_ys.value_counts().idxmax()
            else:  # TODO: Need to do something about this
                cur_dict[max_reduction_feat][branch] = np.random.choice(
                    ["acc", "unacc"]
                )
            # print(
            # str(" " * 4 * (depth + 2))
            # + '`-> "'
            # + branch
            # + '" : '
            # + cur_dict[max_reduction_feat][branch]
            # )🪲
        else:
            if len(relevant_ys) == 0:
                cur_dict[max_reduction_feat][branch] = np.random.choice(
                    ["acc", "unacc"]
                )
            else:
                relevant_xs = x[(x[max_reduction_feat] == branch).values]
                cur_dict[max_reduction_feat][branch] = split_tree(
                    relevant_xs.drop([max_reduction_feat], axis=1),  # type: ignore
                    relevant_ys,
                    depth=depth + 1,
                    parent=branch,
                    value_ranges=value_ranges,
                )

    return cur_dict


""" What I image the tree will look like as ds
buying:
    node1-cat1: 
        node11-maint:,
    node2-cat2: doors,
    node3-cat3: doors,
    node4-cat4: safet
"""


def dealt_with_ds(args: argparse.Namespace):
    if not os.path.exists(args.dataset) and args.kaggle_dataset:
        kg.api.authenticate()
        kg.api.dataset_download_files(args.kaggle_dataset, path="./", unzip=True)
        print("Downloaded file")
    else:
        args.dataset = args.dataset


if __name__ == "__main__":
    args = arguments()
    # Evaluate if you need to sign in. To download
    dealt_with_ds(args)

    col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    # Load Dataset
    df = pd.read_csv(args.dataset)
    df.columns = col_names
    # Print head of dataset
    print(f"Dataset has {len(df.columns)} columns and the head looks like:")
    print(df.head())
    print("\n----------------------------------------\n")

    # Check what categorical values we have in each column
    for col in df.columns:
        print(f"{col} has {df[col].unique()} with null_count {df[col].isnull().sum()}")
    # Take note of said values
    # We will predict buying class

    # Create blank datastructure for treeo
    tree = {}
    # Start the tree splitting
    tree = create_tree(df)
    print("Done")
