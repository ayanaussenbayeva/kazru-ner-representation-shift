"""Re-split the annotated complaints into train/test with multi-label stratification.

The published split placed 34 complaints with STREET_NAME in train and 167 in test,
although it was meant to be stratified. This script pools train.jsonl and test.jsonl
and re-splits them 70/30 with iterative stratification (Sechidis et al., 2011), so that
each entity type appears in both splits in roughly the target proportion.

Usage:
    python make_stratified_split.py --train train.jsonl --test test.jsonl \
        --out_train train_strat.jsonl --out_test test_strat.jsonl --test_size 0.3 --seed 42
"""
import argparse
import json
import random
from collections import Counter

ID2LABEL = {
    0: "O",
    1: "B-ROUTE_NUM", 2: "I-ROUTE_NUM",
    3: "B-BUS_ID", 4: "I-BUS_ID",
    5: "B-PLATE_NUM", 6: "I-PLATE_NUM",
    7: "B-STOP_NAME", 8: "I-STOP_NAME",
    9: "B-STREET_NAME", 10: "I-STREET_NAME",
    11: "B-STREET_INTERSECTION", 12: "I-STREET_INTERSECTION",
}


def entity_types(item):
    """Set of entity types present in a complaint (from its B- labels)."""
    return {ID2LABEL[l][2:] for l in item["labels"] if l != -100 and ID2LABEL[l].startswith("B-")}


def iterative_stratification(label_sets, ratios, seed=42):
    """Assign each example to a fold so that every label follows the target ratios.

    label_sets: list of sets of labels, one per example
    ratios:     list of target fractions per fold (e.g. [0.7, 0.3])
    returns:    list of fold indices, one per example
    """
    rnd = random.Random(seed)
    n = len(label_sets)
    folds = [None] * n
    desired_total = [r * n for r in ratios]
    label_counts = Counter(l for s in label_sets for l in s)
    desired = {l: [r * c for r in ratios] for l, c in label_counts.items()}
    remaining = set(range(n))

    def pick_fold(candidates_score):
        best = max(candidates_score)
        ties = [k for k, v in enumerate(candidates_score) if v == best]
        if len(ties) > 1:
            best_total = max(desired_total[k] for k in ties)
            ties = [k for k in ties if desired_total[k] == best_total]
        return rnd.choice(ties)

    while True:
        # label with the fewest remaining examples
        rem_counts = Counter(l for i in remaining for l in label_sets[i])
        if not rem_counts:
            break
        min_count = min(rem_counts.values())
        label = rnd.choice(sorted(l for l, c in rem_counts.items() if c == min_count))
        examples = [i for i in remaining if label in label_sets[i]]
        rnd.shuffle(examples)
        for i in examples:
            k = pick_fold(desired[label])
            folds[i] = k
            remaining.discard(i)
            desired_total[k] -= 1
            for l in label_sets[i]:
                desired[l][k] -= 1

    # examples without any entity: fill folds by remaining capacity
    rest = list(remaining)
    rnd.shuffle(rest)
    for i in rest:
        k = pick_fold(desired_total)
        folds[i] = k
        desired_total[k] -= 1
    return folds


def report(name, items):
    c = Counter(t for it in items for t in entity_types(it))
    return f"{name:6s} n={len(items):5d}  " + "  ".join(f"{t}={c[t]}" for t in sorted(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--test", default="test.jsonl")
    ap.add_argument("--out_train", default="train_strat.jsonl")
    ap.add_argument("--out_test", default="test_strat.jsonl")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    load = lambda p: [json.loads(line) for line in open(p, encoding="utf-8")]
    old_train, old_test = load(args.train), load(args.test)
    print("Original split (complaints containing each entity type):")
    print(report("train", old_train))
    print(report("test", old_test))

    pooled = old_train + old_test
    folds = iterative_stratification([entity_types(it) for it in pooled],
                                     [1 - args.test_size, args.test_size], seed=args.seed)
    new_train = [it for it, f in zip(pooled, folds) if f == 0]
    new_test = [it for it, f in zip(pooled, folds) if f == 1]

    print("\nNew stratified split:")
    print(report("train", new_train))
    print(report("test", new_test))

    for path, items in [(args.out_train, new_train), (args.out_test, new_test)]:
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\nSaved {args.out_train} and {args.out_test}")


if __name__ == "__main__":
    main()
