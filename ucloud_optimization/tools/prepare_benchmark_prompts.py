# /// script
# requires-python = ">=3.12"
# ///
"""Select one file from each model-duplicate triplet and emit OpenAI chat JSONL.

Files differ only by historical target-model suffix. Group by filename after removing
`_llama3-70b`, `_llama3-8b`, `_mixtral-8x22b`; choose every third file in each sorted
group (default index 0), then flatten each file's list of message arrays.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from collections import defaultdict
from pathlib import Path

MODEL_SUFFIX = re.compile(r"_(?:llama3-70b|llama3-8b|mixtral-8x22b)(?=_V\d+_prompts\.json$)")

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--member", type=int, choices=(0,1,2), default=0,
                   help="member selected from each duplicate triplet (default: 0)")
    p.add_argument("--subset", type=int, default=0,
                   help="stratified subset size across prompt-length range; 0 keeps all")
    p.add_argument("--reasoning", action="store_true",
                   help="replace output constraint with long-form analysis request")
    args = p.parse_args()
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(args.source.rglob("*.json")):
        relative = str(path.relative_to(args.source))
        groups[MODEL_SUFFIX.sub("", relative)].append(path)

    selected, rows, seen = [], [], set()
    for key, files in sorted(groups.items()):
        # Some legacy 31500 groups contain only Mixtral. Keep them once.
        choice = files[min(args.member, len(files)-1)]
        selected.append(str(choice))
        data = json.loads(choice.read_text())
        if not isinstance(data, list):
            raise TypeError(f"{choice}: expected list")
        for messages in data:
            if not isinstance(messages, list) or not messages:
                raise TypeError(f"{choice}: expected list of message arrays")
            canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False)
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            rows.append({"messages": messages, "source": str(choice), "sha256": digest,
                         "chars": sum(len(str(m.get("content", ""))) for m in messages)})

    # Deterministic length-stratified sample: approximately evenly spaced order
    # statistics. This includes short/medium/long requests without random luck.
    if args.subset and args.subset < len(rows):
        ordered = sorted(rows, key=lambda x: (x["chars"], x["sha256"]))
        indices = [round(i * (len(ordered) - 1) / (args.subset - 1))
                   for i in range(args.subset)] if args.subset > 1 else [len(ordered)//2]
        rows = [ordered[i] for i in indices]

    if args.reasoning:
        suffix = ("\n\nFor this benchmark, ignore the earlier instruction to answer with only an option. "
                  "Reason carefully from the supplied evidence. Compare all five options, identify "
                  "supporting and conflicting evidence, discuss uncertainty and plausible alternative "
                  "interpretations, then give a final choice. Produce a detailed analysis of at least "
                  "1,000 words; do not use a fixed schema.")
        for row in rows:
            # Each row remains one independent request. Modify only its final user turn.
            msgs = row["messages"]
            user = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
            if user is None:
                raise ValueError(f"No user message in {row['source']}")
            user["content"] = str(user.get("content", "")) + suffix

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {"source": str(args.source), "member": args.member, "files": selected,
                "groups": len(groups), "unique_prompts": len(rows),
                "subset": args.subset, "reasoning": args.reasoning,
                "char_range": [min(r["chars"] for r in rows), max(r["chars"] for r in rows)]}
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__": main()
