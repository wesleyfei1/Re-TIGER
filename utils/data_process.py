import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Paths
base_dir = Path("~/datasets").expanduser()
reviews_path = base_dir / "Beauty.jsonl"
meta_path = base_dir / "meta_Beauty.jsonl"
output_dir = base_dir / "processed" / "beauty"
output_dir.mkdir(parents=True, exist_ok=True)

# Parameters
min_user_interactions = 5


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def flatten_categories(categories):
    if not categories:
        return ""
    if isinstance(categories, list):
        # Amazon meta often uses list of lists
        flat = []
        for entry in categories:
            if isinstance(entry, list):
                flat.extend(entry)
            else:
                flat.append(entry)
        # Remove empties while preserving order
        seen = set()
        cleaned = []
        for item in flat:
            item = str(item).strip()
            if item and item not in seen:
                cleaned.append(item)
                seen.add(item)
        return " > ".join(cleaned)
    return str(categories).strip()


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()


# Load reviews and group by user
user_interactions = defaultdict(list)
for r in read_jsonl(reviews_path):
    user = r.get("user_id") or r.get("reviewerID")
    item = r.get("parent_asin") or r.get("asin")
    ts = r.get("timestamp") or r.get("unixReviewTime")
    if not user or not item or ts is None:
        continue
    # Handle millisecond timestamps
    ts_int = int(ts)
    if ts_int > 10_000_000_000:
        ts_int = ts_int // 1000
    user_interactions[user].append((ts_int, item))

# Filter users with at least min_user_interactions
user_interactions = {
    u: items
    for u, items in user_interactions.items()
    if len(items) >= min_user_interactions
}

# Leave-one-out split
train_records = []
valid_records = []
test_records = []
all_items = set()

for user, items in user_interactions.items():
    items_sorted = sorted(items, key=lambda x: x[0])
    if len(items_sorted) < 3:
        continue
    *train_items, valid_item, test_item = items_sorted
    for ts, item in train_items:
        train_records.append({"user_id": user, "item_id": item, "timestamp": ts})
        all_items.add(item)
    valid_records.append({"user_id": user, "item_id": valid_item[1], "timestamp": valid_item[0]})
    test_records.append({"user_id": user, "item_id": test_item[1], "timestamp": test_item[0]})
    all_items.add(valid_item[1])
    all_items.add(test_item[1])

# Build item text from meta
item_text_records = []
meta_index = {}
for m in read_jsonl(meta_path):
    parent_asin = m.get("parent_asin") or m.get("asin")
    if parent_asin:
        meta_index[parent_asin] = m

for item_id in sorted(all_items):
    meta = meta_index.get(item_id, {})
    title = normalize_text(meta.get("title"))
    brand = normalize_text(meta.get("brand") or meta.get("store"))
    price = normalize_text(meta.get("price"))
    category = ""
    if "categories" in meta:
        category = flatten_categories(meta.get("categories"))
    elif "category" in meta:
        category = normalize_text(meta.get("category"))
    elif "main_category" in meta:
        category = normalize_text(meta.get("main_category"))
    text = " | ".join([p for p in [title, brand, price, category] if p])
    item_text_records.append({"item_id": item_id, "text": text})

# Save outputs
write_jsonl(output_dir / "train.jsonl", train_records)
write_jsonl(output_dir / "valid.jsonl", valid_records)
write_jsonl(output_dir / "test.jsonl", test_records)
write_jsonl(output_dir / "item_text.jsonl", item_text_records)

stats = {
    "num_users": len(user_interactions),
    "num_items": len(all_items),
    "num_train": len(train_records),
    "num_valid": len(valid_records),
    "num_test": len(test_records),
    "min_user_interactions": min_user_interactions,
    "reviews_path": str(reviews_path),
    "meta_path": str(meta_path),
    "generated_at": datetime.now(timezone.utc).isoformat(),
}
(output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

print("Done. Output dir:", output_dir)
print(stats)
