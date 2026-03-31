import json
import os
import shutil
import uuid
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
NEW_GROUP_DIR = BASE_DIR / "new_group"
ALIYUN_PATH = BASE_DIR / "aliyun_cm_problems.jsonl"
STUDENT_GROUP1_PATH = BASE_DIR / "student_group1.json"
STUDENT_GROUP2_PATH = BASE_DIR / "student_group2.json"
OJ_SAMPLES_PATH = BASE_DIR / "oj_samples.jsonl"
OJ_JSONL_PATH = BASE_DIR / "oj.jsonl"


def clean_code(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        first_newline = code.find("\n")
        if first_newline != -1:
            code = code[first_newline + 1 :]
        if code.endswith("```"):
            code = code[:-3]
    return code.strip()


def count_code_lines(code: str) -> int:
    return sum(1 for line in clean_code(code).splitlines() if line.strip())


def make_sample_id(label: str, domain: str, task: str, style: str, code: str) -> str:
    payload = "\n".join([label, domain, task, style, clean_code(code)])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


def source_to_label(source: str) -> str:
    source = source.strip().lower()
    if source in {"human", "student", "human_written"}:
        return "human"
    if source in {"ai", "ai_generated", "generated"}:
        return "ai"
    raise ValueError(f"Unsupported source: {source}")


def build_student_group2() -> list[dict]:
    records = []
    for path in sorted(NEW_GROUP_DIR.rglob("*.m")):
        if path.name == ".DS_Store":
            continue
        code = path.read_text(encoding="utf-8", errors="ignore")
        records.append(
            {
                "source": "human",
                "domain": "cm_problems",
                "task": path.parent.name,
                "style": "student",
                "code": clean_code(code),
            }
        )

    with open(STUDENT_GROUP2_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records


def load_json_array(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_record(record: dict) -> dict:
    code = clean_code(record["code"])
    label = source_to_label(record["source"])
    domain = record.get("domain", "unknown")
    task = record.get("task", "unknown")
    style = record.get("style", "unknown")
    return {
        "id": make_sample_id(label, domain, task, style, code),
        "label": label,
        "code": code,
        "code_lines": count_code_lines(code),
        "domain": domain,
        "task": task,
        "style": style,
    }


def build_oj_samples(student_group2_records: list[dict]) -> list[dict]:
    aliyun_records = load_jsonl(ALIYUN_PATH)
    student_group1_records = load_json_array(STUDENT_GROUP1_PATH)

    combined = []
    for record in aliyun_records + student_group1_records + student_group2_records:
        combined.append(normalize_record(record))

    with open(OJ_SAMPLES_PATH, "w", encoding="utf-8") as f:
        for record in combined:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    shutil.copyfile(OJ_SAMPLES_PATH, OJ_JSONL_PATH)
    return combined


def main():
    student_group2_records = build_student_group2()
    combined = build_oj_samples(student_group2_records)
    label_stats = {"ai": 0, "human": 0}
    for record in combined:
        label_stats[record["label"]] += 1

    print(f"student_group2.json: {len(student_group2_records)} records -> {STUDENT_GROUP2_PATH}")
    print(f"oj_samples.jsonl: {len(combined)} records -> {OJ_SAMPLES_PATH}")
    print(f"oj.jsonl: copied from oj_samples.jsonl -> {OJ_JSONL_PATH}")
    print(f"label distribution: {label_stats}")


if __name__ == "__main__":
    main()
