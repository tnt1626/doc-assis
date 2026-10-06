"""Script to regrade benchmark result JSONL files using standard Python libraries only.

Does NOT modify original result files, does NOT rerun LLM models.
Outputs summary to terminal, manual review list to `evals/results/regrade_manual.md`,
and regraded dataset to `evals/results/regraded.jsonl`.
"""
import argparse
import glob
import json
from pathlib import Path
import re
import sys
import unicodedata

# CJK Character Regex (Chinese/Japanese/Korean)
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")

UNKNOWN_PHRASES = [
    "không biết", "chưa biết", "không rõ", "chưa nói", "chưa cho mình biết",
    "chưa được cung cấp", "không có thông tin", "chưa có thông tin",
    "không được đề cập", "không chứa thông tin", "không thấy",
    "không có tài liệu", "không có bất kỳ tài liệu"
]

ROOT = Path(__file__).resolve().parent.parent


def norm(text: str | None) -> str:
    """Normalize text for string matching: Unicode NFC + lowercase."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text).lower()


def regrade_answer(scenario: str, answer: str | None) -> tuple[bool, bool]:
    """Regrade response according to scenario rules.
    
    Returns:
        tuple[bool, bool]: (passed_new, manual_flag)
    """
    if answer is None:
        return False, False

    a = norm(answer)
    manual_flag = False

    if scenario == "recall_exam":
        passed = any(k in a for k in ["cơ sở dữ liệu", "csdl", "database"])
    elif scenario == "recall_name":
        passed = "tâm" in a
    elif scenario == "recall_year":
        passed = any(k in a for k in ["năm ba", "năm 3", "năm thứ ba", "năm thứ 3"])
    elif scenario == "update_exam":
        has_new = "mạng máy tính" in a
        has_old = ("cơ sở dữ liệu" in a) or ("csdl" in a)
        passed = has_new
        if has_new and has_old:
            manual_flag = True
    elif scenario == "unknown_exam":
        passed = any(k in a for k in UNKNOWN_PHRASES)
    elif scenario == "recall_project":
        manual_flag = True
        if "mosaic" in a or "paper" in a:
            passed = False
        else:
            passed = any(k in a for k in ["chi tiêu", "hỏi đáp"])
    else:
        passed = False

    return passed, manual_flag


def check_language(answer: str | None) -> bool | None:
    """Check if answer is free of CJK character pollution."""
    if answer is None:
        return None
    return CJK_RE.search(answer) is None


def avg(values: list[float | int | None]) -> float | None:
    """Compute average for non-None numbers."""
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def deduplicate_file_rows(raw_rows: list[dict], filename: str) -> list[dict]:
    """Deduplicate rows by (scenario, rep) key, keeping the LAST occurrence."""
    seen: dict[tuple[str, int], list[dict]] = {}
    order: list[tuple[str, int]] = []

    for row in raw_rows:
        key = (row["scenario"], row["rep"])
        if key not in seen:
            seen[key] = []
            order.append(key)
        seen[key].append(row)

    deduped = []
    for key in order:
        group = seen[key]
        if len(group) > 1:
            dropped = len(group) - 1
            print(f"[CẢNH BÁO] [{filename}] Đã bỏ {dropped} dòng trùng: {key[0]} rep {key[1]}")
        deduped.append(group[-1])

    return deduped


def parse_token_averages(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Calculate token usage averages per call purpose across rows."""
    purpose_stats: dict[str, dict[str, list[int]]] = {}

    for r in rows:
        tt = r.get("tokens_total")
        if not isinstance(tt, dict):
            continue
        for purpose, metrics in tt.items():
            if not isinstance(metrics, dict):
                continue
            if purpose not in purpose_stats:
                purpose_stats[purpose] = {"calls": [], "prompt": [], "completion": []}
            purpose_stats[purpose]["calls"].append(metrics.get("calls", 0))
            purpose_stats[purpose]["prompt"].append(metrics.get("prompt", 0))
            purpose_stats[purpose]["completion"].append(metrics.get("completion", 0))

    if not purpose_stats:
        return {}

    n = len(rows)
    return {
        purpose: {
            "calls": sum(stat["calls"]) / n,
            "prompt": sum(stat["prompt"]) / n,
            "completion": sum(stat["completion"]) / n,
        }
        for purpose, stat in purpose_stats.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Regrade benchmark result JSONL files.")
    parser.add_argument("files", nargs="*", help="JSONL files to regrade (default: evals/results/*.jsonl excluding rerun/regraded/recall2)")
    parser.add_argument("--rerun", action="append", nargs="*", help="Rerun JSONL files to replace error rows")
    parser.add_argument("--replace-scenario", type=str, help="Scenario name to replace completely with rows from --replace-file")
    parser.add_argument("--replace-file", action="append", nargs="*", help="File(s) containing replacement rows for --replace-scenario")
    args = parser.parse_args()

    if args.files:
        filepaths = [Path(f) for f in args.files]
    else:
        filepaths = [
            Path(f) for f in sorted(glob.glob("evals/results/*.jsonl"))
            if not f.endswith("regraded.jsonl") and "-rerun" not in Path(f).name and "_rerun" not in Path(f).name and "-recall2" not in Path(f).name and "_recall2" not in Path(f).name
        ]

    if not filepaths:
        print("Không tìm thấy file jsonl nào để chấm lại.", file=sys.stderr)
        return 1

    # Load replace files for scenario replacement
    replace_map: dict[tuple[str, str], list[dict]] = {}
    replace_paths: list[str] = []
    if args.replace_file:
        for item in args.replace_file:
            if isinstance(item, list):
                replace_paths.extend(item)
            elif isinstance(item, str):
                replace_paths.append(item)

    if args.replace_scenario and replace_paths:
        target_sc = args.replace_scenario
        for rpath in replace_paths:
            p = Path(rpath)
            if not p.exists():
                print(f"[CẢNH BÁO] File replace không tồn tại: {p}", file=sys.stderr)
                continue
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rrow = json.loads(line)
                            lbl = rrow.get("label")
                            sc = rrow.get("scenario")
                            if lbl and sc == target_sc:
                                key = (lbl, sc)
                                if key not in replace_map:
                                    replace_map[key] = []
                                replace_map[key].append(rrow)
                        except json.JSONDecodeError:
                            continue

    # Load rerun files for error row replacement
    rerun_paths: list[str] = []
    if args.rerun is not None:
        for item in args.rerun:
            if isinstance(item, list):
                rerun_paths.extend(item)
            elif isinstance(item, str):
                rerun_paths.append(item)
    else:
        rerun_paths = sorted([
            f for f in glob.glob("evals/results/*rerun*.jsonl")
            if "-recall2" not in f and "_recall2" not in f
        ])

    rerun_queues: dict[tuple[str, str], list[dict]] = {}
    if rerun_paths:
        for rpath in rerun_paths:
            p = Path(rpath)
            if not p.exists():
                print(f"[CẢNH BÁO] File rerun không tồn tại: {p}", file=sys.stderr)
                continue
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rrow = json.loads(line)
                            lbl = rrow.get("label")
                            sc = rrow.get("scenario")
                            if lbl and sc:
                                key = (lbl, sc)
                                if key not in rerun_queues:
                                    rerun_queues[key] = []
                                rerun_queues[key].append(rrow)
                        except json.JSONDecodeError:
                            continue

    all_regraded_rows: list[dict] = []
    file_summaries: list[dict] = []
    manual_review_cases: list[dict] = []

    for path in filepaths:
        if not path.exists():
            print(f"File không tồn tại: {path}", file=sys.stderr)
            continue

        raw_rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        raw_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not raw_rows:
            print(f"File rỗng hoặc không đúng định dạng jsonl: {path}")
            continue

        filename = path.name
        label = raw_rows[0].get("label", path.stem)

        # Step 1: Deduplication
        deduped = deduplicate_file_rows(raw_rows, filename)

        # Apply replacements (scenario replacement + rerun error replacement)
        replaced_log: list[str] = []
        for idx, r in enumerate(deduped):
            r_sc = r.get("scenario")
            r_lbl = r.get("label", label)

            # Priority 1: Replace scenario completely if --replace-scenario is matched
            if args.replace_scenario and r_sc == args.replace_scenario:
                key = (r_lbl, r_sc)
                if key in replace_map and len(replace_map[key]) > 0:
                    replacement_raw = replace_map[key].pop(0)
                    replacement = dict(replacement_raw)
                    orig_rep = r.get("rep")
                    replacement["rep"] = orig_rep
                    replacement["replaced_scenario"] = True
                    deduped[idx] = replacement

                    replaced_log.append(
                        f"  - THAY THẾ TOÀN BỘ SCENARIO '{r_sc}' (rep={orig_rep}) bằng dữ liệu từ replace-file"
                    )
                    continue

            # Priority 2: Rerun error row replacement for other scenarios
            if r.get("error") is not None:
                key = (r_lbl, r_sc)
                if key in rerun_queues and len(rerun_queues[key]) > 0:
                    replacement_raw = rerun_queues[key].pop(0)
                    replacement = dict(replacement_raw)
                    orig_rep = r.get("rep")
                    replacement["rep"] = orig_rep
                    replacement["rerun"] = True
                    deduped[idx] = replacement

                    err_status = f"lỗi mới ({replacement.get('error')})" if replacement.get("error") else "thành công"
                    replaced_log.append(
                        f"  - Thay thế dòng bị lỗi (scenario='{r_sc}', rep={orig_rep}) bằng dòng rerun -> {err_status}"
                    )

        # Steps 2-4: Error handling, Regrading & Language check
        processed_rows = []
        for r in deduped:
            passed_new, manual_flag = regrade_answer(r.get("scenario", ""), r.get("answer"))
            lang_ok_new = check_language(r.get("answer"))

            row_regraded = dict(r)
            row_regraded["passed_new"] = passed_new
            row_regraded["manual_flag"] = manual_flag
            row_regraded["language_ok_new"] = lang_ok_new
            processed_rows.append(row_regraded)

            if manual_flag:
                manual_review_cases.append(row_regraded)

        all_regraded_rows.extend(processed_rows)

        # Statistics computation
        total_rows = len(processed_rows)
        error_rows = [r for r in processed_rows if r.get("error") is not None]
        valid_rows = [r for r in processed_rows if r.get("error") is None]

        kind_stats: dict[str, dict[str, int]] = {}
        for r in valid_rows:
            k = r.get("kind", "other")
            if k not in kind_stats:
                kind_stats[k] = {"passed": 0, "total": 0}
            kind_stats[k]["total"] += 1
            if r["passed_new"]:
                kind_stats[k]["passed"] += 1

        total_valid = len(valid_rows)
        total_passed = sum(1 for r in valid_rows if r["passed_new"])
        lang_failures = sum(1 for r in valid_rows if r.get("language_ok_new") is False)

        avg_question_s = avg([r.get("final_question_s") for r in processed_rows])
        avg_consolidation_s = avg([r.get("consolidation_s") for r in processed_rows])
        token_stats = parse_token_averages(processed_rows)

        file_summaries.append({
            "label": label,
            "filename": filename,
            "total_rows": total_rows,
            "total_valid": total_valid,
            "total_passed": total_passed,
            "error_count": len(error_rows),
            "lang_failures": lang_failures,
            "kind_stats": kind_stats,
            "avg_question_s": avg_question_s,
            "avg_consolidation_s": avg_consolidation_s,
            "token_stats": token_stats,
        })

        # Step 5: Screen output for current file
        print("\n" + "=" * 80)
        print(f"FILE: {path} (Label: {label})")
        print("=" * 80)
        if replaced_log:
            print("Danh sách các dòng đã được thay thế:")
            for log_entry in replaced_log:
                print(log_entry)
            print("-" * 80)

        print(f"Số dòng sau khi loại trùng : {total_rows}")
        print(f"Số dòng bị lỗi (error)     : {len(error_rows)}")
        print("\nBảng kết quả theo loại kịch bản (Đúng / Tổng số ca không lỗi):")
        for k in sorted(kind_stats.keys()):
            ks = kind_stats[k]
            print(f"  {k:<10} : {ks['passed']}/{ks['total']}")
        print(f"  {'TỔNG CHUNG':<10} : {total_passed}/{total_valid}")
        print(f"\nSố ca trượt ngôn ngữ (chứa CJK): {lang_failures}")

        q_s_str = f"{avg_question_s:.2f}s" if avg_question_s is not None else "n/a"
        c_s_str = f"{avg_consolidation_s:.2f}s" if avg_consolidation_s is not None else "n/a"
        print(f"Thời gian trung bình câu hỏi (final_question_s) : {q_s_str}")
        print(f"Thời gian trung bình xóa session (consolidation_s): {c_s_str}")

        print("\nDữ liệu token trung bình:")
        if token_stats:
            for p, st in token_stats.items():
                print(f"  {p:<18} calls={st['calls']:.1f}  prompt={st['prompt']:.0f}  completion={st['completion']:.0f}")
        else:
            print("  không có dữ liệu token")

    # Step 5 (cont): Overall Summary Comparison Table
    if file_summaries:
        print("\n" + "=" * 80)
        print("TỔNG HỢP SO SÁNH CÁC FILE")
        print("=" * 80)
        print(f"{'Label':<20} | {'Đúng / Tổng (Không lỗi)':<24} | {'Trượt ngôn ngữ':<15} | {'Lỗi':<6}")
        print("-" * 80)
        for fs in file_summaries:
            ratio = f"{fs['total_passed']}/{fs['total_valid']}"
            print(f"{fs['label']:<20} | {ratio:<24} | {fs['lang_failures']:<15} | {fs['error_count']:<6}")
        print("=" * 80)

    # Print all unknown_exam scenarios detail
    unknown_rows = [r for r in all_regraded_rows if r.get("scenario") == "unknown_exam"]
    if unknown_rows:
        print("\n" + "=" * 80)
        print("DANH SÁCH CHI TIẾT CÁC CA UNKNOWN_EXAM:")
        print("=" * 80)
        for r in unknown_rows:
            lbl = r.get("label", "n/a")
            rep = r.get("rep", "n/a")
            p_new = r.get("passed_new")
            err = r.get("error")
            ans = r.get("answer") or f"(ERROR: {err})"
            ans_snippet = ans[:120].replace("\n", " ")
            status_str = "ĐÚNG" if p_new else "SAI "
            is_rerun_str = " [RERUN]" if r.get("rerun") else ""
            print(f"Label: {lbl:<15} | Rep: {rep} | Result: {status_str}{is_rerun_str:<8} | Answer: {repr(ans_snippet)}")
        print("=" * 80)

    # Step 6: Manual Review List (stdout + evals/results/regrade_manual.md)
    manual_md_path = ROOT / "evals" / "results" / "regrade_manual.md"
    manual_md_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"DANH SÁCH ĐỌC TAY (Manual Review Cases): {len(manual_review_cases)} ca")
    print("=" * 80)

    md_lines = [
        "# Danh sách các ca cần ĐỌC TAY (Manual Review Cases)\n",
        f"Tong so ca can kiem tra thu cong: {len(manual_review_cases)}\n",
        "---\n"
    ]

    for idx, r in enumerate(manual_review_cases, 1):
        lbl = r.get("label", "n/a")
        sc = r.get("scenario", "n/a")
        rep = r.get("rep", "n/a")
        q = r.get("question", "n/a")
        ans = r.get("answer") or "(Không có câu trả lời)"
        prelim = "ĐÚNG (sơ bộ)" if r.get("passed_new") else "SAI (sơ bộ)"
        is_rerun = " (dòng rerun)" if r.get("rerun") else ""
        is_replaced = " (thay thế scenario)" if r.get("replaced_scenario") else ""

        if sc == "recall_project":
            reason = "recall_project - Không chấm máy chắc chắn"
        elif sc == "update_exam":
            reason = "update_exam - Chứa cả thông tin môn học mới và cũ"
        else:
            reason = "Cần kiểm tra lại bằng tay"

        print(f"\n[{idx}/{len(manual_review_cases)}] Label: {lbl} | Scenario: {sc} | Rep: {rep}{is_rerun}{is_replaced}")
        print(f"  Lý do đọc tay  : {reason}")
        print(f"  Chấm sơ bộ     : {prelim}")
        print(f"  Câu hỏi        : {q}")
        print(f"  Câu trả lời    : {ans[:200]}..." if len(ans) > 200 else f"  Câu trả lời    : {ans}")

        md_lines.append(f"### {idx}. Label: `{lbl}` | Scenario: `{sc}` | Rep: `{rep}`{is_rerun}{is_replaced}\n")
        md_lines.append(f"- **Chấm sơ bộ**: `{prelim}`\n")
        md_lines.append(f"- **Lý do đọc tay**: {reason}\n")
        md_lines.append(f"- **Câu hỏi**: {q}\n")
        md_lines.append(f"- **Câu trả lời đầy đủ**:\n> {ans.replace('\n', '\n> ')}\n\n")
        md_lines.append("---\n")

    manual_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nĐã ghi danh sách đọc tay ra file: {manual_md_path}")

    # Output JSONL
    regraded_jsonl_path = ROOT / "evals/results/regraded.jsonl"
    regraded_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(regraded_jsonl_path, "w", encoding="utf-8") as fout:
        for r in all_regraded_rows:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Đã ghi toàn bộ dữ liệu chấm lại ra file: {regraded_jsonl_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
