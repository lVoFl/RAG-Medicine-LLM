import argparse
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

try:
    from scipy import stats
except Exception:
    stats = None


METRICS = [
    "overall_score",
    "factual_consistency",
    "safety",
    "relevance",
    "completeness",
    "clarity",
    "boundary_awareness",
]

# 可按需要调整
WEIGHTS = {
    "factual_consistency": 0.35,
    "safety": 0.35,
    "completeness": 0.15,
    "relevance": 0.10,
    "clarity": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计多维评测 JSONL 中 base 和 dpo：均值、胜率、错误率、显著性与置信区间"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="model/test/aliyun_multidim_eval_details.jsonl",
        help="输入 JSONL 文件路径",
    )
    parser.add_argument("--bootstrap", type=int, default=3000, help="bootstrap 重采样次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=4.0,
        help="错误率阈值（默认 <4 计入风险）",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_ci_mean_diff(diffs: list[float], n_bootstrap: int, rng: random.Random) -> tuple[float, float]:
    if not diffs:
        return float("nan"), float("nan")
    n = len(diffs)
    samples = []
    for _ in range(n_bootstrap):
        draw = [diffs[rng.randrange(n)] for _ in range(n)]
        samples.append(mean(draw))
    samples.sort()
    return percentile(samples, 0.025), percentile(samples, 0.975)


def weighted_score(prefix: str, obj: dict) -> float | None:
    total = 0.0
    weight_sum = 0.0
    for m, w in WEIGHTS.items():
        v = obj.get(f"{prefix}_{m}")
        if isinstance(v, (int, float)):
            total += float(v) * w
            weight_sum += w
    if weight_sum == 0:
        return None
    return total / weight_sum


def safe_pvalue(metric: str, base_vals: list[float], dpo_vals: list[float]) -> tuple[float, float]:
    if stats is None or len(base_vals) != len(dpo_vals) or len(base_vals) < 2:
        return float("nan"), float("nan")

    diffs = [d - b for b, d in zip(base_vals, dpo_vals)]
    if all(abs(x) < 1e-12 for x in diffs):
        return 1.0, 1.0

    # 分数是离散有序，优先 Wilcoxon；同时给 paired t-test 作为参考
    try:
        w = stats.wilcoxon(dpo_vals, base_vals, zero_method="wilcox", alternative="two-sided")
        p_w = float(w.pvalue)
    except Exception:
        p_w = float("nan")

    try:
        t = stats.ttest_rel(dpo_vals, base_vals)
        p_t = float(t.pvalue)
    except Exception:
        p_t = float("nan")

    return p_w, p_t


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    total = 0
    winner_counter = Counter()

    base_vals = {m: [] for m in METRICS}
    dpo_vals = {m: [] for m in METRICS}
    diff_vals = {m: [] for m in METRICS}

    weighted_base = []
    weighted_dpo = []
    weighted_diff = []

    factual_base_risk = 0
    factual_dpo_risk = 0
    safety_base_risk = 0
    safety_dpo_risk = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行 JSON 解析失败: {exc}") from exc

            total += 1
            winner = obj.get("winner_model")
            if winner in ("base", "dpo"):
                winner_counter[winner] += 1

            for m in METRICS:
                b = obj.get(f"base_{m}")
                d = obj.get(f"dpo_{m}")
                if isinstance(b, (int, float)) and isinstance(d, (int, float)):
                    b = float(b)
                    d = float(d)
                    base_vals[m].append(b)
                    dpo_vals[m].append(d)
                    diff_vals[m].append(d - b)

            b_f = obj.get("base_factual_consistency")
            d_f = obj.get("dpo_factual_consistency")
            b_s = obj.get("base_safety")
            d_s = obj.get("dpo_safety")

            if isinstance(b_f, (int, float)) and float(b_f) < args.error_threshold:
                factual_base_risk += 1
            if isinstance(d_f, (int, float)) and float(d_f) < args.error_threshold:
                factual_dpo_risk += 1
            if isinstance(b_s, (int, float)) and float(b_s) < args.error_threshold:
                safety_base_risk += 1
            if isinstance(d_s, (int, float)) and float(d_s) < args.error_threshold:
                safety_dpo_risk += 1

            wb = weighted_score("base", obj)
            wd = weighted_score("dpo", obj)
            if wb is not None and wd is not None:
                weighted_base.append(wb)
                weighted_dpo.append(wd)
                weighted_diff.append(wd - wb)

    if total == 0:
        print("文件为空或没有有效记录。")
        return

    rng = random.Random(args.seed)

    print(f"输入文件: {input_path}")
    print(f"样本数: {total}")
    print(f"bootstrap: {args.bootstrap}, seed: {args.seed}\n")

    print("=== 胜场统计 ===")
    for model in ("base", "dpo"):
        wins = winner_counter[model]
        rate = wins / total * 100
        print(f"{model:>4}: {wins:>4} ({rate:.2f}%)")
    ties = total - winner_counter["base"] - winner_counter["dpo"]
    print(f" tie: {ties:>4} ({ties / total * 100:.2f}%)\n")

    print("=== 均值 + 配对差值显著性 + 95%CI (dpo-base) ===")
    header = (
        f"{'metric':<22} {'base':>8} {'dpo':>8} {'diff':>8} {'CI95%':>22} {'p_wilcoxon':>12} {'p_ttest':>10}"
    )
    print(header)
    print("-" * len(header))

    for m in METRICS:
        b = base_vals[m]
        d = dpo_vals[m]
        diffs = diff_vals[m]
        b_avg = mean(b)
        d_avg = mean(d)
        diff_avg = mean(diffs)
        ci_l, ci_u = bootstrap_ci_mean_diff(diffs, args.bootstrap, rng)
        p_w, p_t = safe_pvalue(m, b, d)

        print(
            f"{m:<22} {b_avg:>8.4f} {d_avg:>8.4f} {diff_avg:>8.4f} "
            f"[{ci_l:.4f}, {ci_u:.4f}] {p_w:>12.4g} {p_t:>10.4g}"
        )

    print("\n=== 风险错误率（< 阈值）===")
    print(f"阈值: {args.error_threshold}")
    print(
        f"factual_consistency: base {factual_base_risk}/{total} ({factual_base_risk/total*100:.2f}%), "
        f"dpo {factual_dpo_risk}/{total} ({factual_dpo_risk/total*100:.2f}%)"
    )
    print(
        f"safety             : base {safety_base_risk}/{total} ({safety_base_risk/total*100:.2f}%), "
        f"dpo {safety_dpo_risk}/{total} ({safety_dpo_risk/total*100:.2f}%)"
    )

    print("\n=== 加权分（抑制高分扎堆）===")
    print(
        "weights: factual 0.35, safety 0.35, completeness 0.15, relevance 0.10, clarity 0.05"
    )
    wb = mean(weighted_base)
    wd = mean(weighted_dpo)
    wdiff = mean(weighted_diff)
    ci_l, ci_u = bootstrap_ci_mean_diff(weighted_diff, args.bootstrap, rng)
    p_w, p_t = safe_pvalue("weighted", weighted_base, weighted_dpo)
    print(f"base: {wb:.4f}, dpo: {wd:.4f}, diff: {wdiff:.4f}")
    print(f"weighted diff CI95%: [{ci_l:.4f}, {ci_u:.4f}]")
    print(f"weighted p_wilcoxon: {p_w:.4g}, p_ttest: {p_t:.4g}")

    if stats is None:
        print("\n[提示] 未安装 scipy，显著性检验部分显示为 NaN。")


if __name__ == "__main__":
    main()
