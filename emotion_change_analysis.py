import pandas as pd
import numpy as np
from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = "emotion_values_P01_time_sorted.csv"
OUTPUT_FILE = "emotion_change_report.txt"
HISTOGRAM_FILE = "new_rule_histogram.csv"

NROWS = None          # None = cały plik
PROGRESS_EVERY = 100    # co ile momentów wypisać postęp

THRESHOLD = 0.20

EMOTIONS = [
    "anger",
    "disgust",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
]

RULES = [
    (5, 6), (6, 5),
    (4, 6), (6, 4),
    (3, 6), (6, 3),
    (2, 6), (6, 2),
    (1, 6), (6, 1),
]


# =====================================================
# HELPERS
# =====================================================

def parse_active(row):
    raw = row.get("active_emotions", "")
    if pd.isna(raw) or raw == "":
        return set()
    return {e.strip() for e in str(raw).split(";") if e.strip()}


def emotion_active(row, emotion):
    if emotion in parse_active(row):
        return True
    return float(row[emotion]) >= THRESHOLD


def emotion_top2(row, emotion):
    values = [float(row[e]) for e in EMOTIONS]
    target = float(row[emotion])
    top2 = sorted(values, reverse=True)[:2]
    return target in top2


def distance_not_dominant(row, emotion):
    """Odległość tylko gdy emocja aktywna, ale nie dominant_emotion."""
    if not emotion_active(row, emotion):
        return None
    if row["dominant_emotion"] == emotion:
        return None
    values = [float(row[e]) for e in EMOTIONS]
    return max(values) - float(row[emotion])


def analyze_row(row, emotion):
    active = emotion_active(row, emotion)
    top2 = emotion_top2(row, emotion) if active else False
    dist = distance_not_dominant(row, emotion)
    return active, top2, dist


def pct(count, total):
    if total == 0:
        return 0.0
    return count / total * 100


def same_rule_at_combo(row_before, row_after, emo_old, emo_new):
    """Czy w danej kombinacji kamera/system zaszła ta sama reguła dominant->dominant."""
    return (
        row_before["dominant_emotion"] == emo_old
        and row_after["dominant_emotion"] == emo_new
    )


def hidden_other_combos(now_map, after_map, detected_cs, emo_old, emo_new):
    """
    Pozostałe kombinacje bez wykrycia reguły w detected_cs.
    Wyklucza te, gdzie ta sama reguła już zadziałała (dominant przed/po).
    """
    hidden = []
    excluded_same_rule = []

    for cs in now_map:
        if cs == detected_cs:
            continue
        if cs not in after_map:
            continue

        row_before = now_map[cs]
        row_after = after_map[cs]

        if same_rule_at_combo(row_before, row_after, emo_old, emo_new):
            excluded_same_rule.append(cs)
            continue

        hidden.append(cs)

    return hidden, excluded_same_rule


def combos_with_same_rule(now_map, after_map, emo_old, emo_new):
    """Wszystkie kombinacje kamera/system, gdzie zadziałała ta sama reguła."""
    fired = []
    for cs in now_map:
        if cs not in after_map:
            continue
        if same_rule_at_combo(now_map[cs], after_map[cs], emo_old, emo_new):
            fired.append(cs)
    return fired


def summarize_rows(rows, emotion):
    n = len(rows)
    active_count = 0
    top2_count = 0
    active_not_top2 = 0
    distances = []
    bad_cases = []

    for cs, row in rows:
        active, top2, dist = analyze_row(row, emotion)
        if active:
            active_count += 1
            if top2:
                top2_count += 1
            else:
                active_not_top2 += 1
                bad_cases.append(
                    f"  {cs}: {emotion} active but not TOP2 "
                    f"(dom={row['dominant_emotion']}, val={float(row[emotion]):.5f})"
                )
        if dist is not None:
            distances.append(dist)

    return {
        "n": n,
        "active_count": active_count,
        "active_pct": pct(active_count, n),
        "top2_count": top2_count,
        "top2_when_active_pct": pct(top2_count, active_count) if active_count else 0.0,
        "active_not_top2": active_not_top2,
        "distances": distances,
        "dist_mean": np.mean(distances) if distances else None,
        "dist_std": np.std(distances) if distances else None,
        "bad_cases": bad_cases,
    }


def summarize_hidden_both_active(pairs, emo_old, emo_new):
    """
    Kombinacje bez wykrycia reguły, gdzie:
    - przed: stara emocja (emo_old) aktywna
    - po:    nowa emocja (emo_new) aktywna
    """
    n = len(pairs)
    both_count = 0
    before_top2_count = 0
    after_top2_count = 0
    before_distances = []
    after_distances = []
    matches = []

    for cs, row_before, row_after in pairs:
        old_active, old_top2, old_dist = analyze_row(row_before, emo_old)
        new_active, new_top2, new_dist = analyze_row(row_after, emo_new)

        if not (old_active and new_active):
            continue

        both_count += 1
        if old_top2:
            before_top2_count += 1
        if new_top2:
            after_top2_count += 1
        if old_dist is not None:
            before_distances.append(old_dist)
        if new_dist is not None:
            after_distances.append(new_dist)

        matches.append(
            f"  {cs}: {emo_old} przed + {emo_new} po "
            f"(przed dom={row_before['dominant_emotion']}, "
            f"po dom={row_after['dominant_emotion']})"
        )

    return {
        "n": n,
        "both_count": both_count,
        "both_pct": pct(both_count, n),
        "before_top2_count": before_top2_count,
        "before_top2_when_both_pct": pct(before_top2_count, both_count) if both_count else 0.0,
        "after_top2_count": after_top2_count,
        "after_top2_when_both_pct": pct(after_top2_count, both_count) if both_count else 0.0,
        "before_distances": before_distances,
        "after_distances": after_distances,
        "before_dist_mean": np.mean(before_distances) if before_distances else None,
        "before_dist_std": np.std(before_distances) if before_distances else None,
        "after_dist_mean": np.mean(after_distances) if after_distances else None,
        "after_dist_std": np.std(after_distances) if after_distances else None,
        "matches": matches,
    }


def write_summary_block(f, title, stats, emotion_label):
    f.write(f"{title}\n")
    f.write(f"{emotion_label} active: {stats['active_pct']:.2f}% "
            f"({stats['active_count']}/{stats['n']})\n")
    f.write(f"TOP2 when active: {stats['top2_when_active_pct']:.2f}%\n")
    if stats["dist_mean"] is not None:
        f.write(f"Distance mean: {stats['dist_mean']:.5f}\n")
        f.write(f"Distance std: {stats['dist_std']:.5f}\n")
    else:
        f.write("Distance mean: -\n")
        f.write("Distance std: -\n")
    f.write(f"Active not TOP2 cases: {stats['active_not_top2']}\n")


# =====================================================
# LOAD
# =====================================================

read_kwargs = {}
if NROWS is not None:
    read_kwargs["nrows"] = NROWS

df = pd.read_csv(INPUT_FILE, **read_kwargs)
df["cam_sys"] = df["camera"] + "_" + df["system"]
df = df.sort_values(["moment", "cam_sys"])

moments = {m: g for m, g in df.groupby("moment")}

print(f"Wczytano wierszy: {len(df)}")
print(f"Momenty: {df['moment'].min()} - {df['moment'].max()} ({len(moments)} unikalnych)")
print(f"Próg aktywności: {THRESHOLD}")
print("Start analizy...")

# =====================================================
# MAIN LOOP
# =====================================================

all_agreements = []

# hidden = kombinacje bez wykrycia reguły
hidden_before_stats = []
hidden_after_stats = []
hidden_both_stats = []

# detected = kombinacja gdzie wykryto regułę
detected_before_stats = []
detected_after_stats = []

event_details = []

# histogram: ile dodatkowych kombinacji wykryło tę samą regułę w tym samym kroku
# klucz = liczba NOWYCH (poza pierwszą), wartość = ile takich zdarzeń moment+reguła
new_rule_histogram = defaultdict(int)
new_rule_histogram_by_rule = defaultdict(lambda: defaultdict(int))

moment_list = sorted(moments)
processed_pairs = 0
detected_count = 0
change_count = 0

for moment in moment_list:
    if moment + 1 not in moments:
        continue

    processed_pairs += 1
    if processed_pairs % PROGRESS_EVERY == 0:
        print(
            f"  przetworzono par momentów: {processed_pairs}, "
            f"zmian emocji: {change_count}, wykryte reguły: {detected_count}"
        )

    now = moments[moment]
    after = moments[moment + 1]

    now_map = {r.cam_sys: r for _, r in now.iterrows()}
    after_map = {r.cam_sys: r for _, r in after.iterrows()}

    # histogram: raz na parę momentów + typ reguły
    rules_seen_this_step = set()
    for cs, row_before in now_map.items():
        row_after = after_map.get(cs)
        if row_after is None:
            continue

        emo_old = row_before["dominant_emotion"]
        emo_new = row_after["dominant_emotion"]
        if emo_old not in EMOTIONS or emo_new not in EMOTIONS:
            continue
        if emo_old == emo_new:
            continue

        e1 = EMOTIONS.index(emo_old) + 1
        e2 = EMOTIONS.index(emo_new) + 1
        if (e1, e2) not in RULES:
            continue

        rule_key = (emo_old, emo_new)
        if rule_key in rules_seen_this_step:
            continue
        rules_seen_this_step.add(rule_key)

        fired_dominant = combos_with_same_rule(now_map, after_map, emo_old, emo_new)
        if not fired_dominant:
            continue

        # pozostałe bez reguły dominant; liczymy ile ma starą emocję przed + nową po (ponad próg)
        hidden_cs = [
            cs for cs in now_map
            if cs not in fired_dominant and cs in after_map
        ]
        hidden_pairs = [
            (cs, now_map[cs], after_map[cs]) for cs in hidden_cs
        ]
        n_hidden_both = summarize_hidden_both_active(
            hidden_pairs, emo_old, emo_new
        )["both_count"]

        new_rule_histogram[n_hidden_both] += 1
        new_rule_histogram_by_rule[rule_key][n_hidden_both] += 1

    for cs, row_before in now_map.items():
        row_after = after_map.get(cs)
        if row_after is None:
            continue

        emo_before = row_before["dominant_emotion"]
        emo_after = row_after["dominant_emotion"]

        if emo_before not in EMOTIONS or emo_after not in EMOTIONS:
            continue
        if emo_before == emo_after:
            continue

        change_count += 1

        e1 = EMOTIONS.index(emo_before) + 1
        e2 = EMOTIONS.index(emo_after) + 1

        agreement = sum(
            1 for _, r in after.iterrows() if r["dominant_emotion"] == emo_after
        ) / 12
        all_agreements.append(agreement)

        if (e1, e2) not in RULES:
            continue

        detected_count += 1

        # --- kombinacje bez wykrycia (bez tych, gdzie ta sama reguła już zadziałała) ---
        other_cs, excluded_cs = hidden_other_combos(
            now_map, after_map, cs, emo_before, emo_after
        )

        hidden_before_rows = [(k, now_map[k]) for k in other_cs]
        hidden_after_rows = [(k, after_map[k]) for k in other_cs]
        hidden_pairs = [
            (k, now_map[k], after_map[k])
            for k in other_cs
        ]

        hb = summarize_rows(hidden_before_rows, emo_before)
        ha = summarize_rows(hidden_after_rows, emo_after)
        hboth = summarize_hidden_both_active(hidden_pairs, emo_before, emo_after)

        hidden_before_stats.append(hb)
        hidden_after_stats.append(ha)
        hidden_both_stats.append(hboth)

        # --- kombinacja wykryta (1) ---
        db = summarize_rows([(cs, row_before)], emo_after)
        da = summarize_rows([(cs, row_after)], emo_before)

        detected_before_stats.append(db)
        detected_after_stats.append(da)

        detail = []
        detail.append(f"\n{'='*60}")
        detail.append(f"MOMENT {moment} -> {moment + 1}")
        detail.append(f"{cs}: {emo_before} -> {emo_after}")
        detail.append(f"MEAN AGREEMENT: {agreement * 100:.2f}%")
        detail.append(
            f"Hidden with both active ({emo_before} przed + {emo_after} po, bez reguły dominant): "
            f"{hboth['both_count']}/{hboth['n']}"
        )
        detail.append(
            f"Dominant rule also in: {', '.join(excluded_cs) if excluded_cs else '(brak)'}"
        )
        detail.append(f"Hidden combos analyzed: {len(other_cs)}")
        detail.append("")
        detail.append("BEFORE CHANGE (other camera-system, old emotion)")
        detail.append(
            f"  {emo_before} active: {hb['active_pct']:.2f}% "
            f"({hb['active_count']}/{hb['n']})"
        )
        detail.append(f"  TOP2 when active: {hb['top2_when_active_pct']:.2f}%")
        if hb["dist_mean"] is not None:
            detail.append(f"  Distance mean: {hb['dist_mean']:.5f}")
            detail.append(f"  Distance std: {hb['dist_std']:.5f}")
        detail.append(f"  Active not TOP2: {hb['active_not_top2']}")
        if hb["bad_cases"]:
            detail.append("  BAD CASES:")
            detail.extend(hb["bad_cases"])

        detail.append("")
        detail.append("AFTER CHANGE (other camera-system, new emotion)")
        detail.append(
            f"  {emo_after} active: {ha['active_pct']:.2f}% "
            f"({ha['active_count']}/{ha['n']})"
        )
        detail.append(f"  TOP2 when active: {ha['top2_when_active_pct']:.2f}%")
        if ha["dist_mean"] is not None:
            detail.append(f"  Distance mean: {ha['dist_mean']:.5f}")
            detail.append(f"  Distance std: {ha['dist_std']:.5f}")
        detail.append(f"  Active not TOP2: {ha['active_not_top2']}")
        if ha["bad_cases"]:
            detail.append("  BAD CASES:")
            detail.extend(ha["bad_cases"])

        detail.append("")
        detail.append(
            f"HIDDEN BOTH ACTIVE (other cam-sys: {emo_before} przed + {emo_after} po)"
        )
        detail.append(
            f"  Both active: {hboth['both_pct']:.2f}% "
            f"({hboth['both_count']}/{hboth['n']})"
        )
        if hboth["both_count"]:
            detail.append(
                f"  {emo_before} TOP2 when both: {hboth['before_top2_when_both_pct']:.2f}%"
            )
            detail.append(
                f"  {emo_after} TOP2 when both: {hboth['after_top2_when_both_pct']:.2f}%"
            )
            if hboth["before_dist_mean"] is not None:
                detail.append(
                    f"  Before distance mean ({emo_before} not dom): "
                    f"{hboth['before_dist_mean']:.5f}"
                )
            if hboth["after_dist_mean"] is not None:
                detail.append(
                    f"  After distance mean ({emo_after} not dom): "
                    f"{hboth['after_dist_mean']:.5f}"
                )
        if hboth["matches"]:
            detail.append("  Matches:")
            detail.extend(hboth["matches"])

        detail.append("")
        detail.append(f"DETECTED {cs} (before moment, new emotion={emo_after})")
        detail.append(
            f"  {emo_after} active: {db['active_pct']:.2f}% "
            f"({db['active_count']}/{db['n']})"
        )
        detail.append(f"  TOP2 when active: {db['top2_when_active_pct']:.2f}%")
        if db["dist_mean"] is not None:
            detail.append(f"  Distance: {db['dist_mean']:.5f}")

        detail.append("")
        detail.append(f"DETECTED {cs} (after moment, old emotion={emo_before})")
        detail.append(
            f"  {emo_before} active: {da['active_pct']:.2f}% "
            f"({da['active_count']}/{da['n']})"
        )
        detail.append(f"  TOP2 when active: {da['top2_when_active_pct']:.2f}%")
        if da["dist_mean"] is not None:
            detail.append(f"  Distance: {da['dist_mean']:.5f}")

        event_details.append("\n".join(detail))


# =====================================================
# AGGREGATE GLOBAL STATS
# =====================================================

def aggregate_stats(stats_list):
    if not stats_list:
        return None
    total_n = sum(s["n"] for s in stats_list)
    total_active = sum(s["active_count"] for s in stats_list)
    total_top2 = sum(s["top2_count"] for s in stats_list)
    total_not_top2 = sum(s["active_not_top2"] for s in stats_list)
    all_dist = [d for s in stats_list for d in s["distances"]]
    return {
        "active_pct": pct(total_active, total_n),
        "top2_when_active_pct": pct(total_top2, total_active) if total_active else 0.0,
        "active_not_top2": total_not_top2,
        "dist_mean": np.mean(all_dist) if all_dist else None,
        "dist_std": np.std(all_dist) if all_dist else None,
    }


agg_hidden_before = aggregate_stats(hidden_before_stats)
agg_hidden_after = aggregate_stats(hidden_after_stats)
agg_detected_before = aggregate_stats(detected_before_stats)
agg_detected_after = aggregate_stats(detected_after_stats)


def aggregate_both_stats(stats_list):
    if not stats_list:
        return None
    total_n = sum(s["n"] for s in stats_list)
    total_both = sum(s["both_count"] for s in stats_list)
    total_before_top2 = sum(s["before_top2_count"] for s in stats_list)
    total_after_top2 = sum(s["after_top2_count"] for s in stats_list)
    all_before_dist = [d for s in stats_list for d in s["before_distances"]]
    all_after_dist = [d for s in stats_list for d in s["after_distances"]]
    return {
        "both_pct": pct(total_both, total_n),
        "both_count": total_both,
        "total_n": total_n,
        "before_top2_when_both_pct": pct(total_before_top2, total_both) if total_both else 0.0,
        "after_top2_when_both_pct": pct(total_after_top2, total_both) if total_both else 0.0,
        "before_dist_mean": np.mean(all_before_dist) if all_before_dist else None,
        "before_dist_std": np.std(all_before_dist) if all_before_dist else None,
        "after_dist_mean": np.mean(all_after_dist) if all_after_dist else None,
        "after_dist_std": np.std(all_after_dist) if all_after_dist else None,
    }


agg_hidden_both = aggregate_both_stats(hidden_both_stats)


def format_histogram(hist, total_events):
    lines = []
    if total_events == 0:
        return ["(brak zdarzeń)\n"]

    max_n = max(hist.keys()) if hist else 0
    for n in range(max_n + 1):
        count = hist.get(n, 0)
        pct_val = count / total_events * 100 if total_events else 0
        bar = "#" * int(pct_val / 2)
        lines.append(
            f"{n:2d} hidden combos ({n} with old+new active): "
            f"{count:6d} events ({pct_val:6.2f}%) {bar}"
        )
    return lines


total_histogram_events = sum(new_rule_histogram.values())


# =====================================================
# REPORT
# =====================================================

with open(OUTPUT_FILE, "w", encoding="utf8") as f:
    f.write("EMOTION CHANGE ANALYSIS\n")
    f.write(f"Input: {INPUT_FILE}\n")
    if NROWS is not None:
        f.write(f"Rows loaded: {NROWS} (test subset)\n")
    f.write(f"Threshold: {THRESHOLD}\n\n")

    f.write(f"Detected rule events: {len(detected_before_stats)}\n\n")

    if all_agreements:
        f.write(f"MEAN AGREEMENT (all changes): {np.mean(all_agreements) * 100:.2f}%\n\n")

    f.write("=" * 60 + "\n")
    f.write("HIDDEN BOTH ACTIVE HISTOGRAM\n")
    f.write("=" * 60 + "\n")
    f.write(
        "Reguła wykryta w dominant_emotion. Dla pozostałych kombinacji (bez tej reguły "
        "w dominant): ile ma starą emocję aktywną PRZED i nową emocję aktywną PO "
        f"(ponad próg {THRESHOLD} / active_emotions).\n"
    )
    f.write(f"Rule-step events (unique moment+rule): {total_histogram_events}\n\n")
    for line in format_histogram(new_rule_histogram, total_histogram_events):
        f.write(line + "\n")

    f.write("\nPer rule type:\n")
    for (emo_old, emo_new) in sorted(new_rule_histogram_by_rule.keys()):
        rule_hist = new_rule_histogram_by_rule[(emo_old, emo_new)]
        rule_total = sum(rule_hist.values())
        f.write(f"\n  {emo_old} -> {emo_new} ({rule_total} events):\n")
        for line in format_histogram(rule_hist, rule_total):
            f.write(f"    {line}\n")

    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("HIDDEN (other camera-system combinations)\n")
    f.write("=" * 60 + "\n\n")

    if agg_hidden_before:
        f.write("BEFORE CHANGE (old emotion at before moment)\n")
        f.write(f"Emotion active: {agg_hidden_before['active_pct']:.2f}%\n")
        f.write(f"TOP2 when active: {agg_hidden_before['top2_when_active_pct']:.2f}%\n")
        if agg_hidden_before["dist_mean"] is not None:
            f.write(f"Distance mean: {agg_hidden_before['dist_mean']:.5f}\n")
            f.write(f"Distance std: {agg_hidden_before['dist_std']:.5f}\n")
        f.write(f"Active not TOP2 cases: {agg_hidden_before['active_not_top2']}\n\n")

    if agg_hidden_after:
        f.write("AFTER CHANGE (new emotion at after moment)\n")
        f.write(f"Emotion active: {agg_hidden_after['active_pct']:.2f}%\n")
        f.write(f"TOP2 when active: {agg_hidden_after['top2_when_active_pct']:.2f}%\n")
        if agg_hidden_after["dist_mean"] is not None:
            f.write(f"Distance mean: {agg_hidden_after['dist_mean']:.5f}\n")
            f.write(f"Distance std: {agg_hidden_after['dist_std']:.5f}\n")
        f.write(f"Active not TOP2 cases: {agg_hidden_after['active_not_top2']}\n\n")

    if agg_hidden_both:
        f.write("HIDDEN BOTH ACTIVE (old emotion before + new emotion after, same cam-sys)\n")
        f.write(
            f"Both active: {agg_hidden_both['both_pct']:.2f}% "
            f"({agg_hidden_both['both_count']}/{agg_hidden_both['total_n']})\n"
        )
        if agg_hidden_both["both_count"]:
            f.write(
                f"Old emotion TOP2 when both: "
                f"{agg_hidden_both['before_top2_when_both_pct']:.2f}%\n"
            )
            f.write(
                f"New emotion TOP2 when both: "
                f"{agg_hidden_both['after_top2_when_both_pct']:.2f}%\n"
            )
            if agg_hidden_both["before_dist_mean"] is not None:
                f.write(
                    f"Before distance mean (old emotion not dom): "
                    f"{agg_hidden_both['before_dist_mean']:.5f}\n"
                )
                f.write(
                    f"Before distance std: {agg_hidden_both['before_dist_std']:.5f}\n"
                )
            if agg_hidden_both["after_dist_mean"] is not None:
                f.write(
                    f"After distance mean (new emotion not dom): "
                    f"{agg_hidden_both['after_dist_mean']:.5f}\n"
                )
                f.write(
                    f"After distance std: {agg_hidden_both['after_dist_std']:.5f}\n"
                )
        f.write("\n")

    f.write("=" * 60 + "\n")
    f.write("DETECTED (camera-system where rule fired)\n")
    f.write("=" * 60 + "\n\n")

    if agg_detected_before:
        f.write("BEFORE CHANGE (new emotion at before moment)\n")
        f.write(f"Emotion active: {agg_detected_before['active_pct']:.2f}%\n")
        f.write(f"TOP2 when active: {agg_detected_before['top2_when_active_pct']:.2f}%\n")
        if agg_detected_before["dist_mean"] is not None:
            f.write(f"Distance mean: {agg_detected_before['dist_mean']:.5f}\n")
            f.write(f"Distance std: {agg_detected_before['dist_std']:.5f}\n")
        f.write(f"Active not TOP2 cases: {agg_detected_before['active_not_top2']}\n\n")

    if agg_detected_after:
        f.write("AFTER CHANGE (old emotion at after moment)\n")
        f.write(f"Previous emotion active: {agg_detected_after['active_pct']:.2f}%\n")
        f.write(f"TOP2 when active: {agg_detected_after['top2_when_active_pct']:.2f}%\n")
        if agg_detected_after["dist_mean"] is not None:
            f.write(f"Distance mean: {agg_detected_after['dist_mean']:.5f}\n")
            f.write(f"Distance std: {agg_detected_after['dist_std']:.5f}\n")
        f.write(f"Active not TOP2 cases: {agg_detected_after['active_not_top2']}\n\n")

    f.write("=" * 60 + "\n")
    f.write("EVENT DETAILS\n")
    f.write("=" * 60 + "\n")
    for detail in event_details:
        f.write(detail)
        f.write("\n")

hist_rows = []
max_n = max(new_rule_histogram.keys()) if new_rule_histogram else 0
for n in range(max_n + 1):
    hist_rows.append({
        "hidden_both_active_count": n,
        "events": new_rule_histogram.get(n, 0),
        "pct": (
            new_rule_histogram.get(n, 0) / total_histogram_events * 100
            if total_histogram_events else 0
        ),
    })
pd.DataFrame(hist_rows).to_csv(HISTOGRAM_FILE, index=False)

print(
    f"Koniec: par momentów={processed_pairs}, "
    f"zmian emocji={change_count}, wykryte reguły={detected_count}"
)
print("DONE")
print(f"Saved: {OUTPUT_FILE}")
print(f"Saved: {HISTOGRAM_FILE}")
print(f"Detected events: {len(detected_before_stats)}")
print(f"Histogram rule-step events: {total_histogram_events}")
