import pandas as pd
import numpy as np


# =====================================================
# CONFIG
# =====================================================

INPUT_FILE = "emotion_values_P01_time_sorted.csv"

OUTPUT_FILE = "rule_analysis_report.txt"

THRESHOLD = 0.20


EMOTIONS = [
    "anger",
    "disgust",
    "happiness",
    "sadness",
    "surprise",
    "neutral"
]


CAM_SYSTEMS = [
    "DL_FR","DL_LU","DL_XP",
    "DR_FR","DR_LU","DR_XP",
    "UL_FR","UL_LU","UL_XP",
    "UR_FR","UR_LU","UR_XP"
]


RULES = [
    (5,6),
    (6,5),
    (4,6),
    (6,4),
    (3,6),
    (6,3),
    (2,6),
    (6,2),
    (1,6),
    (6,1)
]


idx_to_emo = {
    i+1:e
    for i,e in enumerate(EMOTIONS)
}


# =====================================================
# LOAD
# =====================================================

# =====================================================
# LOAD - TEST FIRST 2000 ROWS
# =====================================================

df = pd.read_csv(
    INPUT_FILE,
    nrows=18000
)


df["cam_sys"] = (
    df["camera"]
    +
    "_"
    +
    df["system"]
)


df = df.sort_values(
    ["moment","cam_sys"]
)


print("TEST ROWS:", len(df))
print(
    "MOMENTS:",
    df["moment"].min(),
    "-",
    df["moment"].max()
)


# =====================================================
# GROUP MOMENTS
# =====================================================

moments = {
    m:g
    for m,g in df.groupby("moment")
}



# =====================================================
# STORAGE
# =====================================================


detected = []

hidden = []


details_detected = []

details_hidden = []


agreement = []



# =====================================================
# HELPERS
# =====================================================


def top2_check(row, emotion):

    values = [
        float(row[e])
        for e in EMOTIONS
    ]

    target = float(row[emotion])

    maximum = max(values)


    top2 = sorted(
        values,
        reverse=True
    )[:2]


    active = target >= THRESHOLD


    is_top2 = target in top2


    distance = None

    if active and is_top2:
        distance = maximum-target


    return active,is_top2,distance



# =====================================================
# MAIN
# =====================================================


counter = 0


for moment in sorted(moments):


    if moment+1 not in moments:
        continue


    now = moments[moment]


    after = moments[moment+1]



    now_map = {
        r.cam_sys:r
        for _,r in now.iterrows()
    }


    after_map = {
        r.cam_sys:r
        for _,r in after.iterrows()
    }



    for cs,row in now_map.items():


        row_after = after_map.get(cs)


        if row_after is None:
            continue



        emo_before = row["dominant_emotion"]

        emo_after = row_after["dominant_emotion"]



        if emo_before not in EMOTIONS:
            continue

        if emo_after not in EMOTIONS:
            continue



        e1 = EMOTIONS.index(emo_before)+1

        e2 = EMOTIONS.index(emo_after)+1



        # =============================================
        # AGREEMENT
        # =============================================

        same = 0

        for _,r in after.iterrows():

            if r["dominant_emotion"] == emo_after:
                same += 1


        agreement.append(
            same/12
        )



        # =============================================
        # DETECTED RULE
        # =============================================


        if (e1,e2) in RULES:


            counter += 1


            b_active,b_top,b_dist = top2_check(
                row,
                emo_after
            )


            a_active,a_top,a_dist = top2_check(
                row_after,
                emo_before
            )



            detected.append(
                [
                    b_active,
                    b_top,
                    b_dist,
                    a_active,
                    a_top,
                    a_dist
                ]
            )



            details_detected.append(
                f"""
MOMENT {moment}
{cs}
RULE {emo_before}->{emo_after}

BEFORE:
{emo_after}
active={b_active}
TOP2={b_top}
distance={b_dist}

AFTER:
{emo_before}
active={a_active}
TOP2={a_top}
distance={a_dist}

"""
            )



        # =============================================
        # HIDDEN POSSIBILITY
        # =============================================


        else:


            # tylko jeśli mogłaby być reguła
            # przed startowa emocja
            # po końcowa emocja


            b_active,_,_ = top2_check(
                row,
                emo_after
            )


            a_active,_,_ = top2_check(
                row_after,
                emo_after
            )


            if b_active and a_active:


                _,b_top,b_dist = top2_check(
                    row,
                    emo_after
                )


                _,a_top,a_dist = top2_check(
                    row_after,
                    emo_after
                )



                hidden.append(
                    [
                        b_top,
                        a_top,
                        b_dist,
                        a_dist
                    ]
                )


                details_hidden.append(
                    f"""
HIDDEN POSSIBILITY

MOMENT {moment}

{cs}

Potential:
{emo_before}->{emo_after}

BEFORE target:
active={b_active}
TOP2={b_top}
distance={b_dist}

AFTER target:
active={a_active}
TOP2={a_top}
distance={a_dist}

"""
                )


    if moment % 1000 == 0:

        print(
            f"processed moment {moment}"
        )



# =====================================================
# REPORT
# =====================================================


def mean_percent(values):

    if len(values)==0:
        return 0

    return np.mean(values)*100



with open(
    OUTPUT_FILE,
    "w",
    encoding="utf8"
) as f:


    f.write(
        "AFFECTIVE RULE ANALYSIS\n\n"
    )


    f.write(
        f"Detected rules: {len(detected)}\n\n"
    )


    f.write(
        "GLOBAL AGREEMENT\n"
    )

    f.write(
        f"Mean agreement: "
        f"{np.mean(agreement)*100:.2f}%\n\n"
    )



    if detected:


        d=np.array(
            detected,
            dtype=object
        )


        f.write(
            "DETECTED RULES\n\n"
        )


        f.write(
            f"Before active: "
            f"{mean_percent(d[:,0]):.2f}%\n"
        )


        f.write(
            f"Before TOP2: "
            f"{mean_percent(d[:,1]):.2f}%\n"
        )


        dist=[
            x for x in d[:,2]
            if x is not None
        ]


        if dist:

            f.write(
                f"Before distance mean: "
                f"{np.mean(dist):.5f}\n"
            )



        f.write(
            f"\nAfter active: "
            f"{mean_percent(d[:,3]):.2f}%\n"
        )


        f.write(
            f"After TOP2: "
            f"{mean_percent(d[:,4]):.2f}%\n"
        )



        dist=[
            x for x in d[:,5]
            if x is not None
        ]


        if dist:

            f.write(
                f"After distance mean: "
                f"{np.mean(dist):.5f}\n"
            )





    f.write(
        "\n\nHIDDEN POSSIBILITY\n\n"
    )


    f.write(
        f"Cases: {len(hidden)}\n"
    )


    if hidden:


        h=np.array(
            hidden,
            dtype=object
        )


        f.write(
            f"TOP2 percentage: "
            f"{np.mean(h[:,0])*100:.2f}%\n"
        )


        dist=[
            x for x in h[:,2]
            if x is not None
        ]


        if dist:

            f.write(
                f"Distance mean: "
                f"{np.mean(dist):.5f}\n"
            )



    f.write(
        "\n\nDETECTED DETAILS\n"
    )


    for x in details_detected:

        f.write(x)



    f.write(
        "\n\nHIDDEN DETAILS\n"
    )


    for x in details_hidden:

        f.write(x)



print("DONE")
print("Saved:", OUTPUT_FILE)