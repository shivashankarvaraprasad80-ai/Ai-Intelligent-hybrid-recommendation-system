import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from mbti import get_mbti_score

# ----------------------------------
# LOAD DATA
# ----------------------------------

users_df = pd.read_csv("data/users.csv")

# ----------------------------------
# CREATE COMBINED PROFILE TEXT
# ----------------------------------

users_df["profile_text"] = (
    users_df["professional_summary"].fillna("") + " "
    + users_df["about_me"].fillna("") + " "
    + users_df["interests"].fillna("")
)

# ----------------------------------
# LOAD NLP MODEL
# ----------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ----------------------------------
# GENERATE EMBEDDINGS
# ----------------------------------

embeddings = model.encode(
    users_df["profile_text"].tolist(),
    convert_to_tensor=False
)

# ----------------------------------
# LOCATION SCORE
# ----------------------------------

def get_location_score(loc1, loc2):

    if loc1 == loc2:
        return 100

    return 50

# ----------------------------------
# RECOMMEND FUNCTION
# ----------------------------------

def get_top_matches(user_id, top_n=5):

    user_row = users_df[
        users_df["user_id"] == user_id
    ].iloc[0]

    user_index = user_row.name

    scores = []

    for idx, candidate in users_df.iterrows():

        if candidate["user_id"] == user_id:
            continue

        # ----------------------------------
        # NLP SIMILARITY
        # ----------------------------------

        text_score = cosine_similarity(
            [embeddings[user_index]],
            [embeddings[idx]]
        )[0][0]

        text_score = text_score * 100

        # ----------------------------------
        # MBTI SCORE
        # ----------------------------------

        mbti_score = get_mbti_score(
            user_row["mbti"],
            candidate["mbti"]
        )

        # ----------------------------------
        # LOCATION SCORE
        # ----------------------------------

        location_score = get_location_score(
            user_row["location"],
            candidate["location"]
        )

        # ----------------------------------
        # FINAL HYBRID SCORE
        # ----------------------------------

        total_score = (
            (0.5 * text_score)
            + (0.3 * mbti_score)
            + (0.2 * location_score)
        )

        scores.append({
            "user_id": candidate["user_id"],
            "name": candidate["name"],
            "profession": candidate["profession"],
            "mbti": candidate["mbti"],
            "location": candidate["location"],
            "compatibility_score": round(total_score, 2),
            "interests": candidate["interests"]
        })

    results = pd.DataFrame(scores)

    print(
        "Scores generated:",
        len(scores)
    )

    return results.sort_values(
        by="compatibility_score",
        ascending=False
    ).head(top_n)

# ----------------------------------
# TESTING
# ----------------------------------

if __name__ == "__main__":

    test_user = "U001"

    matches = get_top_matches(
        test_user,
        top_n=5
    )

    print(
        "\nTop 5 Matches For",
        test_user
    )

    print(matches)