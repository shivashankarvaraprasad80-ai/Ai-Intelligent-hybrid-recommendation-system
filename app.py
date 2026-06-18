from src.reranker import rerank_matches
import streamlit as st
import pandas as pd
import sys

sys.path.append("./src")

from matcher import get_top_matches, users_df


st.set_page_config(
page_title="Profile Matching System",
page_icon="🤝",
layout="wide"
)


st.markdown("""

<style>
.main {
    padding-top: 1rem;
}
</style>""", unsafe_allow_html=True)



st.title("🤝 Intelligent Hybrid Recommendation System")

st.write(
"AI-powered profile matching using NLP, MBTI compatibility, location scoring, and machine learning re-ranking."
)


user_options = users_df["user_id"].tolist()

selected_user = st.selectbox(
"Select User",
user_options
)

if st.button("Find Top Matches"):

 matches = get_top_matches(selected_user)

 current_user = users_df[
    users_df["user_id"] == selected_user
].iloc[0]

 st.subheader("👤 Selected User Profile")

 st.write({
    "Name": current_user["name"],
    "Profession": current_user["profession"],
    "Location": current_user["location"],
    "MBTI": current_user["mbti"],
    "Age": current_user["age"]
    
})

# --------------------------------
# RE-RANK MATCHES
# --------------------------------

 matches = rerank_matches(
    matches,
    current_user
)
 matches["match_percentage"] = (
    matches["compatibility_score"]
    / matches["compatibility_score"].max()
) * 100

 avg_accuracy = round(
    matches["match_percentage"].mean(),
    2
)

 best_accuracy = round(
    matches["match_percentage"].max(),
    2
)

# --------------------------------
# KPI CARDS
# --------------------------------

 col1, col2, col3 = st.columns(3)

 with col1:
    st.metric(
        "Total Users",
        len(users_df)
    )

 with col2:
    st.metric(
        "Top Matches",
        len(matches)
    )

 with col3:
    st.metric(
        "Best Match %",
        f"{best_accuracy}%"
    )

# --------------------------------
# TOP MATCHES TABLE
# --------------------------------

 st.subheader(
    f"🏆 Top Matches for {selected_user}"
)

 st.dataframe(matches)

# --------------------------------
# DISPLAY TOP 5 RESUMES
# --------------------------------

 st.subheader("📄 Top 5 Matched Resumes")

 for rank, (_, match) in enumerate(
    matches.iterrows(),
    start=1
):

    st.markdown(
        f"## 🏅 Rank #{rank}"
    )

    st.write({
        "User ID": match["user_id"],
        "Name": match["name"],
        "Profession": match["profession"],
        "Location": match["location"],
        "MBTI": match["mbti"],
        "Interests": match["interests"],
        "Compatibility %": round(
            match["match_percentage"],
            2
        )
    })

    st.progress(
        int(match["match_percentage"])
    )

    st.info(
        f"""
        • MBTI Type: {match['mbti']}
        • Location: {match['location']}
        • Profession: {match['profession']}
        • Interests: {match['interests']}
        • Compatibility Score: {round(match['match_percentage'],2)}%
        • AI Feedback Score: {round(match['feedback_score'],4)}
        """
    )

    st.markdown("---")

# --------------------------------
# CHARTS
# --------------------------------

 st.subheader("📊 Compatibility Chart")

 st.bar_chart(
    matches.set_index("name")[
        "compatibility_score"
    ]
)

 st.subheader("📈 MBTI Distribution")

 st.bar_chart(
    matches["mbti"].value_counts()
)

# --------------------------------
# BEST MATCH
# --------------------------------

 best = matches.iloc[0]
 st.subheader("🌟 Best Match")

 st.write({
    "Name": best["name"],
    "Profession": best["profession"],
    "Location": best["location"],
    "MBTI": best["mbti"],
    "Compatibility %": round(
        best["match_percentage"],
        2
    ),
    "Feedback Score": round(
        best["feedback_score"],
        4
    )
})

 st.success(
    f"{best['name']} is the highest-ranked recommendation "
    f"based on NLP similarity, MBTI compatibility, "
    f"location score, and Logistic Regression re-ranking."
)

# --------------------------------
# PERFORMANCE ANALYSIS
# --------------------------------

 st.subheader("📈 Performance Analysis")

 avg_accuracy = round(
    matches["match_percentage"].mean(),
    2
)

 st.metric(
    label="Average Recommendation Accuracy",
    value=f"{avg_accuracy}%"
)

# --------------------------------
# AI FEEDBACK SCORES
# --------------------------------

 st.subheader("🤖 AI Feedback Scores")

 st.dataframe(
    matches[
        [
            "user_id",
            "name",
            "feedback_score"
        ]
    ]
)

# --------------------------------
# PROJECT SUMMARY
# --------------------------------

 st.subheader("📋 Project Summary")

 st.write({
    "Users": len(users_df),
    "Matching Method": "NLP + MBTI + Location",
    "NLP Model": "all-MiniLM-L6-v2",
    "ML Model": "Logistic Regression",
    "Top Matches Displayed": 5
})

# --------------------------------
# PROJECT STATISTICS
# --------------------------------

 st.subheader("📊 Project Statistics")

 st.write({
    "Total Users": len(users_df),
    "Top Matches Displayed": len(matches),
    "Recommendation Model": "Hybrid Recommendation System",
    "AI Re-ranking": "Logistic Regression",
    "Accuracy": f"{avg_accuracy}%"
})

# --------------------------------
# DOWNLOAD CSV
# --------------------------------

 csv = matches.to_csv(
    index=False
)

 st.download_button(
    label="⬇️ Download Recommendations",
    data=csv,
    file_name="recommendations.csv",
    mime="text/csv"
)

 st.success(
    "Recommendations Generated Successfully!"
)
