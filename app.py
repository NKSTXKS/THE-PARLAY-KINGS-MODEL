
import streamlit as st
import joblib
import requests
import pandas as pd
import numpy as np
from nba_engine import render_nba_tab
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="PARLAY KINGZ",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# DESIGN
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #080b12;
        color: #ffffff;
    }

    h1, h2, h3 {
        color: #ffffff;
    }

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 900;
        color: #d4af37;
        margin-bottom: 0;
    }

    .subtitle {
        text-align: center;
        color: #b8c0cc;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .pick-card {
        background: #151c29;
        border: 1px solid #35445c;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .pick-team {
        color: #ffffff;
        font-size: 21px;
        font-weight: 800;
    }

    .pick-label {
        color: #d4af37;
        font-size: 13px;
        font-weight: 700;
    }

    .pick-probability {
        color: #39d98a;
        font-size: 25px;
        font-weight: 900;
    }

    div[data-testid="stMetric"] {
        background: #151c29;
        border: 1px solid #35445c;
        padding: 12px;
        border-radius: 10px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 700;
    }

    [data-testid="stRadio"] {
        background: #151c29;
        padding: 12px;
        border-radius: 10px;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">👑 PARLAY KINGZ</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">COLLEGE BASKETBALL PREDICTION ENGINE</div>',
    unsafe_allow_html=True
)


# =========================================================
# NAVIGATION - MAIN SCREEN, NOT SIDEBAR
# =========================================================

pages = [
    "🏠 Home",
    "🎯 Daily Picks",
    "🧠 Model Information",
    "📊 Historical Data"
]

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "🏠 Home"

selected_page = st.radio(
    "Navigation",
    pages,
    horizontal=True,
    label_visibility="collapsed",
    key="selected_page"
)

st.divider()


# =========================================================
# MODEL BACKUP
# =========================================================

BACKUP_PATH = (
    Path(__file__).parent /
    "PARLAY_KINGZ_PRIVATE_BACKUP.joblib"
)

REQUIRED_FEATURES = [
    "home_avg_scored",
    "home_avg_allowed",
    "home_win_rate",
    "away_avg_scored",
    "away_avg_allowed",
    "away_win_rate"
]


@st.cache_resource
def load_private_model():

    if not BACKUP_PATH.exists():
        raise FileNotFoundError(
            "PARLAY_KINGZ_PRIVATE_BACKUP.joblib "
            "was not found in the repository."
        )

    backup = joblib.load(BACKUP_PATH)

    required_items = [
        "base_model",
        "calibrator",
        "raw_scores",
        "features"
    ]

    for item in required_items:
        if item not in backup:
            raise ValueError(
                f"Model backup is missing: {item}"
            )

    model = backup["base_model"]
    calibrator = backup["calibrator"]
    raw_scores = backup["raw_scores"]
    features = list(backup["features"])

    if features != REQUIRED_FEATURES:
        raise ValueError(
            "The saved model feature order does not match "
            "the prediction engine."
        )

    if model.n_features_in_ != len(REQUIRED_FEATURES):
        raise ValueError(
            "The saved model has an unexpected feature count."
        )

    if "game_date_time" not in raw_scores.columns:
        raise ValueError(
            "Historical data is missing game_date_time."
        )

    raw_scores = raw_scores.copy()

    raw_scores["game_date_time"] = pd.to_datetime(
        raw_scores["game_date_time"],
        utc=True,
        errors="coerce"
    )

    raw_scores = raw_scores.dropna(
        subset=[
            "game_date_time",
            "team_id",
            "team_score",
            "opponent_team_score"
        ]
    )

    return backup, model, calibrator, raw_scores


# =========================================================
# LOAD MODEL SAFELY
# =========================================================

try:
    backup, model, calibrator, raw_scores = load_private_model()

    MODEL_READY = True

except Exception as error:

    MODEL_READY = False

    st.error("The prediction model could not be loaded.")
    st.code(str(error))

    st.stop()


# =========================================================
# TEAM NAME HELPERS
# =========================================================

def normalize_name(name):

    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


def build_team_lookup(data):

    lookup = {}

    name_columns = [
        "team_display_name",
        "team_name",
        "team_location",
        "team_short_display_name",
        "team_abbreviation"
    ]

    available_columns = [
        column
        for column in name_columns
        if column in data.columns
    ]

    for _, row in data[
        ["team_id"] + available_columns
    ].drop_duplicates().iterrows():

        team_id = int(row["team_id"])

        for column in available_columns:

            value = row[column]

            if pd.notna(value):

                key = normalize_name(value)

                if key:
                    lookup[key] = team_id

    # Known ESPN/historical name differences
    aliases = {
        "bludevils": 150,
        "duke": 150,
        "fightingillini": 356,
        "illinois": 356
    }

    lookup.update(aliases)

    return lookup


TEAM_LOOKUP = build_team_lookup(raw_scores)


def find_team_id(team):

    possible_names = [
        team.get("displayName"),
        team.get("name"),
        team.get("shortDisplayName"),
        team.get("location"),
        team.get("abbreviation")
    ]

    for name in possible_names:

        if not name:
            continue

        key = normalize_name(name)

        if key in TEAM_LOOKUP:
            return TEAM_LOOKUP[key]

    return None


# =========================================================
# HISTORICAL TEAM FORM
# =========================================================

def get_team_statistics(team_id, game_time):

    cutoff_time = (
        game_time - pd.Timedelta(hours=3)
    )

    team_games = raw_scores[
        (raw_scores["team_id"] == team_id) &
        (raw_scores["game_date_time"] < cutoff_time)
    ].copy()

    team_games = team_games.sort_values(
        "game_date_time",
        ascending=False
    ).head(5)

    if len(team_games) < 3:
        return None

    return {
        "avg_scored": float(
            team_games["team_score"].mean()
        ),

        "avg_allowed": float(
            team_games["opponent_team_score"].mean()
        ),

        "win_rate": float(
            team_games["team_winner"]
            .fillna(False)
            .astype(bool)
            .mean()
        ),

        "games_used": len(team_games)
    }


# =========================================================
# ESPN DAILY SCHEDULE
# =========================================================

ESPN_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/scoreboard"
)


@st.cache_data(ttl=300, show_spinner=False)
def get_daily_schedule(date_string):

    response = requests.get(
        ESPN_URL,
        params={"dates": date_string},
        timeout=20
    )

    response.raise_for_status()

    return response.json()


# =========================================================
# CALIBRATED MODEL PREDICTION
# =========================================================

def calculate_prediction(features):

    feature_frame = pd.DataFrame(
        [features],
        columns=REQUIRED_FEATURES
    )

    base_probability = model.predict_proba(
        feature_frame
    )[0][1]

    base_probability = np.clip(
        base_probability,
        0.000001,
        0.999999
    )

    base_logit = np.log(
        base_probability /
        (1 - base_probability)
    )

    calibrated_probability = calibrator.predict_proba(
        np.array([[base_logit]])
    )[0][1]

    return float(calibrated_probability)


# =========================================================
# DAILY PREDICTION ENGINE
# =========================================================

def generate_daily_picks():

    today = datetime.now(
        ZoneInfo("America/Phoenix")
    )

    date_string = today.strftime("%Y%m%d")

    schedule = get_daily_schedule(date_string)

    predictions = []
    skipped = []

    events = schedule.get("events", [])

    for event in events:

        status = (
            event.get("status", {})
            .get("type", {})
            .get("state", "")
        )

        if status != "pre":
            continue

        competition = (
            event.get("competitions", [{}])[0]
        )

        competitors = competition.get(
            "competitors", []
        )

        home_team = None
        away_team = None

        for competitor in competitors:

            team = competitor.get("team", {})

            if competitor.get("homeAway") == "home":
                home_team = team

            elif competitor.get("homeAway") == "away":
                away_team = team

        if not home_team or not away_team:
            skipped.append(
                "A game was missing home or away team information."
            )
            continue

        home_id = find_team_id(home_team)
        away_id = find_team_id(away_team)

        if home_id is None or away_id is None:

            skipped.append(
                f"Could not match: "
                f"{home_team.get('displayName', 'Home')} "
                f"vs. "
                f"{away_team.get('displayName', 'Away')}"
            )

            continue

        event_time = pd.to_datetime(
            event.get("date"),
            utc=True,
            errors="coerce"
        )

        if pd.isna(event_time):
            continue

        home_stats = get_team_statistics(
            home_id,
            event_time
        )

        away_stats = get_team_statistics(
            away_id,
            event_time
        )

        if home_stats is None or away_stats is None:

            skipped.append(
                f"Not enough historical games: "
                f"{home_team.get('displayName')} "
                f"vs. "
                f"{away_team.get('displayName')}"
            )

            continue

        features = [
            home_stats["avg_scored"],
            home_stats["avg_allowed"],
            home_stats["win_rate"],
            away_stats["avg_scored"],
            away_stats["avg_allowed"],
            away_stats["win_rate"]
        ]

        home_probability = calculate_prediction(
            features
        )

        away_probability = 1 - home_probability

        if home_probability >= away_probability:

            predicted_winner = home_team.get(
                "displayName", "Home Team"
            )

            winning_probability = home_probability

        else:

            predicted_winner = away_team.get(
                "displayName", "Away Team"
            )

            winning_probability = away_probability

        local_game_time = event_time.tz_convert(
            "America/Phoenix"
        )

        predictions.append({

            "Game": (
                f"{away_team.get('displayName')} "
                f"@ "
                f"{home_team.get('displayName')}"
            ),

            "Away Team": away_team.get(
                "displayName", "Away"
            ),

            "Home Team": home_team.get(
                "displayName", "Home"
            ),

            "Predicted Winner": predicted_winner,

            "Win Probability": winning_probability,

            "Home Probability": home_probability,

            "Away Probability": away_probability,

            "Game Time": local_game_time.strftime(
                "%I:%M %p"
            ),

            "Home Games Used": home_stats["games_used"],

            "Away Games Used": away_stats["games_used"]

        })

    predictions.sort(
        key=lambda item: item["Win Probability"],
        reverse=True
    )

    return predictions, skipped


# =========================================================
# HOME PAGE
# =========================================================

if selected_page == "🏠 Home":

    st.subheader("Welcome to PARLAY KINGZ 👑")

    st.write(
        "Your college basketball prediction dashboard."
    )

    today = datetime.now(
        ZoneInfo("America/Phoenix")
    )

    st.caption(
        f"Arizona Date: {today.strftime('%B %d, %Y')}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Model Status",
        "ONLINE" if MODEL_READY else "OFFLINE"
    )

    col2.metric(
        "Historical Records",
        f"{len(raw_scores):,}"
    )

    col3.metric(
        "Model Features",
        len(REQUIRED_FEATURES)
    )

    st.divider()

    st.subheader("Dashboard Navigation")

    st.write(
        "Use the navigation buttons above to access "
        "today's predictions, model information, "
        "and historical data."
    )

    st.info(
        "Select Daily Picks to generate today's "
        "college basketball predictions."
    )


# =========================================================
# DAILY PICKS PAGE
# =========================================================

elif selected_page == "🎯 Daily Picks":

    st.subheader("🎯 Today's Daily Picks")

    today = datetime.now(
        ZoneInfo("America/Phoenix")
    )

    st.caption(
        today.strftime("%A, %B %d, %Y")
    )

    if st.button(
        "🔄 Refresh Today's Predictions",
        use_container_width=True
    ):

        get_daily_schedule.clear()

    try:

        with st.spinner(
            "Analyzing today's college basketball games..."
        ):

            predictions, skipped = generate_daily_picks()

        if not predictions:

            st.warning(
                "No eligible pregame predictions were generated."
            )

            st.write(
                "This can happen when there are no upcoming games, "
                "team names cannot be matched, or teams do not "
                "have enough historical data."
            )

        else:

            st.success(
                f"{len(predictions)} predictions generated."
            )

            st.divider()

            for number, pick in enumerate(
                predictions,
                start=1
            ):

                probability = (
                    pick["Win Probability"] * 100
                )

                st.markdown(
                    f"""
                    <div class="pick-card">
                        <div class="pick-label">
                            PARLAY KINGZ PICK #{number}
                        </div>
                        <div class="pick-team">
                            {pick["Predicted Winner"]}
                        </div>
                        <div class="pick-probability">
                            {probability:.1f}% model probability
                        </div>
                        <p>
                            {pick["Game"]}
                        </p>
                        <p>
                            Game Time: {pick["Game Time"]} Arizona
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with st.expander(
                    f"View matchup details - Pick #{number}"
                ):

                    st.write(
                        f"Home team: {pick['Home Team']}"
                    )

                    st.write(
                        f"Away team: {pick['Away Team']}"
                    )

                    st.write(
                        f"Home win probability: "
                        f"{pick['Home Probability']:.1%}"
                    )

                    st.write(
                        f"Away win probability: "
                        f"{pick['Away Probability']:.1%}"
                    )

                    st.write(
                        f"Home historical games used: "
                        f"{pick['Home Games Used']}"
                    )

                    st.write(
                        f"Away historical games used: "
                        f"{pick['Away Games Used']}"
                    )

            export_data = pd.DataFrame(predictions)

            st.download_button(
                "📥 Download Daily Picks CSV",
                data=export_data.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    f"PARLAY_KINGZ_PICKS_"
                    f"{today.strftime('%Y%m%d')}.csv"
                ),
                mime="text/csv",
                use_container_width=True
            )

        if skipped:

            with st.expander(
                f"Games skipped ({len(skipped)})"
            ):

                for reason in skipped:
                    st.write(f"• {reason}")

    except requests.RequestException as error:

        st.error(
            "ESPN schedule could not be reached."
        )

        st.code(str(error))

    except Exception as error:

        st.error(
            "The daily prediction engine encountered an issue."
        )

        st.code(str(error))


# =========================================================
# MODEL INFORMATION PAGE
# =========================================================

elif selected_page == "🧠 Model Information":

    st.subheader("🧠 Model Information")

    st.write(
        "PARLAY KINGZ uses its existing trained "
        "Random Forest model with a separate "
        "probability calibration model."
    )

    st.divider()

    st.markdown("### Model Architecture")

    st.write("Base model: Random Forest Classifier")

    st.write("Calibration: Logistic Regression")

    st.write("Prediction target: Home team win")

    st.write("Target value 1 = Home win")

    st.write("Target value 0 = Away win")

    st.divider()

    st.markdown("### Model Features")

    for feature in REQUIRED_FEATURES:
        st.write(f"• {feature}")

    st.divider()

    st.markdown("### Historical Test Results")

    col1, col2 = st.columns(2)

    col1.metric(
        "Test Accuracy",
        "65.37%"
    )

    col2.metric(
        "Balanced Accuracy",
        "64.37%"
    )

    col3, col4 = st.columns(2)

    col3.metric(
        "Brier Score",
        "0.2146"
    )

    col4.metric(
        "AUC",
        "0.7029"
    )

    st.info(
        "These are historical model evaluation results, "
        "not a guarantee of future accuracy or betting profit. "
        "A predicted probability is not a guaranteed outcome."
    )

    st.divider()

    st.markdown("### Model Backup")

    st.write(
        "The website loads the existing private model backup. "
        "It does not automatically retrain or overwrite the model."
    )

    st.write(
        f"Backup file found: {'Yes' if BACKUP_PATH.exists() else 'No'}"
    )


# =========================================================
# HISTORICAL DATA PAGE
# =========================================================

elif selected_page == "📊 Historical Data":

    st.subheader("📊 Historical Data")

    st.write(
        "Historical records used by the prediction engine."
    )

    st.metric(
        "Historical Team Records",
        f"{len(raw_scores):,}"
    )

    st.divider()

    st.markdown("### Recent Historical Records")

    display_columns = [
        column
        for column in [
            "game_date_time",
            "team_name",
            "team_location",
            "team_score",
            "opponent_team_score",
            "team_winner"
        ]
        if column in raw_scores.columns
    ]

    recent_data = raw_scores.sort_values(
        "game_date_time",
        ascending=False
    ).head(100)

    st.dataframe(
        recent_data[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "📥 Download Historical Data",
        data=raw_scores.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="PARLAY_KINGZ_HISTORICAL_DATA.csv",
        mime="text/csv",
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#d4af37;">
        👑 PARLAY KINGZ
    </div>
    <div style="text-align:center; color:#8b95a5; font-size:12px;">
        DATA • ANALYSIS • DISCIPLINE
    </div>
    """,
    unsafe_allow_html=True
)
