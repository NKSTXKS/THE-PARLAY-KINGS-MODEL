
import streamlit as st
import joblib
import requests
import pandas as pd
import numpy as np
import re

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PARLAY KINGZ",
    page_icon="👑",
    layout="wide"
)

BACKUP_PATH = (
    Path(__file__).parent /
    "PARLAY_KINGZ_PRIVATE_BACKUP.joblib"
)

ESPN_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/scoreboard"
)


# ==========================================
# WEBSITE DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #080b12, #171b27);
    color: white;
}
h1, h2, h3 {
    color: #e5b93f;
}
.hero {
    background: linear-gradient(135deg, #20202a, #090b10);
    border: 1px solid #b99432;
    border-radius: 18px;
    padding: 35px 20px;
    text-align: center;
    margin-bottom: 25px;
}
.hero-title {
    color: #e5b93f;
    font-size: 42px;
    font-weight: 900;
    letter-spacing: 3px;
}
.hero-subtitle {
    color: white;
    font-size: 17px;
    letter-spacing: 2px;
}
.hero-description {
    color: #b8bdc9;
    font-size: 14px;
}
.section-card {
    background: #171d29;
    border: 1px solid #343b49;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.section-card h3 {
    color: #e5b93f;
}
div[data-testid="stMetric"] {
    background: #171d29;
    border: 1px solid #343b49;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# NAVIGATION
# ==========================================

page = st.sidebar.radio(
    "👑 PARLAY KINGZ",
    [
        "Home",
        "Daily Picks",
        "Model Information",
        "Historical Data"
    ]
)


# ==========================================
# LOAD EXISTING MODEL BACKUP
# ==========================================

@st.cache_resource
def load_model():

    backup = joblib.load(BACKUP_PATH)

    return (
        backup["base_model"],
        backup["calibrator"],
        backup["raw_scores"],
        backup["safe_games"],
        backup["master_ledger"],
        backup["features"]
    )


if not BACKUP_PATH.exists():
    st.error("Model backup file not found.")
    st.stop()

try:
    (
        model,
        calibrator,
        raw_scores,
        safe_games,
        ledger,
        features
    ) = load_model()

    features = list(features)

    required_features = [
        "home_avg_scored",
        "home_avg_allowed",
        "home_win_rate",
        "away_avg_scored",
        "away_avg_allowed",
        "away_win_rate"
    ]

    if features != required_features:
        st.error("Saved model feature order does not match.")
        st.stop()

    if model.n_features_in_ != len(features):
        st.error("Model feature count mismatch.")
        st.stop()

except Exception as error:
    st.error("Unable to load the existing model.")
    st.code(str(error))
    st.stop()


# ==========================================
# TEAM NAME HELPERS
# ==========================================

def normalize_team_name(name):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(name).lower()
    )


def build_team_lookup(data):

    name_columns = [
        "team_display_name",
        "team_name",
        "team_location",
        "team_short_display_name",
        "team_abbreviation",
        "team_nickname"
    ]

    lookup = {}

    for column in name_columns:

        if column not in data.columns:
            continue

        for team_id, name in zip(
            data["team_id"],
            data[column]
        ):

            if pd.notna(name):
                lookup[
                    normalize_team_name(name)
                ] = int(team_id)

    return lookup


# ==========================================
# ESPN SCHEDULE
# ==========================================

@st.cache_data(ttl=300)
def get_daily_schedule(date_value):

    response = requests.get(
        ESPN_URL,
        params={
            "dates": date_value,
            "groups": 50,
            "limit": 500
        },
        timeout=20
    )

    response.raise_for_status()

    return response.json()


# ==========================================
# SPORTSBOOK ODDS
# ==========================================

@st.cache_data(ttl=300)
def get_college_basketball_odds():

    api_key = st.secrets.get("ODDS_API_KEY")

    if not api_key:
        raise ValueError(
            "ODDS_API_KEY is missing from Streamlit Secrets."
        )

    url = (
        "https://api.the-odds-api.com/v4/sports/"
        "basketball_ncaab/odds"
    )

    response = requests.get(
        url,
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american"
        },
        timeout=20
    )

    response.raise_for_status()

    return response.json()


def build_odds_lookup():

    try:
        events = get_college_basketball_odds()
        lookup = {}

        for event in events:

            home_name = event.get("home_team", "")
            away_name = event.get("away_team", "")

            home_key = normalize_team_name(home_name)
            away_key = normalize_team_name(away_name)

            for bookmaker in event.get("bookmakers", []):

                for market in bookmaker.get("markets", []):

                    if market.get("key") != "h2h":
                        continue

                    prices = {}

                    for outcome in market.get("outcomes", []):

                        prices[
                            normalize_team_name(
                                outcome.get("name", "")
                            )
                        ] = outcome.get("price")

                    if home_key in prices and away_key in prices:

                        lookup[(away_key, home_key)] = {
                            "home_odds": prices[home_key],
                            "away_odds": prices[away_key],
                            "bookmaker": bookmaker.get(
                                "title",
                                "Sportsbook"
                            )
                        }

                if (away_key, home_key) in lookup:
                    break

        return lookup

    except Exception as error:
        st.warning(
            f"Sportsbook odds unavailable: {error}"
        )
        return {}


# ==========================================
# HISTORICAL TEAM FORM
# ==========================================

def get_team_form(team_id, game_time):

    history = raw_scores.copy()

    history["parsed_time"] = pd.to_datetime(
        history["game_date_time"],
        utc=True,
        errors="coerce"
    )

    cutoff = (
        pd.Timestamp(game_time) -
        pd.Timedelta(hours=3)
    )

    history = history[
        (history["team_id"] == team_id) &
        (history["parsed_time"] < cutoff)
    ].copy()

    history = history.sort_values(
        "parsed_time"
    ).tail(5)

    if len(history) < 3:
        return None

    points_allowed_column = next(
        (
            column
            for column in [
                "opponent_team_score",
                "team_points_allowed",
                "opponent_score"
            ]
            if column in history.columns
        ),
        None
    )

    if points_allowed_column is None:
        raise ValueError(
            "Historical points-allowed column not found."
        )

    return {
        "avg_scored": pd.to_numeric(
            history["team_score"],
            errors="coerce"
        ).mean(),

        "avg_allowed": pd.to_numeric(
            history[points_allowed_column],
            errors="coerce"
        ).mean(),

        "win_rate": history["team_winner"].astype(float).mean()
    }


# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">👑 PARLAY KINGZ</div>
        <div class="hero-subtitle">
            DATA. DISCIPLINE. DOMINANCE.
        </div>
        <br>
        <div class="hero-description">
            Your home for data-driven college basketball predictions.
            Powered by our existing trained prediction model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## WELCOME TO THE KINGDOM")

    st.write(
        "Access daily college basketball predictions, "
        "model probabilities, historical performance, "
        "and sportsbook market comparisons."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="section-card">
            <h3>🏀 DAILY PICKS</h3>
            <p>Today's model predictions</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-card">
            <h3>📊 MODEL DATA</h3>
            <p>Model probabilities and analysis</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="section-card">
            <h3>📈 PERFORMANCE</h3>
            <p>Historical model records</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("MODEL STATUS")

    col1, col2, col3 = st.columns(3)

    col1.metric("MODEL STATUS", "ONLINE")
    col2.metric("MODEL FEATURES", len(features))
    col3.metric(
        "HISTORICAL GAMES",
        f"{len(safe_games):,}"
    )


# ==========================================
# MODEL INFORMATION PAGE
# ==========================================

elif page == "Model Information":

    st.title("📊 MODEL INFORMATION")

    st.success("🟢 EXISTING MODEL LOADED")

    st.write("**Model:** Random Forest Classifier")
    st.write("**Calibration:** Sigmoid / Logistic Regression")
    st.write("**Training:** Existing trained model")
    st.write("**Automatic retraining:** Disabled")

    st.divider()

    st.subheader("MODEL FEATURES")

    for number, feature in enumerate(features, start=1):
        st.write(f"{number}. {feature}")

    st.divider()

    col1, col2 = st.columns(2)

    col1.metric("FEATURE COUNT", len(features))
    col2.metric("HISTORICAL GAMES", f"{len(safe_games):,}")

    st.info(
        "The model uses historical team performance "
        "to estimate the probability of a home-team win."
    )


# ==========================================
# HISTORICAL DATA PAGE
# ==========================================

elif page == "Historical Data":

    st.title("📚 HISTORICAL MODEL DATA")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "RAW SCORE RECORDS",
        f"{len(raw_scores):,}"
    )

    col2.metric(
        "SAFE GAME RECORDS",
        f"{len(safe_games):,}"
    )

    col3.metric(
        "LEDGER RECORDS",
        f"{len(ledger):,}"
    )

    st.divider()

    st.subheader("MASTER LEDGER")

    if isinstance(ledger, pd.DataFrame) and not ledger.empty:
        st.dataframe(
            ledger,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No master ledger records available.")

    st.divider()

    st.subheader("HISTORICAL SCORES")

    if isinstance(raw_scores, pd.DataFrame) and not raw_scores.empty:

        st.dataframe(
            raw_scores.tail(100),
            use_container_width=True,
            hide_index=True
        )

        csv_data = raw_scores.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Historical Scores",
            data=csv_data,
            file_name="PARLAY_KINGZ_HISTORICAL_SCORES.csv",
            mime="text/csv"
        )

    else:
        st.info("No historical score records available.")


# ==========================================
# DAILY PICKS PAGE
# ==========================================

elif page == "Daily Picks":

    st.title("🏀 PARLAY KINGZ — DAILY PICKS")

    today = datetime.now(
        ZoneInfo("America/Phoenix")
    )

    date_string = today.strftime("%Y%m%d")

    st.caption(
        f"Game date: {today.strftime('%B %d, %Y')} | Arizona time"
    )

    if st.button("🔄 REFRESH DAILY PICKS"):

        get_daily_schedule.clear()
        get_college_basketball_odds.clear()

        st.rerun()

    st.divider()

    try:

        with st.spinner("Analyzing today's games..."):

            schedule = get_daily_schedule(date_string)

            team_lookup = build_team_lookup(raw_scores)

            odds_lookup = build_odds_lookup()

            predictions = []
            skipped = []

            for event in schedule.get("events", []):

                status = (
                    event.get("status", {})
                    .get("type", {})
                    .get("state", "")
                )

                if status != "pre":
                    continue

                competition = event.get(
                    "competitions",
                    [{}]
                )[0]

                competitors = competition.get(
                    "competitors",
                    []
                )

                home = next(
                    (
                        team for team in competitors
                        if team.get("homeAway") == "home"
                    ),
                    None
                )

                away = next(
                    (
                        team for team in competitors
                        if team.get("homeAway") == "away"
                    ),
                    None
                )

                if not home or not away:
                    continue

                game_time = pd.to_datetime(
                    event.get("date"),
                    utc=True,
                    errors="coerce"
                )

                if pd.isna(game_time):
                    continue

                home_name = home.get(
                    "team", {}
                ).get("displayName", "Home")

                away_name = away.get(
                    "team", {}
                ).get("displayName", "Away")

                home_id = team_lookup.get(
                    normalize_team_name(home_name)
                )

                away_id = team_lookup.get(
                    normalize_team_name(away_name)
                )

                if home_id is None or away_id is None:

                    skipped.append(
                        f"{away_name} vs {home_name}: "
                        "team not matched"
                    )

                    continue

                home_form = get_team_form(
                    home_id,
                    game_time
                )

                away_form = get_team_form(
                    away_id,
                    game_time
                )

                if home_form is None or away_form is None:

                    skipped.append(
                        f"{away_name} vs {home_name}: "
                        "insufficient history"
                    )

                    continue

                feature_row = {
                    "home_avg_scored": home_form["avg_scored"],
                    "home_avg_allowed": home_form["avg_allowed"],
                    "home_win_rate": home_form["win_rate"],
                    "away_avg_scored": away_form["avg_scored"],
                    "away_avg_allowed": away_form["avg_allowed"],
                    "away_win_rate": away_form["win_rate"]
                }

                X_today = pd.DataFrame(
                    [[feature_row[column] for column in features]],
                    columns=features
                )

                base_probability = model.predict_proba(
                    X_today
                )[0][1]

                base_probability = np.clip(
                    base_probability,
                    0.000001,
                    0.999999
                )

                logit = np.log(
                    base_probability /
                    (1 - base_probability)
                )

                home_probability = calibrator.predict_proba(
                    np.array([[logit]])
                )[0][1]

                away_probability = 1 - home_probability

                favorite = (
                    home_name
                    if home_probability >= away_probability
                    else away_name
                )

                favorite_probability = max(
                    home_probability,
                    away_probability
                )

                odds_key = (
                    normalize_team_name(away_name),
                    normalize_team_name(home_name)
                )

                odds = odds_lookup.get(odds_key)

                home_odds = None
                away_odds = None
                home_edge = None
                away_edge = None
                home_roi = None
                away_roi = None

                sportsbook = "Unavailable"

                if odds:

                    home_odds = odds["home_odds"]
                    away_odds = odds["away_odds"]

                    sportsbook = odds["bookmaker"]

                    def implied_probability(price):

                        if price > 0:
                            return 100 / (price + 100)

                        return abs(price) / (
                            abs(price) + 100
                        )

                    home_market = implied_probability(
                        home_odds
                    )

                    away_market = implied_probability(
                        away_odds
                    )

                    total_market = (
                        home_market + away_market
                    )

                    home_market /= total_market
                    away_market /= total_market

                    home_edge = (
                        home_probability - home_market
                    )

                    away_edge = (
                        away_probability - away_market
                    )

                    home_decimal = (
                        1 + home_odds / 100
                        if home_odds > 0
                        else 1 + 100 / abs(home_odds)
                    )

                    away_decimal = (
                        1 + away_odds / 100
                        if away_odds > 0
                        else 1 + 100 / abs(away_odds)
                    )

                    home_roi = (
                        home_probability * home_decimal - 1
                    )

                    away_roi = (
                        away_probability * away_decimal - 1
                    )

                predictions.append({

                    "Game": f"{away_name} @ {home_name}",

                    "Model Favorite": favorite,

                    "Favorite Probability": favorite_probability,

                    "Away Probability": away_probability,

                    "Home Probability": home_probability,

                    "Sportsbook": sportsbook,

                    "Away Moneyline": away_odds,

                    "Home Moneyline": home_odds,

                    "Away Market Edge": away_edge,

                    "Home Market Edge": home_edge,

                    "Away Expected ROI": away_roi,

                    "Home Expected ROI": home_roi,

                    "Game Time (Arizona)": (
                        game_time
                        .tz_convert("America/Phoenix")
                        .strftime("%I:%M %p")
                    )

                })

        # ==================================
        # DISPLAY PREDICTIONS
        # ==================================

        if predictions:

            picks_df = pd.DataFrame(predictions)

            picks_df = picks_df.sort_values(
                "Favorite Probability",
                ascending=False
            )

            st.success(
                f"MODEL ONLINE — {len(picks_df)} "
                "eligible games analyzed"
            )

            st.subheader("👑 TODAY'S MODEL PICKS")

            percentage_columns = [
                "Favorite Probability",
                "Away Probability",
                "Home Probability",
                "Away Market Edge",
                "Home Market Edge",
                "Away Expected ROI",
                "Home Expected ROI"
            ]

            st.dataframe(
                picks_df.style.format(
                    {
                        column: "{:.2%}"
                        for column in percentage_columns
                    },
                    na_rep="—"
                ).format(
                    {
                        "Away Moneyline": "{:+.0f}",
                        "Home Moneyline": "{:+.0f}"
                    },
                    na_rep="—"
                ),
                use_container_width=True,
                hide_index=True
            )

            csv_data = picks_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "📥 DOWNLOAD TODAY'S PICKS",
                data=csv_data,
                file_name="PARLAY_KINGZ_DAILY_PICKS.csv",
                mime="text/csv"
            )

        else:

            st.warning(
                "No eligible upcoming games were found for today."
            )

        if skipped:

            with st.expander(
                f"Games skipped ({len(skipped)})"
            ):

                for reason in skipped:
                    st.write("• " + reason)

    except Exception as error:

        st.error(
            "The daily prediction engine encountered an issue."
        )

        st.code(str(error))


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "PARLAY KINGZ | Existing trained model | "
    "No automatic retraining"
)
