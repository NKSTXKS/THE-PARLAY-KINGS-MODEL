
import streamlit as st
import pandas as pd
import requests
import joblib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "PARLAY_KINGZ_NBA_UPGRADED_FINAL.joblib"
STATS_FILE = BASE_DIR / "PARLAY_KINGZ_LIVE_TEAM_STATS.csv"


@st.cache_resource
def load_nba_model():
    bundle = joblib.load(MODEL_FILE)
    return (
        bundle["model"],
        bundle["features"],
        float(bundle.get("threshold", 0.5))
    )


@st.cache_data
def load_nba_stats():
    df = pd.read_csv(STATS_FILE)

    required = [
        "team", "avg_scored", "avg_allowed", "win_rate",
        "games_played", "last5_scored", "last5_allowed",
        "last10_scored", "last10_allowed", "last10_win_rate",
        "last_game"
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing statistics columns: {missing}")

    df["team_key"] = df["team"].apply(normalize_team)
    return df


def normalize_team(name):
    name = str(name).lower().strip()

    aliases = {
        "la clippers": "los angeles clippers",
        "la lakers": "los angeles lakers",
        "ny knicks": "new york knicks",
        "gs warriors": "golden state warriors"
    }

    name = aliases.get(name, name)
    return "".join(ch for ch in name if ch.isalnum())


@st.cache_data(ttl=60)
def get_nba_schedule():
    today = datetime.now(
        ZoneInfo("America/Phoenix")
    ).strftime("%Y%m%d")

    url = (
        "https://site.api.espn.com/apis/site/v2/"
        "sports/basketball/nba/scoreboard"
    )

    response = requests.get(
        url,
        params={"dates": today},
        timeout=20
    )
    response.raise_for_status()

    data = response.json()
    games = []

    for event in data.get("events", []):
        status = event.get("status", {}).get("type", {}).get("state")

        # Only include games that have not started.
        if status != "pre":
            continue

        competitions = event.get("competitions", [])
        if not competitions:
            continue

        home = None
        away = None

        for competitor in competitions[0].get("competitors", []):
            team_name = competitor.get("team", {}).get("displayName")

            if competitor.get("homeAway") == "home":
                home = team_name
            elif competitor.get("homeAway") == "away":
                away = team_name

        if home and away:
            games.append({
                "date": event.get("date"),
                "home": home,
                "away": away
            })

    return games


def build_nba_features(home, away, game_date, features):
    game_time = pd.to_datetime(game_date, utc=True)
    home_last_game = pd.to_datetime(home["last_game"], utc=True)
    away_last_game = pd.to_datetime(away["last_game"], utc=True)

    home_rest = max(
        0,
        (game_time - home_last_game).total_seconds() / 86400
    )

    away_rest = max(
        0,
        (game_time - away_last_game).total_seconds() / 86400
    )

    values = {
        "home_avg_scored": home["avg_scored"],
        "home_avg_allowed": home["avg_allowed"],
        "home_win_rate": home["win_rate"],
        "away_avg_scored": away["avg_scored"],
        "away_avg_allowed": away["avg_allowed"],
        "away_win_rate": away["win_rate"],
        "home_last5_scored": home["last5_scored"],
        "home_last5_allowed": home["last5_allowed"],
        "home_last10_scored": home["last10_scored"],
        "home_last10_allowed": home["last10_allowed"],
        "home_last10_win_rate": home["last10_win_rate"],
        "home_games_played": home["games_played"],
        "home_rest_days": home_rest,
        "away_last5_scored": away["last5_scored"],
        "away_last5_allowed": away["last5_allowed"],
        "away_last10_scored": away["last10_scored"],
        "away_last10_allowed": away["last10_allowed"],
        "away_last10_win_rate": away["last10_win_rate"],
        "away_games_played": away["games_played"],
        "away_rest_days": away_rest,
        "home_court": 1
    }

    return pd.DataFrame(
        [[values[feature] for feature in features]],
        columns=features
    )


def render_nba_tab():
    st.header("👑 NBA Prediction Engine")
    st.caption("Powered by the PARLAY KINGZ upgraded 21-feature model")

    try:
        model, features, threshold = load_nba_model()
        stats = load_nba_stats()

        st.success("NBA model and team statistics loaded.")

        with st.spinner("Checking today's NBA schedule..."):
            games = get_nba_schedule()

        if not games:
            st.info("No upcoming NBA games found for today.")
            return

        predictions = []
        skipped = []

        for game in games:
            home_match = stats[
                stats["team_key"] == normalize_team(game["home"])
            ]

            away_match = stats[
                stats["team_key"] == normalize_team(game["away"])
            ]

            if home_match.empty or away_match.empty:
                skipped.append(
                    f'{game["away"]} at {game["home"]}'
                )
                continue

            home = home_match.iloc[0]
            away = away_match.iloc[0]

            X = build_nba_features(
                home, away, game["date"], features
            )

            probabilities = model.predict_proba(X)[0]
            classes = list(model.classes_)

            if 1 not in classes:
                raise ValueError(
                    "The NBA model does not contain class 1."
                )

            home_probability = float(
                probabilities[classes.index(1)]
            )
            away_probability = 1 - home_probability

            if home_probability >= threshold:
                pick = game["home"]
            else:
                pick = game["away"]

            game_time = pd.to_datetime(
                game["date"], utc=True
            ).tz_convert("America/Phoenix")

            predictions.append({
                "Away Team": game["away"],
                "Home Team": game["home"],
                "Model Pick": pick,
                "Away Win Probability": away_probability,
                "Home Win Probability": home_probability,
                "Arizona Tip-off": game_time.strftime("%I:%M %p")
            })

        if predictions:
            st.subheader("🏀 Today's NBA Matchups")

            for number, prediction in enumerate(predictions, 1):
                st.divider()
                st.subheader(
                    f'{number}. {prediction["Away Team"]} at '
                    f'{prediction["Home Team"]}'
                )

                st.markdown(
                    f'**👑 Model Pick: {prediction["Model Pick"]}**'
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Away Win Probability",
                    f'{prediction["Away Win Probability"]:.1%}'
                )

                col2.metric(
                    "Home Win Probability",
                    f'{prediction["Home Win Probability"]:.1%}'
                )

                st.caption(
                    f'Tip-off: {prediction["Arizona Tip-off"]} Arizona time'
                )

            st.caption(
                f"Model decision threshold: {threshold:.3f}. "
                "Probabilities are estimates, not guarantees."
            )

        else:
            st.warning(
                "The schedule loaded, but no games matched the team statistics."
            )

        if skipped:
            with st.expander("Games skipped because team names did not match"):
                for game in skipped:
                    st.write(game)

    except Exception as error:
        st.error(f"NBA engine error: {error}")
        st.info(
            "Your original college basketball app has not been changed."
        )
