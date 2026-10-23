from datetime import date, time
from pathlib import Path
import geopy.geocoders
from geopy.geocoders import Nominatim


import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


MODEL_DIR = Path(__file__).resolve().parent / "saved_model"
FEATURE_FILE = MODEL_DIR / "feature_columns.joblib"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.joblib"
MODEL_FILE = MODEL_DIR / "accident_severity_model.pth"
HERO_IMAGE = Path(__file__).resolve().parent / "assets" / "accident_hero.jpg"
SCENE_IMAGE = Path(__file__).resolve().parent / "assets" / "accident_scene.jpg"

ROAD_FEATURES = {
    "Amenity": "Amenity",
    "Bump": "Bump",
    "Crossing": "Crossing",
    "Give_Way": "Give way",
    "Junction": "Junction",
    "No_Exit": "No exit",
    "Railway": "Railway",
    "Roundabout": "Roundabout",
    "Station": "Station",
    "Stop": "Stop",
    "Traffic_Calming": "Traffic calming",
    "Traffic_Signal": "Traffic signal",
    "Turning_Loop": "Turning loop",
}

SEVERITY_LABELS = {
    1: "Severity 1",
    2: "Severity 2",
    3: "Severity 3",
    4: "Severity 4",
}


class AccidentSeverityANN(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@st.cache_resource
def load_model_assets():
    missing_files = [
        path.name
        for path in (FEATURE_FILE, PREPROCESSOR_FILE, MODEL_FILE)
        if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing model artifact(s) in {MODEL_DIR}: {', '.join(missing_files)}"
        )

    feature_columns = joblib.load(FEATURE_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False
)
    model = AccidentSeverityANN(
        checkpoint["input_size"],
        checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return feature_columns, preprocessor, model


def get_category_options(preprocessor) -> dict[str, list[str]]:
    categorical_columns = preprocessor.transformers_[1][2]
    encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    return {
        column: [str(value) for value in categories]
        for column, categories in zip(categorical_columns, encoder.categories_)
    }


def predict_severity(input_values, feature_columns, preprocessor, model):
    raw_features = pd.DataFrame([input_values], columns=feature_columns)
    processed_features = preprocessor.transform(raw_features)
    if hasattr(processed_features, "toarray"):
        processed_features = processed_features.toarray()

    feature_tensor = torch.tensor(
        np.asarray(processed_features),
        dtype=torch.float32,
    )
    with torch.no_grad():
        logits = model(feature_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_severity = int(np.argmax(probabilities)) + 1
    return predicted_severity, probabilities


def render_styles():
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --navy: #061b34;
            --panel: #0b2948;
            --panel-soft: #0d3154;
            --line: #17476f;
            --blue: #1387ff;
            --cyan: #24c5e8;
            --text: #e8f3ff;
            --muted: #9bb9d3;
        }
        .stApp {
            background:
                radial-gradient(ellipse at 72% 0%, rgba(17, 89, 150, .22), transparent 40%),
                linear-gradient(145deg, #04162c 0%, #061d38 55%, #04172e 100%);
            color: var(--text);
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        #MainMenu, footer {
            visibility: hidden;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #061a33 0%, #04152a 100%);
            border-right: 1px solid #123554;
            min-width: 220px;
            max-width: 220px;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--muted);
        }
        .block-container {
            max-width: 1500px;
            padding-top: .7rem;
            padding-bottom: 2rem;
        }
        .brand-mark {
            color: #53dcf7;
            font-size: 1.7rem;
            line-height: 1;
        }
        .hero {
            min-height: 145px;
            padding: .15rem .35rem;
            border-radius: 16px;
        }
        .hero-kicker {
            display: inline-block;
            margin-bottom: .42rem;
            padding: .22rem .58rem;
            border: 1px solid #1d5a87;
            border-radius: 999px;
            color: #a7dcff;
            background: rgba(12, 76, 126, .48);
            font-size: .76rem;
        }
        .hero h1 {
            margin: 0;
            color: #f2f8ff;
            font-size: clamp(1.45rem, 2.4vw, 2.15rem);
            line-height: 1.08;
            letter-spacing: -.035em;
        }
        .hero h1 span {
            color: #2995ff;
        }
        .hero p {
            max-width: 540px;
            margin: .42rem 0 .55rem;
            color: #c4d9eb;
            font-size: .82rem;
        }
        .hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: .4rem;
            color: #b9e7ff;
            font-size: .68rem;
        }
        .hero-badges span {
            padding: .28rem .45rem;
            border: 1px solid rgba(60, 160, 219, .35);
            border-radius: 7px;
            background: rgba(5, 45, 79, .58);
        }
        [data-testid="stImage"] img {
            border: 1px solid rgba(59, 123, 169, .42);
            border-radius: 12px;
        }
        .section-kicker {
            color: #a8c7e2;
            font-size: .78rem;
            font-weight: 650;
            letter-spacing: .04em;
            text-transform: uppercase;
        }
        .result-banner {
            padding: 1rem 1.05rem;
            border: 1px solid rgba(47, 213, 186, .38);
            border-radius: 14px;
            background: linear-gradient(115deg, rgba(9, 94, 103, .82), rgba(8, 67, 83, .8));
        }
        .result-banner .result-label {
            color: #9be9dc;
            font-size: .78rem;
        }
        .result-banner .result-number {
            margin: .12rem 0;
            color: white;
            font-size: 1.65rem;
            font-weight: 750;
        }
        .result-banner .result-description {
            color: #c4eee9;
            font-size: .82rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--line);
            border-radius: 14px;
            background: rgba(8, 35, 62, .72);
        }
        div[data-testid="stMetric"] {
            padding: .5rem .62rem;
            border: 1px solid rgba(39, 101, 147, .52);
            border-radius: 11px;
            background: rgba(8, 46, 79, .64);
        }
        div[data-testid="stMetricLabel"] {
            color: #a9c7df;
        }
        div[data-testid="stMetricValue"] {
            color: #f0f7ff;
        }
        .stButton > button, .stFormSubmitButton > button {
            border: 1px solid #278cf4;
            border-radius: 10px;
            color: white;
            background: linear-gradient(100deg, #0878ef, #1695ff);
            font-weight: 700;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            border-color: #5abaff;
            color: white;
            background: linear-gradient(100deg, #0d87ff, #20a4ff);
        }
        [data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, #168bff, #19cfaa);
        }
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        [data-testid="stDateInput"] input,
        [data-testid="stTimeInput"] input {
            border-color: #1b4b73;
            background-color: #082747;
        }
        hr {
            border-color: #164366;
        }
        @media (max-width: 800px) {
            .hero {
                min-height: 0;
                padding: 1.1rem;
            }
            [data-testid="stSidebar"] {
                min-width: 0;
                max-width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div class="brand-mark">🚘 <strong>RoadSafe AI</strong></div>',
            unsafe_allow_html=True,
        )
        st.caption("Accident severity prediction")
        st.divider()
        page = st.radio(
            "Navigation",
            ["Home", "Predict", "Dataset Info", "About", "Contact"],
            label_visibility="collapsed",
            format_func=lambda item: {
                "Home": "⌂  Home",
                "Predict": "◉  Predict",
                "Dataset Info": "▤  Dataset info",
                "About": "ⓘ  About",
                "Contact": "✉  Contact",
            }[item],
        )
        st.divider()
        st.caption("Safer roads, brighter tomorrow.")
    return page


def render_hero():
    with st.container(border=True):
        copy_column, image_column = st.columns([1.2, 1], gap="small")
        with copy_column:
            st.markdown(
                """
                <div class="hero">
                    <div class="hero-kicker">US accidents dataset</div>
                    <h1>Road Accident<br><span>Severity Prediction</span></h1>
                    <p>
                        Use real-world accident data to help make roads safer.
                    </p>
                    <div class="hero-badges">
                        <span>Real US accident data</span>
                        <span>Deep Learning Model (ANN)</span>
                        <span>Safer roads for everyone</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with image_column:
            st.image(
                str(HERO_IMAGE),
                width="stretch",
                alt="Accident scene with emergency vehicles at dusk",
            )


def render_overview():
    with st.container(border=True):
        st.markdown("#### Dataset overview")
        metric_columns = st.columns(3, gap="small")
        metric_columns[0].metric("Records", "49,417")
        metric_columns[1].metric("Features", "28")
        metric_columns[2].metric("Classes", "4")
        st.caption("Severity classes 1–4")

def render_prediction_form(category_options):
    st.markdown("### Accident details")
    st.caption("Enter the available accident, weather, road, and time conditions.")
    # Location search: city or place name
    st.markdown("#### Choose accident location")

    if "accident_latitude" not in st.session_state:
        st.session_state["accident_latitude"] = 37.7749
        st.session_state["accident_longitude"] = -122.4194
        st.session_state["accident_place"] = "San Francisco, California"

    place_query = st.text_input(
        "City / Place / State",
        placeholder="Example: Kottakkal, Kerala, India",
        key="place_search_input",
    )

    if st.button("Search location", key="search_location_button"):
        if not place_query.strip():
            st.warning("Please enter a city or place name.")
        else:
            try:
                geolocator = Nominatim(
                    user_agent="roadsafe_ai_accident_severity_app"
                )
                found_place = geolocator.geocode(
                    place_query.strip(),
                    timeout=10,
                )

                if found_place is None:
                    st.warning(
                        "Location not found. Try a more specific place name."
                    )
                else:
                    st.session_state["accident_latitude"] = (
                        found_place.latitude
                    )
                    st.session_state["accident_longitude"] = (
                        found_place.longitude
                    )
                    st.session_state["accident_place"] = (
                        found_place.address
                    )
                    st.success(
                        f"Selected: {found_place.address}"
                    )
            except Exception:
                st.error(
                    "Could not find the location. Check your internet "
                    "connection and try again."
                )

    st.caption(
        f"Selected location: "
        f"{st.session_state['accident_place']}"
    )
    st.caption(
        f"Coordinates: "
        f"{st.session_state['accident_latitude']:.5f}, "
        f"{st.session_state['accident_longitude']:.5f}"
    )
    with st.form("accident_prediction_form"):
        with st.container(border=True):
            st.markdown("**1. Accident and location**")
            location_columns = st.columns(2)

            latitude = st.session_state["accident_latitude"]
            longitude = st.session_state["accident_longitude"]

            location_columns[0].text_input(
                "Selected place",
                value=st.session_state["accident_place"],
                disabled=True,
            )

            distance = location_columns[1].number_input(
                "Distance (miles)",
                min_value=0.0,
                value=0.1,
                step=0.1,
                format="%.2f",
            )
            date_columns = st.columns(2)
            start_date = date_columns[0].date_input(
                "Start date (UTC)",
                value=date.today(),
            )
            start_time = date_columns[1].time_input(
                "Start time (UTC)",
                value=time(12, 0),
            )

        with st.container(border=True):
            st.markdown("**2. Weather conditions**")
            weather_columns = st.columns(3)
            temperature = weather_columns[0].number_input(
                "Temperature (°F)", value=65.0, step=1.0
            )
            humidity = weather_columns[1].number_input(
                "Humidity (%)", min_value=0, max_value=100, value=50
            )
            pressure = weather_columns[2].number_input(
                "Pressure (in)", min_value=0.0, value=29.9, step=0.1
            )
            weather_columns = st.columns(3)
            visibility = weather_columns[0].number_input(
                "Visibility (miles)", min_value=0.0, value=10.0, step=0.5
            )
            wind_speed = weather_columns[1].number_input(
                "Wind speed (mph)", min_value=0.0, value=5.0, step=0.5
            )
            precipitation = weather_columns[2].number_input(
                "Precipitation (in)", min_value=0.0, value=0.0, step=0.1
            )
            weather_columns = st.columns(2)
            wind_options = category_options["Wind_Direction"]
            weather_options = category_options["Weather_Condition"]
            wind_direction = weather_columns[0].selectbox(
                "Wind direction",
                wind_options,
                index=wind_options.index("Calm")
                if "Calm" in wind_options
                else 0,
            )
            weather_condition = weather_columns[1].selectbox(
                "Weather condition",
                weather_options,
                index=weather_options.index("Clear")
                if "Clear" in weather_options
                else 0,
            )

        with st.container(border=True):
            st.markdown("**3. Road features**")
            road_columns = st.columns(4)
            road_values = {}
            for index, (feature, label) in enumerate(ROAD_FEATURES.items()):
                road_values[feature] = road_columns[index % 4].checkbox(
                    label,
                    value=False,
                    key=f"road_feature_{feature}",
                )

        with st.container(border=True):
            st.markdown("**4. Date and time features**")
            time_columns = st.columns(4)
            sunrise_sunset = time_columns[0].selectbox(
                "Sunrise / sunset",
                category_options["Sunrise_Sunset"],
                index=category_options["Sunrise_Sunset"].index("Day")
                if "Day" in category_options["Sunrise_Sunset"]
                else 0,
            )
            time_columns[1].metric("Start hour", f"{start_time.hour:02d}")
            time_columns[2].metric("Start month", f"{start_date.month}")
            time_columns[3].metric("Day of week", start_date.strftime("%A"))

        input_values = {
            "Start_Lat": latitude,
            "Start_Lng": longitude,
            "Distance(mi)": distance,
            "Temperature(F)": temperature,
            "Humidity(%)": humidity,
            "Pressure(in)": pressure,
            "Visibility(mi)": visibility,
            "Wind_Direction": wind_direction,
            "Wind_Speed(mph)": wind_speed,
            "Precipitation(in)": precipitation,
            "Weather_Condition": weather_condition,
            **{
                feature: "True" if value else "False"
                for feature, value in road_values.items()
            },
            "Sunrise_Sunset": sunrise_sunset,
            "Start_Hour": start_time.hour,
            "Start_Month": start_date.month,
            "Start_DayOfWeek": start_date.weekday(),
            "Selected_Place":st.session_state["accident_place"],
        }        
        

        submitted = st.form_submit_button(
            "✦  Predict severity",
            width="stretch",
        )

    return submitted, input_values, start_date, start_time


def render_prediction_result():
    st.markdown("### Prediction result")
    result = st.session_state.get("prediction_result")
    if result is None:
        with st.container(border=True):
            st.markdown("**Prediction ready**")
            st.caption(
                "Complete the form and select Predict severity to see the "
                "model result and class probabilities."
            )
    else:
        severity = result["severity"]
        probabilities = result["probabilities"]
        input_values = result["input_values"]
        st.markdown(
            f"""
            <div class="result-banner">
                <div class="result-label">Predicted severity</div>
                <div class="result-number">{SEVERITY_LABELS[severity]}</div>
                <div class="result-description">Model class {severity} of 4</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        with st.container(border=True):
            st.markdown("**Class probabilities**")
            for class_index, probability in enumerate(probabilities, start=1):
                probability_columns = st.columns([1.1, 3.5, 0.7])
                probability_columns[0].caption(SEVERITY_LABELS[class_index])
                probability_columns[1].progress(float(probability))
                probability_columns[2].caption(f"{probability:.1%}")

        with st.container(border=True):
            st.markdown("**Submitted conditions**")
            st.caption(
                f"Location · {input_values.get('Selected_Place', 'Unknown')}"
            )
            st.caption(
                f"Weather · {input_values['Weather_Condition']} · "
                f"{input_values['Wind_Direction']}"
            )
            st.caption(f"Time · {result['start_datetime']} UTC")

        st.caption(
            "Predictions are estimates from the trained model and are not a "
            "substitute for official accident assessments."
        )

    map_values = (
        result["input_values"]
        if result is not None
        else {"Start_Lat": 37.7749, "Start_Lng": -122.4194}
    )
    with st.container(border=True):
        map_column, scene_column = st.columns(2, gap="small")
        with map_column:
            st.markdown("**Accident location**")
            location = pd.DataFrame(
                {
                    "latitude": [map_values["Start_Lat"]],
                    "longitude": [map_values["Start_Lng"]],
                }
            )
            st.map(
                location,
                latitude="latitude",
                longitude="longitude",
                zoom=11,
                height=150,
                alt="Map showing the accident location",
            )
        with scene_column:
            st.markdown("**Accident image**")
            st.image(
                str(SCENE_IMAGE),
                width="stretch",
                alt="Two vehicles involved in an accident",
            )


def render_dashboard(feature_columns, preprocessor, model):
    hero_column, overview_column = st.columns([2.6, 1], gap="small")
    with hero_column:
        render_hero()
    with overview_column:
        render_overview()

    st.write("")
    category_options = get_category_options(preprocessor)
    form_column, result_column = st.columns([1.05, 0.95], gap="medium")
    with form_column:
        submitted, input_values, start_date, start_time = render_prediction_form(
            category_options
        )
    if submitted:
        severity, probabilities = predict_severity(
            input_values,
            feature_columns,
            preprocessor,
            model,
        )
        st.session_state["prediction_result"] = {
            "severity": severity,
            "probabilities": probabilities,
            "input_values": input_values,
            "start_datetime": (
                f"{start_date.isoformat()} {start_time.strftime('%H:%M')}"
            ),
        }
    with result_column:
        render_prediction_result()


def render_dataset_info(feature_columns, preprocessor):
    st.title("Dataset information")
    st.write(
        "The model uses the US Accidents data described in the training "
        "notebook. The source dataset contains 7,728,394 records and 46 "
        "original columns; this model uses a selected set of 28 input features."
    )
    metrics = st.columns(3)
    metrics[0].metric("Source records", "7,728,394")
    metrics[1].metric("Model inputs", len(feature_columns))
    metrics[2].metric("Severity classes", "4")
    st.subheader("Model input columns")
    st.dataframe(
        pd.DataFrame({"Feature": feature_columns}),
        hide_index=True,
        height=520,
    )
    st.caption(
        "Numeric values are imputed and scaled; categorical values are "
        "imputed and one-hot encoded by the saved preprocessing pipeline."
    )
    st.caption(
        f"Categorical input fields: "
        f"{len(preprocessor.transformers_[1][2])}"
    )


def render_about():
    st.title("About this project")
    st.write(
        "RoadSafe AI provides a front end for the saved accident severity "
        "model in this project. It submits the same model features used "
        "during training, applies the saved preprocessing pipeline, and "
        "displays the predicted severity class with all four class "
        "probabilities."
    )
    st.info(
        "The model output is an estimate based on its training data. "
        "Actual severity can differ based on real-world conditions."
    )


def render_contact():
    st.title("Contact")
    st.write(
        "This project does not currently define contact details or a support "
        "endpoint. Add your preferred contact information here when available."
    )


def main():
    st.set_page_config(
        page_title="RoadSafe AI | Accident Severity",
        page_icon="🚘",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_styles()
    page = render_sidebar()
    feature_columns, preprocessor, model = load_model_assets()

    if page in {"Home", "Predict"}:
        render_dashboard(feature_columns, preprocessor, model)
    elif page == "Dataset Info":
        render_dataset_info(feature_columns, preprocessor)
    elif page == "About":
        render_about()
    else:
        render_contact()


main()