import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="HSLU Sense of Belonging", layout="wide")

# ---------------------------------------------------
# STYLE
# ---------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0.8rem;
            padding-bottom: 1.5rem;
            max-width: 100%;
        }

        [data-testid="stSidebar"] {
            background-color: #f5f5f5;
            border-right: 1px solid #d9d9d9;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.1rem;
        }

        .sidebar-title {
            font-size: 1.45rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 1rem;
            color: #1f1f1f;
        }

        .main-title {
            font-size: 2.55rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.35rem;
            color: #202020;
        }

        [data-baseweb="tab-list"] {
            display: grid !important;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            width: 100%;
            margin-bottom: 0.6rem;
        }

        [data-baseweb="tab"] {
            width: 100% !important;
            justify-content: center !important;
            background: #3a3a3a;
            color: white;
            border-radius: 0;
            padding: 0.55rem 1rem;
            min-height: 42px;
            border: none;
        }

        [data-baseweb="tab"]:hover {
            background: #222222;
            color: white;
        }

        button[role="tab"][aria-selected="true"] {
            background: #111111 !important;
            color: white !important;
        }

        .main-note {
            color: #555555;
            font-size: 0.95rem;
            margin-top: 0.15rem;
            margin-bottom: 1rem;
        }

        .section-label {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
        }



        .box-title {
            font-size: 1.08rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
DEFAULT_FILE = "Fragebogen_ Sense of Belonging im Studium (Responses).xlsx"

SCREENSHOT_IGNORE_KEYWORDS = [
    "erste person in ihrer familie",
    "wichtige gründe für ihr studium",
    "familie ihre entscheidung beeinflusst",
    "wie stark treffen folgende herausforderungen auf sie zu",
    "wann empfanden sie ihr studium bisher als besonders herausfordernd",
    "welche unterstützungsangebote kennen sie",
    "welche angebote haben sie genutzt",
    "welche unterstützung hätten sie sich gewünscht",
    "was hilft ihnen im studium besonders",
    "familie kann mich bei studienbezogenen fragen unterstützen",
    "mehr leisten zu müssen als andere",
    "finanzielle sorgen beeinflussen mein studium",
]

STOPWORDS_DE = {
    "ich", "und", "der", "die", "das", "dass", "ist", "sind", "mit", "mich", "mir",
    "mein", "meine", "meiner", "meinem", "wir", "uns", "zu", "im", "in", "an", "auf",
    "für", "von", "bei", "nicht", "auch", "eine", "einer", "einem", "ein", "einen",
    "den", "dem", "des", "oder", "als", "aber", "man", "mehr", "sehr", "noch", "nur",
    "durch", "wenn", "werden", "wird", "studium", "hochschule", "hslu", "sich",
    "sein", "zum", "zur", "am", "da", "es", "vor", "nach", "über", "unter", "hat",
    "haben", "hilft", "fühle", "fühlen", "zugehörig", "wohler", "integrierter"
}

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def normalize(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def is_ignored_column(col_name):
    col_norm = normalize(col_name)
    return any(keyword in col_norm for keyword in SCREENSHOT_IGNORE_KEYWORDS)


def find_column(df, contains_text):
    contains_text = normalize(contains_text)
    for col in df.columns:
        if contains_text in normalize(col):
            return col
    return None


def find_columns(df, contains_texts):
    cols = []
    for text in contains_texts:
        col = find_column(df, text)
        if col:
            cols.append(col)
    return cols


def to_numeric_series(series):
    return pd.to_numeric(
        series.astype(str)
        .str.extract(r"(\d+(?:[.,]\d+)?)")[0]
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def shorten_question(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= 75:
        return text
    return text[:72] + "..."


def get_filter_options(series):
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    unique_values = sorted(values.unique().tolist(), key=lambda x: str(x).lower())
    return unique_values


def apply_single_filter(df, col, selected_values):
    if not col or not selected_values:
        return df
    series = df[col].astype(str).str.strip()
    selected_set = {str(v).strip() for v in selected_values}
    return df[series.isin(selected_set)]


@st.cache_data
def load_data():
    if not os.path.exists(DEFAULT_FILE):
        return None, None

    raw = pd.read_excel(DEFAULT_FILE)
    raw.columns = [str(c).strip() for c in raw.columns]

    for col in raw.columns:
        if raw[col].dtype == object:
            raw[col] = raw[col].replace({"": np.nan})

    ignored_columns = [col for col in raw.columns if is_ignored_column(col)]
    return raw, ignored_columns


def add_score_columns(df):
    general_cols = find_columns(df, [
        "ich fühle mich an meiner hochschule willkommen",
        "ich habe das gefühl, dass ich als student",
        "ich empfinde ein zugehörigkeitsgefühl gegenüber meiner studienrichtung",
        "ich werde als person an der hochschule wahrgenommen",
        "ich identifiziere mich mit der kultur und den werten meiner hochschule",
    ])

    social_cols = find_columns(df, [
        "ich habe im studium freundschaften",
        "ich fühle mich in der studierendenschaft gut integriert",
        "ich habe mitstudierende, mit denen ich offen über herausforderungen sprechen kann",
        "in gruppenarbeiten oder im unterricht fühle ich mich ernst genommen",
        "ich weiss, wo ich soziale unterstützung an der hochschule finden kann",
        "ich habe im studium personen, an die ich mich bei persönlichen oder sozialen unsicherheiten wenden kann",
    ])

    academic_cols = find_columns(df, [
        "ich fühle mich im unterricht / in lehrveranstaltungen respektiert",
        "ich habe das gefühl, dass meine sichtweisen",
        "ich kann fragen stellen oder beiträge leisten, ohne mich unwohl zu fühlen",
        "die hochschule unterstützt mich in meiner persönlichen und akademischen entwicklung",
        "ich weiss, an wen ich mich bei fachlichen fragen oder unsicherheiten wenden kann",
    ])

    diversity_cols = find_columns(df, [
        "meine herkunft, sprache oder soziale situation werden an der hochschule respektiert",
        "ich habe das gefühl, dass vielfalt im studium wertgeschätzt wird",
        "ich kann mich im hochschulalltag authentisch zeigen",
    ])

    groups = {
        "Allgemein": general_cols,
        "Sozial": social_cols,
        "Akademisch": academic_cols,
        "Vielfalt": diversity_cols,
    }

    df = df.copy()
    all_score_cols = []

    for group_name, cols in groups.items():
        numeric_cols = []
        for col in cols:
            num_col = f"__num__{col}"
            df[num_col] = to_numeric_series(df[col])
            numeric_cols.append(num_col)
            all_score_cols.append(num_col)

        df[f"score_{group_name.lower()}"] = df[numeric_cols].mean(axis=1) if numeric_cols else np.nan

    df["score_overall"] = df[all_score_cols].mean(axis=1) if all_score_cols else np.nan
    return df, groups


def make_mean_bar(data, x, y, title, orientation="v", height=320):
    if x is None or y is None or y not in data.columns:
        return None

    temp = data[[x, y]].dropna().copy()
    if temp.empty:
        return None

    temp[x] = temp[x].astype(str).str.strip()

    chart_df = (
        temp.groupby(x, dropna=False)[y]
        .mean()
        .reset_index()
        .sort_values(y, ascending=(orientation == "h"))
    )

    fig = px.bar(
        chart_df,
        x=x if orientation == "v" else y,
        y=y if orientation == "v" else x,
        orientation=orientation,
        text_auto=".2f",
        title=title,
    )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )

    if orientation == "v":
        fig.update_yaxes(range=[1, 5])
    else:
        fig.update_xaxes(range=[1, 5])

    return fig


def make_count_chart(data, group_col, title, orientation="v", height=320):
    if not group_col or group_col not in data.columns:
        return None

    temp = data[[group_col]].dropna().copy()
    if temp.empty:
        return None

    temp[group_col] = temp[group_col].astype(str).str.strip()
    temp = temp[temp[group_col] != ""]
    if temp.empty:
        return None

    chart_df = temp[group_col].value_counts().reset_index()
    chart_df.columns = [group_col, "Anzahl"]

    if orientation == "h":
        chart_df = chart_df.sort_values("Anzahl", ascending=True)

    fig = px.bar(
        chart_df,
        x=group_col if orientation == "v" else "Anzahl",
        y="Anzahl" if orientation == "v" else group_col,
        orientation=orientation,
        text_auto=True,
        title=title,
    )

    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig


def make_item_mean_chart(df, cols, title):
    rows = []
    for col in cols:
        num = to_numeric_series(df[col])
        rows.append({
            "Frage": shorten_question(col),
            "Mittelwert": num.mean(),
        })

    chart_df = pd.DataFrame(rows).dropna()
    if chart_df.empty:
        return None

    fig = px.bar(
        chart_df,
        x="Mittelwert",
        y="Frage",
        orientation="h",
        text_auto=".2f",
        title=title,
    )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(range=[1, 5])
    return fig


def make_distribution_chart(df, score_col, title):
    temp = df[[score_col]].dropna().copy()
    if temp.empty:
        return None

    temp["Bewertung"] = temp[score_col].round().clip(1, 5).astype(int).astype(str)
    chart_df = temp["Bewertung"].value_counts().sort_index().reset_index()
    chart_df.columns = ["Bewertung", "Anzahl"]

    fig = px.bar(chart_df, x="Bewertung", y="Anzahl", text_auto=True, title=title)
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def extract_keywords(df, text_columns, top_n=12):
    texts = []
    for col in text_columns:
        if col in df.columns:
            texts.extend(df[col].dropna().astype(str).tolist())

    tokens = []
    for text in texts:
        cleaned = re.sub(r"[^a-zA-ZäöüÄÖÜß\s-]", " ", text.lower())
        cleaned = cleaned.replace("-", " ")
        parts = cleaned.split()

        for token in parts:
            if len(token) >= 4 and token not in STOPWORDS_DE:
                tokens.append(token)

    counts = Counter(tokens).most_common(top_n)
    if not counts:
        return None

    chart_df = pd.DataFrame(counts, columns=["Wort", "Anzahl"]).sort_values("Anzahl", ascending=True)

    fig = px.bar(
        chart_df,
        x="Anzahl",
        y="Wort",
        orientation="h",
        text_auto=True,
        title="Häufige Begriffe aus Freitexten",
    )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def render_plot(fig):
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Für diese Ansicht sind keine passenden Daten vorhanden.")


# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
raw_df, ignored_columns = load_data()

if raw_df is None:
    st.error(
        f"Datei nicht gefunden: {DEFAULT_FILE}. "
        "Lege die Excel-Datei in denselben Ordner wie die App."
    )
    st.stop()

df, groups = add_score_columns(raw_df)

# ---------------------------------------------------
# COLUMN MAPPING
# ---------------------------------------------------
timestamp_col = find_column(df, "timestamp")
migration_col = find_column(df, "migrationshintergrund")
age_col = find_column(df, "wie alt sind sie")
gender_col = find_column(df, "wie identifizieren sie sich geschlechtlich")
year_col = find_column(df, "in welchem jahr haben sie ihr studium abgeschlossen")
work_col = find_column(df, "arbeiten sie neben dem studium")
program_col = find_column(df, "welchen studiengang studieren sie")

free_text_cols = [
    find_column(df, "was hat ihnen bisher geholfen, sich im studium zugehörig zu fühlen"),
    find_column(df, "wann oder in welchen situationen fühlen sie sich nicht zugehörig"),
    find_column(df, "was wünschen sie, um sich an der hochschule wohler und integrierter zu fühlen"),
]
free_text_cols = [c for c in free_text_cols if c]

if timestamp_col:
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df["Antwortjahr"] = df[timestamp_col].dt.year
else:
    df["Antwortjahr"] = np.nan

# ---------------------------------------------------
# SIDEBAR / FILTERS
# ---------------------------------------------------
year_from = None
year_to = None

with st.sidebar:
    st.markdown("<div class='sidebar-title'>HSLU Sense of Belonging</div>", unsafe_allow_html=True)
    st.markdown("### Filter")

    if "Antwortjahr" in df.columns and df["Antwortjahr"].notna().any():
        available_years = sorted(df["Antwortjahr"].dropna().astype(int).unique().tolist())

        st.markdown("#### Antwortjahr")
        c1, c2 = st.columns(2)
        with c1:
            year_from = st.selectbox("Jahr von", available_years, index=0, key="year_from")
        with c2:
            year_to = st.selectbox("Jahr bis", available_years, index=len(available_years) - 1, key="year_to")

    study_program_selected = st.multiselect(
        "Studiengang",
        get_filter_options(df[program_col]) if program_col else [],
        key="filter_program",
    ) if program_col else []

    gender_selected = st.multiselect(
        "Geschlecht",
        get_filter_options(df[gender_col]) if gender_col else [],
        key="filter_gender",
    ) if gender_col else []

    age_selected = st.multiselect(
        "Alter",
        get_filter_options(df[age_col]) if age_col else [],
        key="filter_age",
    ) if age_col else []

    migration_selected = st.multiselect(
        "Migrationshintergrund",
        get_filter_options(df[migration_col]) if migration_col else [],
        key="filter_migration",
    ) if migration_col else []

    graduation_year_selected = st.multiselect(
        "Abschlussjahr",
        get_filter_options(df[year_col]) if year_col else [],
        key="filter_gradyear",
    ) if year_col else []

    work_selected = st.multiselect(
        "Nebenjob",
        get_filter_options(df[work_col]) if work_col else [],
        key="filter_work",
    ) if work_col else []

filtered_df = df.copy()

filtered_df = apply_single_filter(filtered_df, program_col, study_program_selected)
filtered_df = apply_single_filter(filtered_df, gender_col, gender_selected)
filtered_df = apply_single_filter(filtered_df, age_col, age_selected)
filtered_df = apply_single_filter(filtered_df, migration_col, migration_selected)
filtered_df = apply_single_filter(filtered_df, year_col, graduation_year_selected)
filtered_df = apply_single_filter(filtered_df, work_col, work_selected)

if year_from is not None and year_to is not None:
    if year_from <= year_to:
        filtered_df = filtered_df[
            filtered_df["Antwortjahr"].between(year_from, year_to, inclusive="both")
        ]
    else:
        st.sidebar.warning("Jahr von muss kleiner oder gleich Jahr bis sein.")
        filtered_df = filtered_df.iloc[0:0]

if filtered_df.empty:
    st.error("Mit diesen Filtern gibt es keine Daten.")
    st.stop()

# ---------------------------------------------------
# SUMMARY
# ---------------------------------------------------
mean_scores = {
    "Allgemein": filtered_df["score_allgemein"].mean(),
    "Sozial": filtered_df["score_sozial"].mean(),
    "Akademisch": filtered_df["score_akademisch"].mean(),
    "Vielfalt": filtered_df["score_vielfalt"].mean(),
}

valid_mean_scores = {k: v for k, v in mean_scores.items() if pd.notna(v)}
best_dimension = max(valid_mean_scores, key=valid_mean_scores.get) if valid_mean_scores else "-"
weakest_dimension = min(valid_mean_scores, key=valid_mean_scores.get) if valid_mean_scores else "-"

overall_mean = filtered_df["score_overall"].mean()

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
st.markdown("<div class='main-title'>HSLU Sense of Belonging</div>", unsafe_allow_html=True)

tabs = st.tabs(["Übersicht", "Allgemein", "Sozial", "Akademisch", "Vielfalt"])

st.markdown(
    "<div class='main-note'><strong>Hinweis:</strong> 5 = beste Bewertung, 1 = schlechteste Bewertung.</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# OVERVIEW
# ---------------------------------------------------
with tabs[0]:
    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Antworten", len(filtered_df))
    m2.metric("Ø Gesamt", f"{overall_mean:.2f} / 5" if pd.notna(overall_mean) else "-")
    m3.metric("Stärkste Dimension", best_dimension)
    m4.metric("Schwächste Dimension", weakest_dimension)

    if year_from is not None and year_to is not None:
        m5.metric("Antwortjahr", f"{year_from} - {year_to}")
    elif timestamp_col and filtered_df[timestamp_col].notna().any():
        m5.metric("Letzte Antwort", filtered_df[timestamp_col].max().strftime("%d.%m.%Y"))
    else:
        m5.metric("Studiengänge", filtered_df[program_col].nunique() if program_col else 0)

    top1, top2, top3 = st.columns(3)

    with top1:
        score_df = pd.DataFrame({
            "Dimension": list(mean_scores.keys()),
            "Mittelwert": list(mean_scores.values()),
        }).dropna()

        if not score_df.empty:
            fig = px.bar(
                score_df,
                x="Dimension",
                y="Mittelwert",
                text_auto=".2f",
                title="Durchschnitt pro Dimension"
            )
            fig.update_yaxes(range=[1, 5])
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
            render_plot(fig)
        else:
            st.info("Keine Score-Daten vorhanden.")

    with top2:
        render_plot(make_distribution_chart(filtered_df, "score_overall", "Verteilung der Gesamtwerte"))

    with top3:
        keyword_fig = extract_keywords(filtered_df, free_text_cols)
        if keyword_fig:
            render_plot(keyword_fig)
        else:
            st.info("Keine Freitextdaten für die Keyword-Auswertung gefunden.")

    avg_box = st.container()
    with avg_box:
        st.markdown("<div class='gray-box-marker'></div>", unsafe_allow_html=True)
        st.markdown("<div class='box-title'>Durchschnitt nach demografischen Daten</div>", unsafe_allow_html=True)

        row1 = st.columns(3)
        with row1[0]:
            render_plot(make_mean_bar(filtered_df, gender_col, "score_overall", "Ø Gesamt nach Geschlecht"))
        with row1[1]:
            render_plot(make_mean_bar(filtered_df, age_col, "score_overall", "Ø Gesamt nach Alter"))
        with row1[2]:
            render_plot(make_mean_bar(filtered_df, migration_col, "score_overall", "Ø Gesamt nach Migrationshintergrund"))

        row2 = st.columns(2)
        with row2[0]:
            render_plot(make_mean_bar(filtered_df, program_col, "score_overall", "Ø Gesamt nach Studiengang", orientation="h"))
        with row2[1]:
            render_plot(make_mean_bar(filtered_df, work_col, "score_overall", "Ø Gesamt nach Nebenjob", orientation="h"))

    count_box = st.container()
    with count_box:
        st.markdown("<div class='yellow-box-marker'></div>", unsafe_allow_html=True)
        st.markdown("<div class='box-title'>Personenzahl nach demografischen Daten</div>", unsafe_allow_html=True)

        row1 = st.columns(3)
        with row1[0]:
            render_plot(make_count_chart(filtered_df, gender_col, "Personenzahl nach Geschlecht"))
        with row1[1]:
            render_plot(make_count_chart(filtered_df, age_col, "Personenzahl nach Alter"))
        with row1[2]:
            render_plot(make_count_chart(filtered_df, migration_col, "Personenzahl nach Migrationshintergrund"))

        row2 = st.columns(2)
        with row2[0]:
            render_plot(make_count_chart(filtered_df, program_col, "Personenzahl nach Studiengang", orientation="h"))
        with row2[1]:
            render_plot(make_count_chart(filtered_df, work_col, "Personenzahl nach Nebenjob", orientation="h"))

# ---------------------------------------------------
# DETAIL TABS
# ---------------------------------------------------
def render_detail_tab(tab_name, score_col, question_cols):
    top_left, top_right = st.columns([1.2, 1])

    with top_left:
        render_plot(make_item_mean_chart(filtered_df, question_cols, f"{tab_name}: Mittelwert pro Frage"))

    with top_right:
        render_plot(make_distribution_chart(filtered_df, score_col, f"{tab_name}: Verteilung"))

    row2 = st.columns(3)
    with row2[0]:
        render_plot(make_mean_bar(filtered_df, program_col, score_col, f"{tab_name} nach Studiengang"))
    with row2[1]:
        render_plot(make_mean_bar(filtered_df, gender_col, score_col, f"{tab_name} nach Geschlecht"))
    with row2[2]:
        render_plot(make_mean_bar(filtered_df, age_col, score_col, f"{tab_name} nach Alter"))

    row3 = st.columns(2)
    with row3[0]:
        render_plot(make_mean_bar(filtered_df, migration_col, score_col, f"{tab_name} nach Migrationshintergrund"))
    with row3[1]:
        render_plot(make_mean_bar(filtered_df, work_col, score_col, f"{tab_name} nach Nebenjob", orientation="h"))

with tabs[1]:
    render_detail_tab("Allgemein", "score_allgemein", groups["Allgemein"])

with tabs[2]:
    render_detail_tab("Sozial", "score_sozial", groups["Sozial"])

with tabs[3]:
    render_detail_tab("Akademisch", "score_akademisch", groups["Akademisch"])

with tabs[4]:
    render_detail_tab("Vielfalt", "score_vielfalt", groups["Vielfalt"])