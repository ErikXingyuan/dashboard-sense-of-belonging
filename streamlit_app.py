import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="HSLU Sense of Belonging (SoB)", layout="wide")

if "freitext_seed" not in st.session_state:
    st.session_state["freitext_seed"] = 42


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.6rem;
            padding-bottom: 1.5rem;
            max-width: 100%;
        }

        [data-testid="stSidebar"] {
            background-color: #f7f7f8;
            border-right: 1px solid #dddddd;
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
            font-size: 2.4rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.25rem;
            color: #1f1f1f;
        }

        .main-note {
            color: #5a5f66;
            font-size: 0.95rem;
            margin-top: 0.1rem;
            margin-bottom: 0.85rem;
        }

        .info-card {
            background: #f6f7f9;
            border: 1px solid #e3e5e8;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 1rem;
            color: #2f343a;
            font-size: 0.96rem;
            line-height: 1.5;
        }

        .info-card strong {
            color: #1f1f1f;
        }

        [data-baseweb="tab-list"] {
            display: grid !important;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            width: 100%;
            margin-bottom: 0.65rem;
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

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 0.75rem 0.9rem;
        }

        .section-title {
            font-size: 1.06rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            color: #1f1f1f;
        }

        .subsection-title {
            font-size: 0.98rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
            color: #1f1f1f;
        }

        .panel-marker {
            display: none;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.panel-marker) {
            background: #f6f7f9;
            border: 1px solid #e3e5e8 !important;
            border-radius: 14px;
            padding: 0.95rem 0.95rem 0.45rem 0.95rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULT_FILE = "data/Fragebogen_ Sense of Belonging im Studium (Responses).xlsx"
LOGO_FILE = "image/HSLU_Logo_DE_Schwarz_rgb.png"

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

DIMENSION_COLORS = {
    "Allgemein": "#5B7DB1",
    "Sozial": "#5FA8A1",
    "Akademisch": "#D9965B",
    "Vielfalt": "#7EAF68",
}

UI_COLORS = {
    "distribution": "#D9965B",
    "count": "#8E97A3",
    "neutral_blue": "#6F8FB8",
}

QUESTION_PREFIX = {
    "Allgemein": "A",
    "Sozial": "S",
    "Akademisch": "AK",
    "Vielfalt": "V",
}


def normalize(text):
    text = str(text or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def find_column(df, search_text):
    search_text = normalize(search_text)
    for col in df.columns:
        if search_text in normalize(col):
            return col
    return None


def find_columns(df, search_texts):
    return [col for text in search_texts if (col := find_column(df, text))]


def to_numeric(series):
    return pd.to_numeric(
        series.astype(str)
        .str.extract(r"(\d+(?:[.,]\d+)?)")[0]
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def get_options(series):
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    return sorted(values.unique().tolist(), key=lambda x: str(x).lower())


def apply_filter(df, col, selected_values):
    if not col or not selected_values:
        return df
    values = df[col].astype(str).str.strip()
    selected = {str(v).strip() for v in selected_values}
    return df[values.isin(selected)]


def comparison_orientation(label):
    return "h" if label in {"Studiengang", "Nebenjob", "Abschlussjahr"} else "v"


def question_label_map(tab_name, cols):
    prefix = QUESTION_PREFIX.get(tab_name, "Q")
    return {col: f"{prefix}{i + 1}" for i, col in enumerate(cols)}


def default_compare_label(options):
    order = ["Geschlecht", "Studiengang", "Alter", "Migrationshintergrund", "Nebenjob", "Abschlussjahr"]
    for label in order:
        if label in options:
            return label
    return next(iter(options)) if options else None


@st.cache_data
def load_data():
    if not os.path.exists(DEFAULT_FILE):
        return None

    df = pd.read_excel(DEFAULT_FILE)
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].replace({"": np.nan})

    return df


def add_scores(df):
    groups = {
        "Allgemein": find_columns(df, [
            "ich fühle mich an meiner hochschule willkommen",
            "ich habe das gefühl, dass ich als student",
            "ich empfinde ein zugehörigkeitsgefühl gegenüber meiner studienrichtung",
            "ich werde als person an der hochschule wahrgenommen",
            "ich identifiziere mich mit der kultur und den werten meiner hochschule",
        ]),
        "Sozial": find_columns(df, [
            "ich habe im studium freundschaften",
            "ich fühle mich in der studierendenschaft gut integriert",
            "ich habe mitstudierende, mit denen ich offen über herausforderungen sprechen kann",
            "in gruppenarbeiten oder im unterricht fühle ich mich ernst genommen",
            "ich weiss, wo ich soziale unterstützung an der hochschule finden kann",
            "ich habe im studium personen, an die ich mich bei persönlichen oder sozialen unsicherheiten wenden kann",
        ]),
        "Akademisch": find_columns(df, [
            "ich fühle mich im unterricht / in lehrveranstaltungen respektiert",
            "ich habe das gefühl, dass meine sichtweisen",
            "ich kann fragen stellen oder beiträge leisten, ohne mich unwohl zu fühlen",
            "die hochschule unterstützt mich in meiner persönlichen und akademischen entwicklung",
            "ich weiss, an wen ich mich bei fachlichen fragen oder unsicherheiten wenden kann",
        ]),
        "Vielfalt": find_columns(df, [
            "meine herkunft, sprache oder soziale situation werden an der hochschule respektiert",
            "ich habe das gefühl, dass vielfalt im studium wertgeschätzt wird",
            "ich kann mich im hochschulalltag authentisch zeigen",
        ]),
    }

    df = df.copy()
    all_numeric_cols = []

    for group_name, cols in groups.items():
        numeric_cols = []
        for col in cols:
            num_col = f"__num__{col}"
            df[num_col] = to_numeric(df[col])
            numeric_cols.append(num_col)
            all_numeric_cols.append(num_col)

        score_col = f"score_{group_name.lower()}"
        df[score_col] = df[numeric_cols].mean(axis=1) if numeric_cols else np.nan

    df["score_overall"] = df[all_numeric_cols].mean(axis=1) if all_numeric_cols else np.nan
    return df, groups


def style_fig(fig, height=320):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=True,
        legend_title_text="",
        title=dict(x=0, xanchor="left", font=dict(size=17, color="#1f1f1f")),
        font=dict(color="#1f1f1f", size=13),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#d7dbe0", title=None)
    fig.update_yaxes(gridcolor="#eceff3", zeroline=False, linecolor="#d7dbe0", title=None)
    return fig


def make_mean_bar(data, group_col, score_col, title, orientation="v", height=320, color=UI_COLORS["neutral_blue"]):
    if not group_col or score_col not in data.columns:
        return None

    temp = data[[group_col, score_col]].dropna().copy()
    if temp.empty:
        return None

    temp[group_col] = temp[group_col].astype(str).str.strip()

    chart_df = (
        temp.groupby(group_col, dropna=False)[score_col]
        .mean()
        .reset_index()
        .sort_values(score_col, ascending=(orientation == "h"))
    )

    fig = px.bar(
        chart_df,
        x=group_col if orientation == "v" else score_col,
        y=score_col if orientation == "v" else group_col,
        orientation=orientation,
        text_auto=".2f",
        title=title,
        color_discrete_sequence=[color],
    )

    fig = style_fig(fig, height)
    fig.update_layout(showlegend=False)

    if orientation == "v":
        fig.update_yaxes(range=[1, 5])
    else:
        fig.update_xaxes(range=[1, 5])

    return fig


def make_count_chart(data, group_col, title, orientation="v", height=320, color=UI_COLORS["count"]):
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
        color_discrete_sequence=[color],
    )

    fig = style_fig(fig, height)
    fig.update_layout(showlegend=False)
    return fig


def make_item_mean_chart(df, cols, title, color, labels_map):
    rows = []
    for col in cols:
        rows.append({
            "Label": labels_map[col],
            "Mittelwert": to_numeric(df[col]).mean(),
        })

    chart_df = pd.DataFrame(rows).dropna()
    if chart_df.empty:
        return None

    fig = px.bar(
        chart_df,
        x="Mittelwert",
        y="Label",
        orientation="h",
        text_auto=".2f",
        title=title,
        color_discrete_sequence=[color],
    )

    fig = style_fig(fig, 360)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[1, 5])
    return fig


def make_group_chart_for_value(df, cols, group_col, group_value, title, color, labels_map, height=360):
    if not group_col or group_col not in df.columns:
        return None

    temp_df = df[df[group_col].astype(str).str.strip() == str(group_value).strip()].copy()
    if temp_df.empty:
        return None

    rows = []
    for col in cols:
        rows.append({
            "Label": labels_map[col],
            "Mittelwert": to_numeric(temp_df[col]).mean(),
        })

    chart_df = pd.DataFrame(rows).dropna()
    if chart_df.empty:
        return None

    fig = px.bar(
        chart_df,
        x="Mittelwert",
        y="Label",
        orientation="h",
        text_auto=".2f",
        title=title,
        color_discrete_sequence=[color],
    )

    fig = style_fig(fig, height)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[1, 5])
    return fig


def make_distribution_chart(df, score_col, title):
    temp = df[[score_col]].dropna().copy()
    if temp.empty:
        return None

    temp["Bewertung"] = temp[score_col].round().clip(1, 5).astype(int).astype(str)
    chart_df = temp["Bewertung"].value_counts().sort_index().reset_index()
    chart_df.columns = ["Bewertung", "Anzahl"]

    fig = px.bar(
        chart_df,
        x="Bewertung",
        y="Anzahl",
        text_auto=True,
        title=title,
        color_discrete_sequence=[UI_COLORS["distribution"]],
    )

    fig = style_fig(fig, 320)
    fig.update_layout(showlegend=False)
    return fig


def make_boxplot(data, group_col, score_col, title, orientation="v", height=380, color=UI_COLORS["distribution"]):
    if not group_col or score_col not in data.columns:
        return None

    temp = data[[group_col, score_col]].dropna().copy()
    if temp.empty:
        return None

    temp[group_col] = temp[group_col].astype(str).str.strip()
    temp = temp[temp[group_col] != ""]
    if temp.empty:
        return None

    if orientation == "v":
        fig = px.box(
            temp,
            x=group_col,
            y=score_col,
            title=title,
            color_discrete_sequence=[color],
            points=False,
        )
        fig = style_fig(fig, height)
        fig.update_yaxes(range=[1, 5])
    else:
        fig = px.box(
            temp,
            x=score_col,
            y=group_col,
            title=title,
            color_discrete_sequence=[color],
            points=False,
        )
        fig = style_fig(fig, height)
        fig.update_xaxes(range=[1, 5])

    fig.update_layout(showlegend=False)
    return fig


def render_plot(fig):
    if fig is None:
        st.info("Für diese Ansicht sind keine passenden Daten vorhanden.")
    else:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_question_legend(tab_name, labels_map):
    with st.expander(f"Fragenübersicht {tab_name}"):
        for full_question, short_label in reversed(list(labels_map.items())):
            st.markdown(f"**{short_label}** — {full_question}")


def render_random_text_column(title, series, seed, min_quotes=1, max_quotes=3, max_len=180):
    st.markdown(f"<div class='subsection-title'>{title}</div>", unsafe_allow_html=True)

    if series is None:
        st.info("Keine Freitextantworten vorhanden.")
        return

    responses = [str(x).strip() for x in series.dropna().tolist() if str(x).strip()]
    if not responses:
        st.info("Keine Freitextantworten vorhanden.")
        return

    st.caption(f"{len(responses)} Antworten")

    sample_size = max(min_quotes, min(max_quotes, len(responses)))
    sampled = pd.Series(responses).sample(n=sample_size, random_state=seed).tolist()

    for quote in sampled:
        quote = re.sub(r"\s+", " ", quote)
        if len(quote) > max_len:
            quote = quote[:max_len - 3] + "..."
        st.markdown(f"> {quote}")


def build_data_basis_text(data, timestamp_col):
    if timestamp_col and timestamp_col in data.columns and data[timestamp_col].notna().any():
        start_date = data[timestamp_col].min().strftime("%d.%m.%Y")
        end_date = data[timestamp_col].max().strftime("%d.%m.%Y")

        if start_date == end_date:
            return f"Erhebungsdatum: {start_date}."

        return f"Erhebungszeitraum: {start_date} bis {end_date}."

    return "Ein Erhebungszeitraum kann nicht angezeigt werden, weil keine gültige Timestamp-Spalte gefunden wurde."


raw_df = load_data()

if raw_df is None:
    st.error(f"Datei nicht gefunden: {DEFAULT_FILE}. Lege die Excel-Datei in den Ordner data.")
    st.stop()

df, groups = add_scores(raw_df)

timestamp_col = find_column(df, "timestamp")
migration_col = find_column(df, "migrationshintergrund")
age_col = find_column(df, "wie alt sind sie")
gender_col = find_column(df, "wie identifizieren sie sich geschlechtlich")
year_col = find_column(df, "in welchem jahr haben sie ihr studium abgeschlossen")
work_col = find_column(df, "arbeiten sie neben dem studium")
program_col = find_column(df, "welchen studiengang studieren sie")

help_text_col = find_column(df, "was hat ihnen bisher geholfen, sich im studium zugehörig zu fühlen")
difficult_text_col = find_column(df, "wann oder in welchen situationen fühlen sie sich nicht zugehörig")
wish_text_col = find_column(df, "was wünschen sie, um sich an der hochschule wohler und integrierter zu fühlen")

if timestamp_col:
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df["Antwortjahr"] = df[timestamp_col].dt.year
else:
    df["Antwortjahr"] = np.nan


year_from = None
year_to = None

with st.sidebar:
    if os.path.exists(LOGO_FILE):
        st.image(LOGO_FILE, use_container_width=True)
    else:
        st.warning(f"Logo nicht gefunden: {LOGO_FILE}")

    st.markdown("### Filter")

    if df["Antwortjahr"].notna().any():
        available_years = sorted(df["Antwortjahr"].dropna().astype(int).unique().tolist())
        c1, c2 = st.columns(2)
        with c1:
            year_from = st.selectbox("Jahr von", available_years, index=0, key="year_from")
        with c2:
            year_to = st.selectbox("Jahr bis", available_years, index=len(available_years) - 1, key="year_to")

    study_program_selected = st.multiselect("Studiengang", get_options(df[program_col]) if program_col else [], key="filter_program") if program_col else []
    gender_selected = st.multiselect("Geschlecht", get_options(df[gender_col]) if gender_col else [], key="filter_gender") if gender_col else []
    age_selected = st.multiselect("Alter", get_options(df[age_col]) if age_col else [], key="filter_age") if age_col else []
    migration_selected = st.multiselect("Migrationshintergrund", get_options(df[migration_col]) if migration_col else [], key="filter_migration") if migration_col else []
    graduation_year_selected = st.multiselect("Abschlussjahr", get_options(df[year_col]) if year_col else [], key="filter_gradyear") if year_col else []
    work_selected = st.multiselect("Nebenjob", get_options(df[work_col]) if work_col else [], key="filter_work") if work_col else []


filtered_df = df.copy()
filtered_df = apply_filter(filtered_df, program_col, study_program_selected)
filtered_df = apply_filter(filtered_df, gender_col, gender_selected)
filtered_df = apply_filter(filtered_df, age_col, age_selected)
filtered_df = apply_filter(filtered_df, migration_col, migration_selected)
filtered_df = apply_filter(filtered_df, year_col, graduation_year_selected)
filtered_df = apply_filter(filtered_df, work_col, work_selected)

if year_from is not None and year_to is not None:
    if year_from <= year_to:
        filtered_df = filtered_df[filtered_df["Antwortjahr"].between(year_from, year_to, inclusive="both")]
    else:
        st.sidebar.warning("Jahr von muss kleiner oder gleich Jahr bis sein.")
        filtered_df = filtered_df.iloc[0:0]

if filtered_df.empty:
    st.error("Mit diesen Filtern gibt es keine Daten.")
    st.stop()


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

comparison_options = {}
if gender_col:
    comparison_options["Geschlecht"] = gender_col
if program_col:
    comparison_options["Studiengang"] = program_col
if age_col:
    comparison_options["Alter"] = age_col
if migration_col:
    comparison_options["Migrationshintergrund"] = migration_col
if work_col:
    comparison_options["Nebenjob"] = work_col
if year_col:
    comparison_options["Abschlussjahr"] = year_col

default_label = default_compare_label(comparison_options)
data_basis_text = build_data_basis_text(filtered_df, timestamp_col)


st.markdown("<div class='main-title'>HSLU Sense of Belonging (SoB)</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='main-note'>Interaktives Dashboard zur Visualisierung von Sense-of-Belonging-Daten im Departement Informatik.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class='info-card'>
        <strong>Interpretationshinweis:</strong> Höhere Werte zeigen eine stärkere Ausprägung des Sense of Belonging.
        5 = höchste Ausprägung des SoB, 1 = niedrigste Ausprägung.<br>
        <strong>Datenbasis:</strong> Anonymisierte Antworten aus der Google-Forms-Umfrage
        «Sense of Belonging im Studium». {data_basis_text}
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Übersicht", "Allgemein", "Sozial", "Akademisch", "Vielfalt"])


with tabs[0]:
    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Antworten", len(filtered_df))
    m2.metric("Ø SoB Gesamt", f"{overall_mean:.2f} / 5" if pd.notna(overall_mean) else "-")
    m3.metric("Stärkste SoB-Dimension", best_dimension)
    m4.metric("Niedrigste SoB-Dimension", weakest_dimension)

    if year_from is not None and year_to is not None:
        m5.metric("Antwortjahr", f"{year_from} - {year_to}")
    elif timestamp_col and filtered_df[timestamp_col].notna().any():
        m5.metric("Letzte Antwort", filtered_df[timestamp_col].max().strftime("%d.%m.%Y"))
    else:
        m5.metric("Studiengänge", filtered_df[program_col].nunique() if program_col else 0)

    upper_box = st.container(border=True)
    with upper_box:
        st.markdown("<div class='panel-marker'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Ergebnislage</div>", unsafe_allow_html=True)

        c1, c2 = st.columns([1.2, 1])

        with c1:
            score_df = pd.DataFrame({
                "Dimension": list(mean_scores.keys()),
                "Mittelwert": list(mean_scores.values()),
            }).dropna()

            if not score_df.empty:
                fig = px.bar(
                    score_df,
                    x="Dimension",
                    y="Mittelwert",
                    color="Dimension",
                    color_discrete_map=DIMENSION_COLORS,
                    text_auto=".2f",
                    title="Durchschnittlicher SoB pro Dimension",
                )
                fig = style_fig(fig, 320)
                fig.update_layout(showlegend=False)
                fig.update_yaxes(range=[1, 5])
                render_plot(fig)
            else:
                st.info("Keine Score-Daten vorhanden.")

        with c2:
            render_plot(make_distribution_chart(filtered_df, "score_overall", "Verteilung der SoB-Gesamtwerte"))

    avg_box = st.container(border=True)
    with avg_box:
        st.markdown("<div class='panel-marker'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Durchschnittlicher SoB nach demografischen Daten</div>", unsafe_allow_html=True)

        row1 = st.columns(3)
        with row1[0]:
            render_plot(make_mean_bar(filtered_df, gender_col, "score_overall", "Ø SoB nach Geschlecht"))
        with row1[1]:
            render_plot(make_mean_bar(filtered_df, age_col, "score_overall", "Ø SoB nach Alter"))
        with row1[2]:
            render_plot(make_mean_bar(filtered_df, migration_col, "score_overall", "Ø SoB nach Migrationshintergrund"))

        row2 = st.columns(3)
        with row2[0]:
            render_plot(make_mean_bar(filtered_df, program_col, "score_overall", "Ø SoB nach Studiengang", orientation="h"))
        with row2[1]:
            render_plot(make_mean_bar(filtered_df, work_col, "score_overall", "Ø SoB nach Nebenjob", orientation="h"))
        with row2[2]:
            render_plot(make_mean_bar(filtered_df, year_col, "score_overall", "Ø SoB nach Abschlussjahr", orientation="h"))

    count_box = st.container(border=True)
    with count_box:
        st.markdown("<div class='panel-marker'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Personenzahl nach demografischen Daten</div>", unsafe_allow_html=True)

        row1 = st.columns(3)
        with row1[0]:
            render_plot(make_count_chart(filtered_df, gender_col, "Geschlecht"))
        with row1[1]:
            render_plot(make_count_chart(filtered_df, age_col, "Alter"))
        with row1[2]:
            render_plot(make_count_chart(filtered_df, migration_col, "Migrationshintergrund"))

        row2 = st.columns(3)
        with row2[0]:
            render_plot(make_count_chart(filtered_df, program_col, "Studiengang", orientation="h"))
        with row2[1]:
            render_plot(make_count_chart(filtered_df, work_col, "Nebenjob", orientation="h"))
        with row2[2]:
            render_plot(make_count_chart(filtered_df, year_col, "Abschlussjahr", orientation="h"))

    text_box = st.container(border=True)
    with text_box:
        st.markdown("<div class='panel-marker'></div>", unsafe_allow_html=True)

        title_col, button_col = st.columns([6, 1])
        with title_col:
            st.markdown("<div class='section-title'>Freitext-Antworten</div>", unsafe_allow_html=True)
        with button_col:
            if st.button("Aktualisieren", key="refresh_freitext"):
                st.session_state["freitext_seed"] += 1

        col1, col2, col3 = st.columns(3)

        with col1:
            render_random_text_column(
                "Was hat Ihnen bisher geholfen, sich im Studium zugehörig zu fühlen?",
                filtered_df[help_text_col] if help_text_col else None,
                seed=st.session_state["freitext_seed"],
            )

        with col2:
            render_random_text_column(
                "Wann oder in welchen Situationen fühlen Sie sich nicht zugehörig?",
                filtered_df[difficult_text_col] if difficult_text_col else None,
                seed=st.session_state["freitext_seed"] + 1,
            )

        with col3:
            render_random_text_column(
                "Was wünschen Sie, um sich an der Hochschule wohler und integrierter zu fühlen?",
                filtered_df[wish_text_col] if wish_text_col else None,
                seed=st.session_state["freitext_seed"] + 2,
            )


def render_detail_tab(tab_name, score_col, question_cols, select_key):
    labels_map = question_label_map(tab_name, question_cols)

    result_box = st.container(border=True)
    with result_box:
        st.markdown("<div class='panel-marker'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Ergebnisse innerhalb der Dimension</div>", unsafe_allow_html=True)

        left, right = st.columns([1.35, 1])

        with left:
            render_plot(
                make_item_mean_chart(
                    filtered_df,
                    question_cols,
                    f"{tab_name}: durchschnittlicher SoB pro Frage",
                    color=DIMENSION_COLORS.get(tab_name, UI_COLORS["neutral_blue"]),
                    labels_map=labels_map,
                )
            )

        with right:
            render_plot(make_distribution_chart(filtered_df, score_col, f"{tab_name}: Verteilung der SoB-Werte"))

        render_question_legend(tab_name, labels_map)

    compare_box = st.container(border=True)
    with compare_box:
        st.markdown("<div class='panel-marker'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Demografische SoB-Gegenüberstellung</div>", unsafe_allow_html=True)

        selected_label = st.selectbox(
            "Demografie",
            list(comparison_options.keys()),
            index=list(comparison_options.keys()).index(default_label) if default_label else 0,
            key=select_key,
            label_visibility="collapsed",
        )
        selected_col = comparison_options[selected_label]
        orientation = comparison_orientation(selected_label)

        group_values = get_options(filtered_df[selected_col]) if selected_col else []

        if group_values:
            st.markdown(f"<div class='subsection-title'>Fragenvergleich nach {selected_label}</div>", unsafe_allow_html=True)

            cols_per_row = 2 if len(group_values) == 2 else 3

            for start in range(0, len(group_values), cols_per_row):
                row_values = group_values[start:start + cols_per_row]
                row_cols = st.columns(cols_per_row)

                for i, group_value in enumerate(row_values):
                    with row_cols[i]:
                        render_plot(
                            make_group_chart_for_value(
                                filtered_df,
                                question_cols,
                                selected_col,
                                group_value,
                                f"{group_value}",
                                color=DIMENSION_COLORS.get(tab_name, UI_COLORS["neutral_blue"]),
                                labels_map=labels_map,
                                height=340,
                            )
                        )

        summary_left, summary_right = st.columns([1.15, 1])

        with summary_left:
            render_plot(
                make_mean_bar(
                    filtered_df,
                    selected_col,
                    score_col,
                    f"{tab_name}: durchschnittlicher SoB nach {selected_label}",
                    orientation=orientation,
                    height=340,
                    color=DIMENSION_COLORS.get(tab_name, UI_COLORS["neutral_blue"]),
                )
            )

        with summary_right:
            render_plot(
                make_count_chart(
                    filtered_df,
                    selected_col,
                    f"{tab_name}: Personenzahl nach {selected_label}",
                    orientation=orientation,
                    height=340,
                    color=UI_COLORS["count"],
                )
            )

        render_plot(
            make_boxplot(
                filtered_df,
                selected_col,
                score_col,
                f"{tab_name}: SoB-Verteilung innerhalb der Gruppen ({selected_label})",
                orientation=orientation,
                height=380,
                color=UI_COLORS["distribution"],
            )
        )


with tabs[1]:
    render_detail_tab("Allgemein", "score_allgemein", groups["Allgemein"], "compare_allgemein")

with tabs[2]:
    render_detail_tab("Sozial", "score_sozial", groups["Sozial"], "compare_sozial")

with tabs[3]:
    render_detail_tab("Akademisch", "score_akademisch", groups["Akademisch"], "compare_akademisch")

with tabs[4]:
    render_detail_tab("Vielfalt", "score_vielfalt", groups["Vielfalt"], "compare_vielfalt")