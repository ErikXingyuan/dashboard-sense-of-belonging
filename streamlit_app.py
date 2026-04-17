import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(
    page_title="HSLU Sense of Belonging",
    layout="wide",
)

# ---------------------------------------------------
# CSS
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

        [data-baseweb="tab-list"] {
            gap: 10px;
            margin-bottom: 1rem;
        }

        [data-baseweb="tab"] {
            background: #3a3a3a;
            color: white;
            border-radius: 0;
            padding: 0.45rem 1.2rem;
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

        .dashboard-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        .section-title {
            font-size: 1rem;
            font-weight: 700;
            margin-top: 0.2rem;
            margin-bottom: 0.35rem;
        }

        .small-label {
            font-size: 0.85rem;
            color: #666666;
            margin-bottom: 0.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------
def make_bar_values(seed: int, n: int = 8):
    rng = np.random.default_rng(seed)
    return rng.integers(20, 95, size=n)

def make_pie_values(seed: int):
    rng = np.random.default_rng(seed)
    a = int(rng.integers(20, 45))
    b = 100 - a
    return [a, b]

# ---------------------------------------------------
# CHART HELPERS
# ---------------------------------------------------
def bar_chart(seed: int, height: int = 220):
    values = make_bar_values(seed)
    labels = [f"C{i+1}" for i in range(len(values))]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color="#d9d9d9",
            marker_line_color="#4a4a4a",
            marker_line_width=1.1,
        )
    )

    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )

    fig.update_xaxes(showgrid=False, showticklabels=False, title=None, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, title=None, zeroline=False)

    return fig

def pie_chart(seed: int, height: int = 220):
    values = make_pie_values(seed)

    fig = go.Figure(
        data=[
            go.Pie(
                values=values,
                hole=0,
                sort=False,
                textinfo="none",
                marker=dict(
                    colors=["#bdbdbd", "#efefef"],
                    line=dict(color="#4a4a4a", width=1.1),
                ),
            )
        ]
    )

    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        showlegend=False,
    )

    return fig

# ---------------------------------------------------
# DEMO EXPORT
# ---------------------------------------------------
def get_export_data():
    return pd.DataFrame(
        {
            "department": ["Informatik", "Wirtschaftsinformatik", "Data Science"],
            "score": [72, 68, 75],
            "responses": [45, 38, 41],
        }
    )

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.markdown("### HSLU Sense of Belonging")

    col_y1, col_y2 = st.columns(2)
    with col_y1:
        year_from = st.text_input("Jahr von", placeholder="YYYY")
    with col_y2:
        year_to = st.text_input("Jahr bis", placeholder="YYYY")

    department = st.selectbox(
        "Department",
        ["Select...", "Informatik", "Wirtschaftsinformatik", "Data Science"],
    )

    gender = st.selectbox(
        "Gender",
        ["Select...", "Male", "Female", "Other"],
    )

    age = st.selectbox(
        "Alter",
        ["Select...", "18-22", "23-27", "28+"],
    )

    migration = st.selectbox(
        "Migrationshintergrund",
        ["Select...", "Ja", "Nein", "Keine Angabe"],
    )

    st.button("Reset")

# ---------------------------------------------------
# HEADER + EXPORT
# ---------------------------------------------------
export_df = get_export_data()
csv_buffer = StringIO()
export_df.to_csv(csv_buffer, index=False)

top_left, top_right = st.columns([12, 1.2])

with top_left:
    st.markdown("<div class='dashboard-title'>Dashboard</div>", unsafe_allow_html=True)

with top_right:
    st.download_button(
        label="Export",
        data=csv_buffer.getvalue(),
        file_name="sense_of_belonging_export.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tabs = st.tabs(
    [
        "Overview",
        "Department",
        "Gender",
        "Alter",
        "Migrationshintergrund",
    ]
)

# ---------------------------------------------------
# TAB 1
# ---------------------------------------------------
with tabs[0]:
    metric_cols = st.columns(5)
    metrics = [
        ("Antworten", "124"),
        ("Present Score", "72%"),
        ("Zuordnung", "89%"),
        ("Studentische Daten", "4 Gruppen"),
        ("Status", "Aktiv"),
    ]

    for col, (label, value) in zip(metric_cols, metrics):
        with col:
            st.metric(label, value)

    row1 = st.columns([2.2, 1.1, 2.2, 1.1])

    with row1[0]:
        st.markdown("<div class='section-title'>Allgemein</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-label'>Overview</div>", unsafe_allow_html=True)
        st.plotly_chart(bar_chart(1), use_container_width=True, config={"displayModeBar": False})

    with row1[1]:
        st.markdown("<div class='section-title'>Verteilung</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-label'>select...</div>", unsafe_allow_html=True)
        st.plotly_chart(pie_chart(2), use_container_width=True, config={"displayModeBar": False})

    with row1[2]:
        st.markdown("<div class='section-title'>Akademisch</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-label'>Overview</div>", unsafe_allow_html=True)
        st.plotly_chart(bar_chart(3), use_container_width=True, config={"displayModeBar": False})

    with row1[3]:
        st.markdown("<div class='section-title'>Verteilung</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-label'>select...</div>", unsafe_allow_html=True)
        st.plotly_chart(pie_chart(4), use_container_width=True, config={"displayModeBar": False})

    row2 = st.columns([2.2, 1.1, 2.2, 1.1])

    with row2[0]:
        st.markdown("<div class='section-title'>Sozial</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-label'>Scores</div>", unsafe_allow_html=True)
        st.plotly_chart(bar_chart(5), use_container_width=True, config={"displayModeBar": False})

    with row2[1]:
        st.markdown("<div class='section-title'>Verteilung</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-label'>select...</div>", unsafe_allow_html=True)
        st.plotly_chart(pie_chart(6), use_container_width=True, config={"displayModeBar": False})

    with row2[2]:
        st.markdown("<div class='section-title'>Wohlbefinden</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-label'>Scores</div>", unsafe_allow_html=True)
        st.plotly_chart(bar_chart(7), use_container_width=True, config={"displayModeBar": False})

    with row2[3]:
        st.markdown("<div class='section-title'>Verteilung</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-label'>select...</div>", unsafe_allow_html=True)
        st.plotly_chart(pie_chart(8), use_container_width=True, config={"displayModeBar": False})

    row3 = st.columns([2.2, 1.1, 1.1, 1.1, 1.1])

    with row3[0]:
        st.markdown("<div class='section-title'>Weitere Kategorie</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-label'>Scores</div>", unsafe_allow_html=True)
        st.plotly_chart(bar_chart(9), use_container_width=True, config={"displayModeBar": False})

    with row3[1]:
        st.markdown("<div class='section-title'>Verteilung</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-label'>select...</div>", unsafe_allow_html=True)
        st.plotly_chart(pie_chart(10), use_container_width=True, config={"displayModeBar": False})

    with row3[2]:
        st.markdown("<div class='section-title'>Open Answers</div>", unsafe_allow_html=True)
        st.plotly_chart(pie_chart(11), use_container_width=True, config={"displayModeBar": False})

    with row3[3]:
        st.markdown("<div class='section-title'>Open Answers</div>", unsafe_allow_html=True)
        st.plotly_chart(pie_chart(12), use_container_width=True, config={"displayModeBar": False})

    with row3[4]:
        st.markdown("<div class='section-title'>Open Answers</div>", unsafe_allow_html=True)
        st.plotly_chart(pie_chart(13), use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------
# DETAIL TABS
# 4 REIHEN × 5 CHARTS
# ---------------------------------------------------
def render_detail_tab(title: str, seed_base: int):
    st.markdown(f"<div class='dashboard-title'>{title}</div>", unsafe_allow_html=True)

    row_titles = [
        "Department",
        "Gender",
        "Alter",
        "Migrationshintergrund",
    ]

    for row_index, row_title in enumerate(row_titles):
        st.markdown(f"<div class='section-title'>{row_title}</div>", unsafe_allow_html=True)
        cols = st.columns(5)

        for col_index, col in enumerate(cols):
            with col:
                seed = seed_base + row_index * 10 + col_index
                st.plotly_chart(
                    bar_chart(seed, height=190),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

with tabs[1]:
    render_detail_tab("Department", 100)

with tabs[2]:
    render_detail_tab("Gender", 200)

with tabs[3]:
    render_detail_tab("Alter", 300)

with tabs[4]:
    render_detail_tab("Migrationshintergrund", 400)