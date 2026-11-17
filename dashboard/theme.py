import streamlit as st


PAGE_GROUPS = {
    "OVERVIEW": [("Dashboard Overview", "🌱")],
    "DATA MANAGEMENT": [
        ("Dataset Explorer", "🗂️"),
        ("Crop Dataset", "🌾"),
        ("Hyperspectral Dataset", "🌈"),
        ("Pest Image Dataset", "🐛"),
    ],
    "ANALYSIS & AI": [
        ("Crop & Soil Analysis", "🧪"),
        ("Hyperspectral Analysis", "📡"),
        ("Pest Image Analysis", "🔎"),
        ("Unsupervised ML Lab", "🧠"),
    ],
    "REPORTS": [
        ("Analytics & Visualizations", "📊"),
        ("Reports & Downloads", "📥"),
    ],
}


def navigate_to(page):
    st.session_state["active_page"] = page


def render_sidebar():
    active_page = st.session_state.get("active_page", "Dashboard Overview")
    with st.sidebar:
        st.markdown("## 🌱 AgriVision AI")
        st.caption("Intelligent Agricultural Analytics")
        st.divider()
        for group, pages in PAGE_GROUPS.items():
            st.markdown(f'<p class="av-nav-heading">{group}</p>', unsafe_allow_html=True)
            for title, icon in pages:
                st.button(
                    f"{icon}  {title}",
                    key=f"navigation_{title}",
                    on_click=navigate_to,
                    args=(title,),
                    type="primary" if title == active_page else "secondary",
                    width="stretch",
                )
                if title == active_page:
                    st.session_state["active_page"] = title
        st.divider()
        st.markdown(
            '<div class="av-sidebar-footer"><strong>● Ready</strong>'
            '<span>Version 1.0</span></div>',
            unsafe_allow_html=True,
        )
    return active_page


def apply_theme():
    st.markdown(
        """
        <style>
        :root {
            --forest: #102B24;
            --emerald: #16845B;
            --mint: #e8f4ec;
            --paper: #F7F9F6;
            --ink: #1c2b23;
            --muted: #68766d;
        }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(247,249,246,.92); }
        [data-testid="stMain"] {
            color: var(--ink);
            color-scheme: light;
        }
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4,
        [data-testid="stMain"] p,
        [data-testid="stMain"] li,
        [data-testid="stMain"] label,
        [data-testid="stMain"] [data-testid="stWidgetLabel"] *,
        [data-testid="stMain"] [data-testid="stCaptionContainer"] * {
            color: var(--ink);
        }
        [data-testid="stMain"] [data-testid="stCaptionContainer"] * {
            color: #52675c;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #102B24 0%, #0c211c 100%);
            min-width: 260px;
            max-width: 280px;
        }
        [data-testid="stSidebar"] * { color: #eef6ef; }
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
            background: transparent;
            border: 0;
            text-align: left;
            white-space: nowrap;
            font-size: .86rem;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
            background: rgba(22, 132, 91, .42);
            border: 1px solid rgba(123, 218, 172, .25);
            text-align: left;
            white-space: nowrap;
            font-size: .86rem;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
            background: rgba(255,255,255,.09);
            border-color: transparent;
            color: #ffffff;
        }
        [data-testid="stSidebar"] .stButton button {
            min-height: 2.2rem;
            justify-content: flex-start;
            padding: .25rem .55rem;
            border-radius: 8px;
            box-shadow: none;
            color: #f3faf5 !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
            color: #ffffff !important;
        }
        .av-nav-heading {
            color: #9dc2ab !important;
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .13em;
            margin: .75rem 0 .2rem;
        }
        .av-sidebar-footer {
            display: flex;
            justify-content: space-between;
            color: #c5d9cb;
            font-size: .78rem;
            padding: .25rem .15rem;
        }
        h1, h2, h3 { color: var(--forest); letter-spacing: -.025em; }
        [data-testid="stMetric"] {
            background: #fff;
            border: 1px solid #e6ebe4;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 4px 16px rgba(24,56,39,.045);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--forest); }
        [data-testid="stTabs"] button {
            color: var(--ink) !important;
        }
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--emerald);
            border-bottom-color: var(--emerald);
        }

        [data-testid="stMain"] [data-testid="stBaseButton-primary"],
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button {
            background: #16845B !important;
            color: #FFFFFF !important;
            border: 1px solid #16845B !important;
            border-radius: 9px;
            font-weight: 650;
            min-height: 2.55rem;
            padding: .45rem .9rem;
            opacity: 1;
        }
        [data-testid="stMain"] [data-testid="stBaseButton-primary"]:hover,
        [data-testid="stMain"] [data-testid="stFormSubmitButton"] button:hover {
            background: #116B49 !important;
            border-color: #116B49 !important;
            color: #FFFFFF !important;
        }
        [data-testid="stMain"] [data-testid="stBaseButton-secondary"],
        [data-testid="stMain"] [data-testid="stDownloadButton"] button {
            background: #FFFFFF !important;
            color: #102B24 !important;
            border: 1px solid #B7D4C2 !important;
            border-radius: 9px;
            font-weight: 600;
            min-height: 2.55rem;
            padding: .45rem .9rem;
            opacity: 1;
        }
        [data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover,
        [data-testid="stMain"] [data-testid="stDownloadButton"] button:hover {
            background: #E8F4EC !important;
            border-color: #16845B !important;
            color: #102B24 !important;
        }
        [data-testid="stMain"] [data-testid="stBaseButton-secondary"]:active,
        [data-testid="stMain"] [data-testid="stDownloadButton"] button:active {
            background: #DCEBE1 !important;
            border-color: #16845B !important;
            color: #102B24 !important;
        }
        [data-testid="stMain"] button:focus-visible {
            outline: 3px solid rgba(22,132,91,.35) !important;
            outline-offset: 2px;
        }
        [data-testid="stMain"] button:active {
            background-color: #116B49;
        }
        [data-testid="stMain"] button:disabled,
        [data-testid="stMain"] button[disabled] {
            background: #E9EFEB !important;
            border-color: #D2DDD6 !important;
            color: #607068 !important;
            opacity: 1;
            cursor: not-allowed;
        }

        [data-testid="stMain"] input,
        [data-testid="stMain"] textarea {
            background: #FFFFFF !important;
            color: #102B24 !important;
            border-color: #B7C9BE !important;
            caret-color: #16845B;
        }
        [data-testid="stMain"] input::placeholder,
        [data-testid="stMain"] textarea::placeholder {
            color: #68766D !important;
            opacity: 1;
        }
        [data-testid="stMain"] [data-baseweb="select"] > div {
            background: #FFFFFF !important;
            color: #102B24 !important;
            border-color: #B7C9BE !important;
        }
        [data-testid="stMain"] [role="option"] {
            background: #FFFFFF;
            color: #102B24;
        }
        [data-testid="stMain"] [role="option"]:hover,
        [data-testid="stMain"] [role="option"][aria-selected="true"] {
            background: #E8F4EC !important;
            color: #102B24 !important;
        }
        [data-testid="stMain"] [data-testid="stExpander"] {
            background: #FFFFFF;
            border: 1px solid #DDE7E0;
            border-radius: 10px;
        }
        [data-testid="stMain"] [data-testid="stExpander"] summary,
        [data-testid="stMain"] [data-testid="stExpander"] summary * {
            color: #102B24 !important;
        }
        [data-testid="stMain"] [data-testid="stFileUploader"] section {
            background: #FFFFFF;
            border-color: #B7C9BE;
        }
        [data-testid="stMain"] [data-testid="stFileUploader"] * {
            color: #24372D;
        }
        [data-testid="stMain"] input[type="checkbox"],
        [data-testid="stMain"] input[type="radio"] {
            accent-color: #16845B;
        }
        [data-testid="stMain"] [data-testid="stCheckbox"] label,
        [data-testid="stMain"] [data-testid="stRadio"] label,
        [data-testid="stMain"] [data-testid="stSlider"] label {
            color: #102B24;
        }
        [data-testid="stMain"] [data-testid="stAlert"] * {
            color: #24372d;
        }
        [data-testid="stMain"] [data-testid="stDataFrame"] {
            background: #FFFFFF;
            color: #102B24;
        }
        .av-eyebrow {
            color: var(--emerald); font-size: .78rem; font-weight: 700;
            letter-spacing: .12em; text-transform: uppercase;
        }
        .av-card {
            background: #fff; border: 1px solid #e6ebe4; border-radius: 14px;
            padding: 1.1rem 1.2rem; height: 100%;
            color: #102B24;
        }
        .av-card strong, .av-card h3 { color: #102B24; }
        .av-muted { color: #52675c; }
        .av-card h3 { margin-top: 0; }
        @media (max-width: 768px) {
            [data-testid="stSidebar"] { min-width: 0; max-width: 100%; }
            .av-card { padding: .8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_intro(eyebrow, title, description):
    st.markdown(f'<p class="av-eyebrow">{eyebrow}</p>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)
