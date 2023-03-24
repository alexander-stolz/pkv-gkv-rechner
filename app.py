import streamlit as st
from bokeh.plotting import figure
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="PKV vs GKV",
    # page_icon=':chart_with_upwards_trend:',
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("Einstellungen")

    # Alter
    cols_alter = st.columns(2)
    alter_start = cols_alter[0].number_input(
        'Alter', min_value=18, max_value=100, value=35
    )
    rente_ab = cols_alter[1].number_input(
        'Rentenalter', min_value=18, max_value=100, value=67
    )

    # Beiträge
    cols_beitrag = st.columns(2)
    pkv_beitrag = cols_beitrag[0].number_input(
        'PKV-Betrag bei Eintritt', min_value=150, max_value=2500, value=680
    )
    gkv_beitrag = cols_beitrag[1].number_input(
        'GKV-Betrag bei Eintritt', min_value=150, max_value=2500, value=990
    )
    rente = st.number_input(
        'Geschätzte Rente bei Renteneintritt', min_value=0, max_value=10000, value=2000
    )

    with st.expander("Kinder"):
        kinder_anzahl = st.number_input(
            'Geplante Anzahl Kinder (PKV versichert)',
            min_value=0,
            max_value=10,
            value=0,
        )
        kinder_ab = st.number_input(
            'Kinder ab Alter', min_value=alter_start, max_value=67, value=40
        )
        kinder_versichert_bis = st.number_input(
            'Kinder in PKV bis Alter', min_value=18, max_value=30, value=25
        )
        kinder_beitrag = st.number_input(
            'PKV-Beitrag / Kind', min_value=0, max_value=1000, value=150
        )

    with st.expander("Beitragsanpassung"):
        # Anpassung
        cols_anpassung = st.columns(2)
        anpassung_pkv = cols_anpassung[0].number_input(
            'PKV Anpassung (%)',
            min_value=0.0,
            max_value=25.0,
            value=3.0,
            step=0.1,
            format="%.1f",
        )
        anpassung_gkv = cols_anpassung[1].number_input(
            'GKV Anpassung (%)',
            min_value=0.0,
            max_value=25.0,
            value=3.0,
            step=0.1,
            format="%.1f",
        )
        cols_anpassung_60 = st.columns(2)
        anpassung_pkv_60 = cols_anpassung_60[0].number_input(
            'PKV Anpassung ab 60 (%)',
            min_value=0.0,
            max_value=25.0,
            value=1.5,
            step=0.1,
            format="%.1f",
        )
        anpassung_pkv_80 = cols_anpassung_60[0].number_input(
            'PKV Anpassung ab 80 (%)',
            min_value=0.0,
            max_value=25.0,
            value=0.0,
            step=0.1,
            format="%.1f",
        )
        anpassung_gkv_60 = cols_anpassung_60[1].number_input(
            'GKV Anpassung ab 60 (%)',
            min_value=0.0,
            max_value=25.0,
            value=3.0,
            step=0.1,
            format="%.1f",
        )

        von_anpassung_ausgeschlossen_pkv = st.number_input(
            'Von Anpassung unbeteiliger PKV-Betrag (€)',
            min_value=0,
            max_value=pkv_beitrag,
            value=63 + 50 + 3 + 22,
        )

    with st.expander("Rückstellungen / Rückzahlungen"):
        faktor_rueckstellung = st.number_input(
            'Gesetzliche Rückstellung vom Beitrag (%)',
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
        )
        entlastung_pkv = st.number_input(
            'Altersentlastung ab Rente (€)', min_value=0, max_value=1000, value=235 + 31
        )
        rueckzahlung_beitragsfrei = st.number_input(
            'Rückzahlung bei Leistungsfreiheit (%)',
            min_value=0.0,
            max_value=100.0,
            value=2.5 / 12 * 100,
            step=1.0,
        )
        anteil_beitragsfrei = st.number_input(
            'Anteil der beitragsfreien Jahre (%)',
            min_value=0,
            max_value=100,
            value=25,
        )


def get_gkv_beitrag(x_alter: np.ndarray) -> np.ndarray:
    y_beitrag = np.zeros_like(x_alter)
    for i, a in enumerate(x_alter):
        if a >= 67:
            beitrag = rente * (0.146 / 2 + 0.0305) * 1.03 ** (a - rente_ab)
        else:
            beitrag = gkv_beitrag * (1 + anpassung_gkv / 100) ** i
        y_beitrag[i] = beitrag
    return y_beitrag


def get_pkv_beitrag(x_alter: np.ndarray) -> np.ndarray:
    y_beitrag = np.zeros_like(x_alter)
    pkv_dynamisch = pkv_beitrag - von_anpassung_ausgeschlossen_pkv
    pkv_fix = von_anpassung_ausgeschlossen_pkv
    for i, a in enumerate(x_alter):
        kosten = 0
        if kinder_anzahl > 0 and kinder_ab <= a <= kinder_ab + kinder_versichert_bis:
            kosten_kinder = kinder_anzahl * kinder_beitrag
            if a < rente_ab:
                # AG übernimmt die Hälfte
                kosten_kinder /= 2
            kosten += kosten_kinder
        if a < 60:
            beitrag = pkv_dynamisch * (1 + anpassung_pkv / 100) ** i + pkv_fix
        elif 60 <= a < rente_ab:
            beitrag = (
                pkv_dynamisch
                * (1 + anpassung_pkv / 100) ** (60 - alter_start)
                * (1 + anpassung_pkv_60 / 100) ** (a - 60)
                + pkv_fix
            )
        elif rente_ab <= a < 80:
            beitrag = (
                pkv_dynamisch
                * (1 + anpassung_pkv / 100) ** (60 - alter_start)
                * (1 + anpassung_pkv_60 / 100) ** (a - 60)
                + pkv_fix
            )
            beitrag *= 1 - faktor_rueckstellung / 100
            kosten -= entlastung_pkv + rente * 1.03 ** (a - rente_ab) * 0.146 / 2
        elif a >= 80:
            beitrag = (
                pkv_dynamisch
                * (1 + anpassung_pkv / 100) ** (60 - alter_start)
                * (1 + anpassung_pkv_60 / 100) ** (80 - 60)
                * (1 + anpassung_pkv_80 / 100) ** (a - 80)
                + pkv_fix
            )
            beitrag *= 1 - faktor_rueckstellung / 100
            kosten -= entlastung_pkv + rente * 1.03 ** (a - rente_ab) * 0.146 / 2

        kosten -= beitrag * rueckzahlung_beitragsfrei / 100 * anteil_beitragsfrei / 100
        y_beitrag[i] = beitrag + kosten
    return y_beitrag


st.title("Simulierter Beitragsverlauf")

x = np.arange(alter_start, 100, 1)
y_gkv = get_gkv_beitrag(x)
y_pkv = get_pkv_beitrag(x)

p = figure(
    title="Beitragsverlauf",
    x_axis_label='Alter',
    y_axis_label='Beitrag (€)',
    plot_width=1000,
    plot_height=600,
    sizing_mode='stretch_width',
    toolbar_location=None,
    tools="",
)
p.line(x, y_gkv, line_width=2, color='#1f77b4', legend_label="GKV")
p.line(x, y_pkv, line_width=2, color="green", legend_label="PKV")
st.bokeh_chart(p)

st.subheader('Zusammenfassung')

df = pd.DataFrame(
    data=[['PKV', y_pkv.sum()], ['GKV', y_gkv.sum()]],
    columns=["Versicherung", "Summe aller Beiträge"],
).round(2)

# Anzeigen des DataFrames in Streamlit
st.table(df.style.hide_index())
