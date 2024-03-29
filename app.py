"Vergleich PKV vs. GKV, `streamlit run app.py`"

from typing import List, OrderedDict, Tuple
import streamlit as st

import plotly.graph_objects as go
import numpy as np
import pandas as pd

__author__ = 'Alexander Stolz'
__email__ = 'amstolz@gmail.com'
__updated__ = '(ast) 2024-01-23 @ 02:24'


st.set_page_config(
    page_title="PKV vs GKV",
    layout="wide",
    initial_sidebar_state="expanded",
)

footer = """<style>
.footer {
  position: fixed;
  bottom: 10px;
  color: #888;
}

a:link , a:visited {
  color: #999;
  text-decoration: none;
}
</style>

<div class="footer">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20">
        <path color="#999" d="M16 13V5H6v8a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2zM5 3h15a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2h-2v3a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4V4a1 1 0 0 1 1-1zm13 2v3h2V5h-2zM2 19h18v1H2v-2z" style="fill:#7b7b7b;fill-opacity:0.4"/>
    </svg>
    &gt;_ Alexander Stolz
    <!-- <a href="mailto:amstolz@gmail.com">Kontakt</a> -->
</div>
"""

with st.sidebar:
    st.title("Einstellungen")

    # Alter
    cols_alter = st.columns(2)
    alter_start = cols_alter[0].number_input(
        'Alter', min_value=18, max_value=100, value=39
    )
    rente_ab = cols_alter[1].number_input(
        'Rentenalter', min_value=63, max_value=67, value=67
    )

    # Beiträge
    cols_beitrag = st.columns(2)
    pkv_beitrag = cols_beitrag[0].number_input(
        'PKV-Betrag bei Eintritt', min_value=150, max_value=2500, value=800
    )
    gkv_beitrag = cols_beitrag[1].number_input(
        'GKV-Betrag bei Eintritt', min_value=150, max_value=2500, value=1070
    )
    pkv_eigenanteil = st.number_input(
        'Eigenanteil PKV (€ / Jahr)', min_value=0, max_value=5000, value=480
    )

    cols_rente = st.columns(2)
    rente = cols_rente[0].number_input(
        'Regelaltersrente', min_value=0, max_value=10000, value=2000
    )
    rentenanpassung = cols_rente[1].number_input(
        'Rentenanpassung (%)',
        min_value=0.0,
        max_value=25.0,
        value=1.5,
        step=0.1,
        format="%.1f",
    )

    with st.expander("Kinder"):
        kinder_anzahl_start = st.number_input(
            'Anzahl Kinder aktuell',
            min_value=0,
            max_value=10,
            value=0,
        )
        kinder_anzahl = st.number_input(
            'Geplante Anzahl Kinder (PKV versichert)',
            min_value=0,
            max_value=10,
            value=0,
        )
        kinder_ab = st.number_input(
            'Kinder ab Alter',
            min_value=alter_start,
            max_value=67,
            value=max(40, alter_start + 2),
        )
        kinder_versichert_bis = st.number_input(
            'Kinder in PKV bis Alter', min_value=18, max_value=30, value=25
        )
        kinder_beitrag = st.number_input(
            'PKV-Beitrag / Kind', min_value=0, max_value=1000, value=150
        )

    with st.expander("Beitragsanpassung"):
        # Anpassung
        anpassung_pkv = st.number_input(
            'PKV Anpassung (%)',
            min_value=0.0,
            max_value=25.0,
            value=3.0,
            step=0.1,
            format="%.1f",
        )
        anpassung_pkv_60 = st.number_input(
            'PKV Anpassung ab 60 (%)',
            min_value=0.0,
            max_value=25.0,
            value=1.5,
            step=0.1,
            format="%.1f",
        )
        anpassung_pkv_80 = st.number_input(
            'PKV Anpassung ab 80 (%)',
            min_value=0.0,
            max_value=25.0,
            value=0.0,
            step=0.1,
            format="%.1f",
        )
        anpassung_gkv = st.number_input(
            'GKV JAEG Anpassung (%)',
            min_value=0.0,
            max_value=25.0,
            value=3.0,
            step=0.1,
            format="%.1f",
        )
        von_anpassung_ausgeschlossen_pkv = st.number_input(
            'Von PKV-Anpassung unbeteiliger Betrag',
            min_value=0,
            max_value=pkv_beitrag,
            value=28 + 80,  # 63 + 50 + 3 + 22,
        )

        st.write(
            'Unbeteiligter Betrag: Dieser Anteil ist fix, wird also nicht '
            'jährlich angepasst. Beispiele sind Krankentagegeld, Altersrückstellung '
            'oder eine ergänzende Pflegeversicherung (die normale wird angepasst).'
        )

    with st.expander("Rückstellungen / Rückzahlungen"):
        entlastung_pkv = st.number_input(
            'PKV-Entlastungen ab Rente (€)',
            min_value=0,
            max_value=1000,
            value=230 + 28 + 80,
        )
        st.write(
            'Das ist bspw. der Wegfall des Krankentagegeldes oder die '
            'Beitragsermäßigung im Alter '
            '(Altersentlastung + Wegfall des Rückstellungsbeitrags).'
        )
        rueckzahlung_leistungsfrei_prz = st.number_input(
            'Rückzahlung bei Leistungsfreiheit (%)',
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
        )
        rueckzahlung_leistungsfrei_mb = st.number_input(
            'Rückzahlung bei Leistungsfreiheit (Monatsbeiträge)',
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
        )
        rueckzahlung_leistungsfrei_abs = st.number_input(
            'Rückzahlung bei Leistungsfreiheit (€ / Jahr)',
            min_value=0.0,
            max_value=5000.0,
            value=900.0,
            step=1.0,
        )
        anteil_leistungsfrei = st.number_input(
            'Anteil der leistungsfreien Jahre (%)',
            min_value=0,
            max_value=100,
            value=25,
        )
        st.write(
            'Für eine mittlere Rückzahlung (bspw. bei Staffelung) den Anteil auf 100 '
            'und die Rückzahlung auf das geschätzte Mittel setzen.'
        )

    # Sparen
    with st.expander("Anlage der Ersparnis"):
        cols_sparen = st.columns(2)
        sparquote = cols_sparen[0].number_input(
            'Sparquote (%)', min_value=0, max_value=100, value=100, step=5
        )
        sparrendite = cols_sparen[1].number_input(
            'Verzinsung (%)', min_value=0.0, max_value=50.0, value=3.0, step=0.5
        )
        st.write('Sparquote: Welcher Teil der Ersparnis soll zurückgelegt werden?')

    # Steuern
    with st.expander("Steuern"):
        steuer_beruecksichtigen = st.checkbox(
            'Steuererstattung für das Ansparen berücksichtigen', value=True
        )
        steuer_maximal_absetzbar = st.number_input(
            'Maximal absetzbarer Betrag (€ / Jahr)',
            min_value=0,
            max_value=10000,
            value=1900,
        )
        steuersatz_bis_rente = st.number_input(
            'Steuersatz bis Rente (%)', min_value=0, max_value=100, value=42
        )
        steuersatz_ab_rente = st.number_input(
            'Steuersatz ab Rente (%)', min_value=0, max_value=45, value=35
        )
        absetzbar_pkv = st.number_input(
            'Absetzbarer Anteil PKV (%)', min_value=0, max_value=100, value=75
        )
        st.write(
            'Der GKV-Beitrag ist zu 100 % absetzbar. Von dem PKV-Beitrag sind '
            'nur die Leistungen absetzbar, die auch die GKV abdeckt '
            '(nicht absetzbar sind bspw. Chefarztbehandlung und Einzelzimmer).'
        )

    with st.expander("Sonstiges"):
        faktor_rueckstellung = st.number_input(
            'Gesetzliche Rückstellung vom Beitrag (%)',
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
        )
        st.write(
            'Gesetzliche Rückstellung: Dieser Anteil wird vor der Rente für die '
            'Altersentlastung angespart und entfällt ab der Rente. Ist im PKV-Beitrag '
            'oben bereits enthalten.'
        )
        berechnung_bis = st.number_input(
            'Berechnung bis Alter', min_value=18, max_value=150, value=100
        )
        # Beitragssatz GKV seit 01.01.2021: 14,6 %
        beitragssatz_gkv = st.number_input(
            'Beitragssatz GKV (%)', min_value=0.0, max_value=100.0, value=14.6
        )
        # # Allgemeiner Beitragssatz Pflegeversicherung seit 01.07.2023: 3,4 %
        # beitragssatz_pflegeversicherung = st.number_input(
        #     'Beitragssatz Pflegeversicherung (%)',
        #     min_value=0.0,
        #     max_value=100.0,
        #     value=3.4,
        # )

    st.markdown(footer, unsafe_allow_html=True)


def get_kinder(alter: int, nur_versichert: bool = False) -> int:
    if nur_versichert:
        if kinder_ab <= alter <= kinder_ab + kinder_versichert_bis:
            return kinder_anzahl + kinder_anzahl_start
        return kinder_anzahl_start
    if kinder_ab <= alter:
        return kinder_anzahl + kinder_anzahl_start
    return kinder_anzahl_start


def get_rente(alter: int) -> float:
    faktor_rentenanpassung = (1 + rentenanpassung / 100) ** (alter - alter_start)
    # für jeden Monat frühzeitig Rente gibt es 0,3% weniger Rente (dauerhaft)
    faktor_fruehzeitig = 1 - 0.003 * (67 - rente_ab) * 12
    if alter >= rente_ab:
        return rente * faktor_rentenanpassung * faktor_fruehzeitig
    else:
        return 0


def get_gkv_beitrag(x_alter: np.ndarray) -> Tuple[np.ndarray, set]:
    hinweise = []
    y_beitrag = np.zeros_like(x_alter)

    # {Kinder: (Gesamt, Arbeitnehmer)}
    pflegeversicherung_satz_by_kinder = {
        0: (3.4 + 0.6, 2.3),
        1: (3.4, 1.7),
        2: (3.15, 1.45),
        3: (2.9, 1.2),
        4: (2.65, 0.95),
        5: (2.4, 0.7),
    }

    satz_pflege_start = pflegeversicherung_satz_by_kinder[
        min(get_kinder(alter_start), 5)
    ]

    for i, a in enumerate(x_alter):
        kinder = min(get_kinder(a), 5)
        satz_pflege = pflegeversicherung_satz_by_kinder.get(kinder)
        if i == 0:
            hinweise.append(
                f'GKV-Beitrag: {beitragssatz_gkv / 2:.1f} % vom Brutto zzgl. '
                f'{satz_pflege[1]:.1f} % Pflegeversicherung ({kinder} Kinder).'
            )
        if kinder > 0 and a == kinder_ab:
            hinweise.append(
                f'GKV-Beitrag: {beitragssatz_gkv / 2:.1f} % vom Brutto zzgl. '
                f'{satz_pflege[1]:.1f} % Pflegeversicherung ({kinder} Kinder).'
            )
        if a < rente_ab:
            # Beitrag komplett
            beitrag = gkv_beitrag * (1 + anpassung_gkv / 100) ** i
            # Beitrag komplett ohne Pflege
            beitrag_ohne_pflege = beitrag - beitrag * (
                1 - beitragssatz_gkv / (beitragssatz_gkv + satz_pflege_start[0])
            )
            # Beitrag Pflege Arbeitnehmeranteil
            beitrag_pflege_an = beitrag * (
                1 - beitragssatz_gkv / (beitragssatz_gkv + satz_pflege[1])
            )
            beitrag = beitrag_ohne_pflege / 2 + beitrag_pflege_an

        elif a >= rente_ab:
            beitrag = get_rente(a) * (beitragssatz_gkv / 100 / 2 + satz_pflege[0] / 100)
        y_beitrag[i] = beitrag
        if a == rente_ab:
            hinweise.append(
                f'Bruttorente bei Renteneintritt: {get_rente(a):.0f} €. '
                'Ab diesem Zeitpunkt beträgt der '
                f'GKV-Beitrag {beitragssatz_gkv / 2:.1f} % von der Rente zzgl. '
                f'{satz_pflege[0]:.1f} % Pflegeversicherung ({kinder} Kinder).'
            )
    return y_beitrag, hinweise


def get_pkv_beitrag(x_alter: np.ndarray) -> Tuple[np.ndarray, set]:
    hinweise = []
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
                hinweise.append(
                    f'Kinderbeitrag {kosten_kinder} € (AG übernimmt die Hälfte)'
                )
            else:
                hinweise.append(
                    f'Kinderbeitrag in Rente {kosten_kinder} € (AG übernimmt nicht)'
                )
            kosten += kosten_kinder
        if a < 60:
            beitrag = (pkv_dynamisch * (1 + anpassung_pkv / 100) ** i + pkv_fix) / 2
            hinweise.append(
                f'Dynamischer Beitragsanteil: {pkv_dynamisch:.0f} € zu jährlich '
                f'{anpassung_pkv:.1f} % bis 60 Jahre. '
                f'Fixer Anteil: {pkv_fix:.0f} €. AG übernimmt die Hälfte.'
            )
        elif 60 <= a < 67:
            beitrag = (
                pkv_dynamisch
                * (1 + anpassung_pkv / 100) ** (60 - alter_start)
                * (1 + anpassung_pkv_60 / 100) ** (a - 60)
                + pkv_fix
            )
            hinweise.append(
                f'Dynamischer Beitragsanteil: {pkv_dynamisch:.0f} € zu jährlich '
                f'{anpassung_pkv_60:.1f} % zwischen 60 - 67 Jahre. '
                f'Fixer Anteil: {pkv_fix:.0f} €.'
            )
            if a < rente_ab:
                # AG übernimmt die Hälfte
                beitrag /= 2
                hinweise[-1] += ' AG übernimmt die Hälfte.'
        elif 67 <= a < 80:
            beitrag = (
                pkv_dynamisch
                * (1 + anpassung_pkv / 100) ** (60 - alter_start)
                * (1 + anpassung_pkv_60 / 100) ** (a - 60)
                + pkv_fix
            )
            beitrag *= 1 - faktor_rueckstellung / 100
            kosten -= entlastung_pkv + get_rente(a) * beitragssatz_gkv / 100 / 2
            hinweise.append(
                f'Dynamischer Beitragsanteil: {pkv_dynamisch:.0f} € zu jährlich '
                f'{anpassung_pkv_60:.1f} % zwischen 67 - 80. '
                f'Fixer Anteil: {pkv_fix:.0f} €. '
                f'Reduzierung um {faktor_rueckstellung:.1f} % '
                '(Gesetzliche Beitragsrückstellung entfällt), '
                f'{entlastung_pkv:.0f} € (Entlastung PKV) und 7.3 % von der Rente.'
            )
        elif a >= 80:
            beitrag = (
                pkv_dynamisch
                * (1 + anpassung_pkv / 100) ** (60 - alter_start)
                * (1 + anpassung_pkv_60 / 100) ** (80 - 60)
                * (1 + anpassung_pkv_80 / 100) ** (a - 80)
                + pkv_fix
            )
            beitrag *= 1 - faktor_rueckstellung / 100
            kosten -= entlastung_pkv + get_rente(a) * beitragssatz_gkv / 100 / 2
            hinweise.append(
                f'Dynamischer Beitragsanteil: {pkv_dynamisch:.0f} € zu jährlich '
                f'{anpassung_pkv_80:.1f} % ab 80. '
                f'Fixer Anteil: {pkv_fix:.0f} €. '
                f'Reduzierung um {faktor_rueckstellung:.1f} % '
                '(Gesetzliche Beitragsrückstellung entfällt), '
                f'{entlastung_pkv:.0f} € (Entlastung PKV) und 7.3 % von der Rente.'
            )

        # Rückzahlung bei Leistungsfreiheit
        rel_rueckzahlung_leistungsfrei = (
            (
                (rueckzahlung_leistungsfrei_prz / 100)
                + (rueckzahlung_leistungsfrei_mb / 12)
            )
            * anteil_leistungsfrei
            / 100
        )
        abs_rueckzahlung_leistungsfrei = (
            rueckzahlung_leistungsfrei_abs * anteil_leistungsfrei / 100 / 12
        )
        if steuer_beruecksichtigen:
            steuersatz = steuersatz_bis_rente if a < rente_ab else steuersatz_ab_rente
            rueckzahlung = (
                beitrag * rel_rueckzahlung_leistungsfrei
                + abs_rueckzahlung_leistungsfrei
            ) * (1 - steuersatz / 100)
            kosten -= rueckzahlung
        else:
            kosten -= (
                beitrag * rel_rueckzahlung_leistungsfrei
                + abs_rueckzahlung_leistungsfrei
            )

        # Eigenanteil durchschnittlich
        kosten += pkv_eigenanteil * (1 - anteil_leistungsfrei / 100) / 12
        y_beitrag[i] = beitrag + kosten

    if steuer_beruecksichtigen:
        hinweise.append(
            f'Mittlere Rückzahlung bei Leistungsfreiheit: '
            f'dynamisch: {rel_rueckzahlung_leistungsfrei * 100: .1f} %, '
            f'fix: {abs_rueckzahlung_leistungsfrei:.0f} €. '
            f'Abzüglich Steuern.'
        )
    else:
        hinweise.append(
            f'Mittlere Rückzahlung bei Leistungsfreiheit: '
            f'dynamisch: {rel_rueckzahlung_leistungsfrei * 100: .1f} %, '
            f'fix: {abs_rueckzahlung_leistungsfrei:.0f} €. '
            'Steuern nicht berücksichtigt.'
        )
    hinweise.append(
        f'Eigenanteil: {pkv_eigenanteil:.0f} € (durchschnittlich, '
        f'bei {anteil_leistungsfrei:.0f} % Leistungsfreiheit).'
    )
    hinweise = OrderedDict.fromkeys(hinweise)
    return y_beitrag, hinweise


def get_sparkonto(alter, beitraege_pkv, beitraege_gkv) -> List[float]:
    gespart = [0]
    for i in range(len(beitraege_pkv)):
        gespart.append(gespart[-1])
        gespart[-1] *= 1 + sparrendite / 100
        gezahlt_pkv = beitraege_pkv[i]
        gezahlt_gkv = beitraege_gkv[i]
        if steuer_beruecksichtigen:
            steuersatz = (
                steuersatz_bis_rente if alter[i] < rente_ab else steuersatz_ab_rente
            )
            gezahlt_pkv -= (
                min(
                    beitraege_pkv[i] * absetzbar_pkv / 100,
                    steuer_maximal_absetzbar / 12.0,
                )
                * steuersatz
                / 100
            )
            gezahlt_gkv -= (
                min(
                    beitraege_gkv[i],
                    steuer_maximal_absetzbar / 12.0,
                )
                * steuersatz
                / 100
            )
        if gezahlt_pkv < gezahlt_gkv:
            gespart[-1] += (gezahlt_gkv - gezahlt_pkv) * 12 * sparquote / 100
        else:
            gespart[-1] -= (gezahlt_pkv - gezahlt_gkv) * 12
    return gespart


st.title("Simulierter Verlauf")

x = np.arange(alter_start, berechnung_bis + 1, 1)
y_gkv, hinweise_gkv = get_gkv_beitrag(x)
y_pkv, hinweise_pkv = get_pkv_beitrag(x)

x_rente = np.arange(rente_ab, berechnung_bis + 1, 1)
y_rente = [get_rente(_) * (1 - steuersatz_ab_rente / 100) for _ in x_rente]
y_sparkonto = get_sparkonto(x, y_pkv, y_gkv)

# Anzeige des Plots
st.subheader('Beiträge')
st.write(
    'Bis zur Rente bezahlt der Arbeitgeber die Hälfte der Beiträge (auch für Kinder). '
    'Über die Legende kann die Rente eingeblendet werden. '
    f'Sie ist mit {steuersatz_ab_rente:.0f} % (Einstellungen/Sonstiges) versteuert.'
)

fig = go.Figure(
    layout=dict(
        xaxis_title="Alter",
        yaxis_title="Netto-Beträge (€)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
)
fig.add_scatter(x=x, y=y_gkv, mode='lines', name='GKV-Beitrag')
fig.add_scatter(x=x, y=y_pkv, mode='lines', name='PKV-Beitrag')
fig.add_scatter(
    x=x_rente,
    y=y_rente,
    mode='lines',
    name=f'Netto-Rente (Steuersatz {steuersatz_ab_rente} %)',
    visible='legendonly',
)
fig.update_layout(
    width=1000,
    height=600,
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)
st.plotly_chart(fig, use_container_width=True)


# Anzeige des Ersparten
st.subheader(
    f'Verlauf des Sparkontos bei PKV-Vertrag ({sparrendite:.1f} % Verzinsung)',
)
st.write(
    'Unter der Annahme, dass man die Differenz zwischen GKV- und PKV-Beitrag anlegt, '
    'falls der PKV-Beitrag niedriger ist und die Differenz aus dem Sparkonto '
    'bezahlt, falls der PKV-Beitrag höher ist.'
)

fig = go.Figure(
    layout=dict(
        xaxis_title="Alter",
        yaxis_title="Sparkonto (€)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
)
fig.add_scatter(x=x, y=y_sparkonto, mode='lines', name='Sparkonto')
fig.add_shape(
    type="line",
    x0=alter_start,
    y0=0,
    x1=berechnung_bis,
    y1=0,
    line=dict(color="#553333", width=1),
)
fig.update_layout(
    width=1000,
    height=500,
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)
st.plotly_chart(fig, use_container_width=True)

# Anzeigen der Hinweise
st.subheader('Hinweise zur Berechnung')
st.table(
    pd.DataFrame(
        data=list(hinweise_gkv),
        columns=["Hinweise zur GKV"],
    )
)
st.table(
    pd.DataFrame(
        data=list(hinweise_pkv),
        columns=["Hinweise zur PKV"],
    )
)
