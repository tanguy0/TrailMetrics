"""Central translation table for TrailMetrics.

One source of truth for every user-facing string, in English (``en``) and French
(``fr``). Pure Python with no framework dependency, so the domain, the plot
registry and the API all read from it.

Usage:
    from src.translations import translate
    translate("plot.metric_trend.label", "fr")

Everything user-facing is translated **server-side**: plot labels and parameter
schemas are rendered into the ``/registry`` payload, and chart IR carries finished
text rather than keys. The web app therefore has no translation table of its own —
it displays what it is given, and adding a language here covers the whole product.

Strings may contain ``{placeholder}`` fields — format them at the call site,
e.g. ``translate("panel.dropped_streamless", lang).format(count=12)``.
"""

LANGUAGES = {"fr": "Français", "en": "English"}
DEFAULT_LANG = "en"

# key -> {"en": ..., "fr": ...}
TRANSLATIONS = {
    # --- Built-in page names -------------------------------------------------
    "page.gap.title": {
        "en": "Personalized GAP Simulator",
        "fr": "Simulateur GAP personnalisé",
    },
    "page.races.title": {"en": "Race Comparator", "fr": "Comparateur de courses"},

    # --- GAP model labels, parameters and captions ---------------------------
    "gap.intro": {
        "en": """
    Build personalized GAP (Gradient Adjusted Pace) curves from your own activities
    and compare them against reference curves. Set the panel's data source to
    several time windows to fit one curve per period.
    """,
        "fr": """
    Construisez des courbes GAP (allure ajustée à la pente) personnalisées à partir
    de vos propres activités et comparez-les à des courbes de référence. Utilisez
    plusieurs fenêtres temporelles dans la source du panneau pour obtenir une
    courbe par période.
    """,
    },
    "gap.models.caption": {
        "en": "Pick at least one model to plot.",
        "fr": "Choisissez au moins un modèle à afficher.",
    },
    "gap.models.efficiency": {"en": "Efficiency model", "fr": "Modèle d'efficacité"},
    "gap.models.auto": {"en": "Auto-Learning model", "fr": "Modèle auto-apprenant"},
    "gap.refs.caption": {
        "en": "Optional overlays — uncheck both to hide them.",
        "fr": "Superpositions optionnelles — décochez les deux pour les masquer.",
    },
    "gap.refs.balanced": {"en": "Balanced runner", "fr": "Coureur équilibré"},
    "gap.refs.kilian": {"en": "Kilian curve", "fr": "Courbe Kilian"},
    "gap.display.show_std": {
        "en": "Show standard deviation bands",
        "fr": "Afficher les bandes d'écart-type",
    },
    "gap.display.show_std_help": {
        "en": "Shade ±1 std around each curve. Turn off for a cleaner overlay.",
        "fr": "Ombre ±1 écart-type autour de chaque courbe. Décochez pour une "
        "superposition plus épurée.",
    },
    "gap.params.split_min_time": {
        "en": "Split min time (seconds)",
        "fr": "Durée min. de segment (secondes)",
    },
    "gap.params.hr_tol": {"en": "HR tolerance (bpm)", "fr": "Tolérance FC (bpm)"},
    "gap.params.eff_min_samples": {
        "en": "Efficiency model: min samples per bucket",
        "fr": "Modèle d'efficacité : nb min. d'échantillons par classe",
    },
    "gap.params.eff_subset_min_samples": {
        "en": "Efficiency model (per-intensity slice): min samples per bucket",
        "fr": "Modèle d'efficacité (tranche par intensité) : nb min. "
        "d'échantillons par classe",
    },
    "gap.params.eff_subset_help": {
        "en": "Lower than the full-dataset value because each HR slice has fewer "
        "points.",
        "fr": "Plus bas que la valeur globale car chaque tranche de FC contient "
        "moins de points.",
    },
    "gap.params.bin_width": {"en": "Bin width (m/km)", "fr": "Largeur de classe (m/km)"},
    "gap.intensity.low": {"en": "Low", "fr": "Basse"},
    "gap.intensity.high": {"en": "High", "fr": "Haute"},
    "gap.summary.item": {
        "en": "{label}: {n} splits",
        "fr": "{label} : {n} segments",
    },
    "gap.summary": {
        "en": "Simulation complete — {summary}.",
        "fr": "Simulation terminée — {summary}.",
    },
    "gap.nothing_to_plot": {
        "en": "Nothing to plot — select a personal model or a reference curve.",
        "fr": "Rien à tracer — sélectionnez un modèle personnel ou une courbe de "
        "référence.",
    },
    "gap.caption.main": {
        "en": "Colour = data-source group · line style = model and heart-rate band · "
        "dashed = reference curves.",
        "fr": "Couleur = groupe de la source · style de trait = modèle et zone de "
        "fréquence cardiaque · tirets = courbes de référence.",
    },
    "gap.caption.per_year": {
        "en": "One colour per calendar year, both models. A year whose curve sits "
        "lower cost you less pace per metre of climb.",
        "fr": "Une couleur par année civile, les deux modèles. Une année dont la "
        "courbe est plus basse vous a coûté moins d'allure par mètre de dénivelé.",
    },
    "gap.caption.intensity": {
        "en": "The same fit, split by heart-rate band — how the cost of climbing "
        "changes with intensity.",
        "fr": "Le même ajustement, séparé par zone de fréquence cardiaque — comment "
        "le coût de la montée évolue avec l'intensité.",
    },

    # --- Race / stream signal labels and columns -----------------------------
    "races.intro": {
        "en": """
    Compare races side by side. Pick the workouts in the panel's data source — every
    plot below then describes that same selection. Around {max} at once stays
    readable; past that, raise each plot's series limit.
    """,
        "fr": """
    Comparez des courses côte à côte. Choisissez les séances dans la source du
    panneau — tous les graphiques décrivent ensuite cette même sélection. Environ
    {max} à la fois reste lisible ; au-delà, augmentez la limite de séries de chaque
    graphique.
    """,
    },
    "races.select.subheader": {
        "en": "Pick the workouts in the data source above",
        "fr": "Choisissez les séances dans la source de données ci-dessus",
    },
    "races.col.date": {"en": "Date", "fr": "Date"},
    "races.col.sport": {"en": "Sport", "fr": "Sport"},
    "races.signal.pace": {"en": "Pace / GAP", "fr": "Allure / GAP"},
    "races.signal.altitude": {"en": "Altitude", "fr": "Altitude"},
    "races.signal.hr": {"en": "Heart rate", "fr": "Fréquence cardiaque"},
    "races.signal.power": {"en": "Power", "fr": "Puissance"},
    "races.xaxis.time": {"en": "Time", "fr": "Temps"},
    "races.xaxis.distance": {"en": "Distance", "fr": "Distance"},
    "races.xaxis.help": {
        "en": "Elapsed time, or distance covered.",
        "fr": "Temps écoulé, ou distance parcourue.",
    },
    "races.weight_needed": {
        "en": "Set your weight to enable the power metrics.",
        "fr": "Renseignez votre poids pour activer les métriques de puissance.",
    },

    # --- Plot labels: GAP curves (domain) -----------------------------------
    "plot.gap.xlabel": {"en": "Elevation Gain (m/km)", "fr": "Dénivelé (m/km)"},
    "plot.gap.ylabel": {
        "en": "Speed Adjuster (GAP/speed)",
        "fr": "Facteur de vitesse (GAP/vitesse)",
    },
    "plot.gap.title_std": {
        "en": "GAP Curve(s) and standard deviation(s)",
        "fr": "Courbe(s) GAP et écart(s)-type(s)",
    },
    "plot.gap.title": {"en": "GAP Curve(s)", "fr": "Courbe(s) GAP"},

    # --- Plot labels: race comparison (domain) ------------------------------
    "plot.races.gap_pace.y": {
        "en": "GAP pace (min/km, lower = faster)",
        "fr": "Allure GAP (min/km, plus bas = plus rapide)",
    },
    "plot.races.gap_speed.y": {
        "en": "GAP speed (km/h, higher = faster)",
        "fr": "Vitesse GAP (km/h, plus haut = plus rapide)",
    },
    "plot.races.power.y": {"en": "Power (W)", "fr": "Puissance (W)"},
    "plot.races.hr.y": {"en": "Heart rate (bpm)", "fr": "Fréquence cardiaque (bpm)"},
    "plot.races.p2hr.y": {"en": "Power / HR (W/bpm)", "fr": "Puissance / FC (W/bpm)"},
    "plot.races.x.time": {"en": "Time (min)", "fr": "Temps (min)"},
    "plot.races.x.distance": {"en": "Distance (km)", "fr": "Distance (km)"},

    # --- Long-term progress labels (records, bands, sections) ---------------
    "page.ltp.title": {
        "en": "Long-Term Progress",
        "fr": "Progression long terme",
    },
    "ltp.intro": {
        "en": "Season-over-season trends across your **entire** history (runs and "
        "trail runs). The first run crunches every activity — best efforts, "
        "gradients — then the controls below just re-shape the results instantly.",
        "fr": "Tendances saison après saison sur **tout** votre historique (course "
        "et trail). Le premier passage analyse chaque activité — meilleurs efforts, "
        "pentes — puis les options ci-dessous se contentent de réafficher les "
        "résultats instantanément.",
    },

    # Season definition

    # Section 1 — Personal records
    "ltp.section.records": {
        "en": "Evolution of personal records",
        "fr": "Évolution des records personnels",
    },
    "ltp.section.records.help": {
        "en": "One line per distance: a point each time you set a new record. For "
        "every activity long enough, the fastest contiguous segment of each "
        "distance is found; the best of those is your record. Click a distance in "
        "the legend to show or hide it.",
        "fr": "Une ligne par distance : un point à chaque nouveau record. Pour "
        "chaque activité assez longue, on cherche le segment continu le plus rapide "
        "de chaque distance ; le meilleur d'entre eux est votre record. Cliquez sur "
        "une distance dans la légende pour l'afficher ou la masquer.",
    },
    "ltp.records.col.distance": {"en": "Distance", "fr": "Distance"},
    "ltp.records.col.record": {"en": "Record", "fr": "Record"},
    "ltp.records.col.pace": {"en": "Pace", "fr": "Allure"},
    "ltp.records.col.date": {"en": "Date", "fr": "Date"},

    # Section 2 — Annual mileage

    # Section 3 — Annual elevation gain

    # Section 4 — Average gradient per season

    # Section 5 — Gradient map
    "ltp.gradient_map.help": {
        "en": "Share of moving time spent in each gradient band, per bin. Each bar "
        "sums to 100%. Click a band in the legend to show or hide it.",
        "fr": "Part du temps en mouvement passé dans chaque catégorie de pente, par "
        "période. Chaque barre totalise 100 %. Cliquez sur une catégorie dans la "
        "légende pour l'afficher ou la masquer.",
    },
    "ltp.band.steep_descent": {
        "en": "Steep descent (< -12%)", "fr": "Forte descente (< -12 %)",
    },
    "ltp.band.gentle_descent": {
        "en": "Gentle descent (-12% to -3%)", "fr": "Descente douce (-12 % à -3 %)",
    },
    "ltp.band.flat": {"en": "Flat (-3% to 3%)", "fr": "Plat (-3 % à 3 %)"},
    "ltp.band.gentle_ascent": {
        "en": "Gentle ascent (3% to 12%)", "fr": "Montée douce (3 % à 12 %)",
    },
    "ltp.band.steep_ascent": {
        "en": "Steep ascent (> 12%)", "fr": "Forte montée (> 12 %)",
    },

    # Section 6 — Power-to-HR
    "ltp.section.power_hr": {
        "en": "Evolution of power-to-HR",
        "fr": "Évolution du rapport puissance / FC",
    },
    "ltp.section.power_hr.help": {
        "en": "Weekly average of each session's mean power-to-heart-rate ratio "
        "(an aerobic-efficiency proxy — higher is better), on one continuous "
        "timeline. Each season has its own color; click a season in the legend to "
        "show or hide it.",
        "fr": "Moyenne hebdomadaire du rapport puissance / fréquence cardiaque moyen "
        "de chaque séance (un indicateur d'efficacité aérobie — plus haut est "
        "meilleur), sur une frise continue. Chaque saison a sa couleur ; cliquez sur "
        "une saison dans la légende pour l'afficher ou la masquer.",
    },

    # --- Plot labels: long-term progress (domain) ---------------------------
    "plot.ltp.records.title": {
        "en": "Personal-record evolution",
        "fr": "Évolution des records personnels",
    },
    "plot.ltp.records.y_pace": {
        "en": "Record pace (min/km, faster = higher)",
        "fr": "Allure record (min/km, plus rapide = plus haut)",
    },
    "plot.ltp.records.y_time": {
        "en": "Record time (faster = higher)",
        "fr": "Temps record (plus rapide = plus haut)",
    },
    "plot.ltp.records.hover_record": {"en": "Record", "fr": "Record"},
    "plot.ltp.records.hover_pace": {"en": "Pace", "fr": "Allure"},
    "plot.ltp.gradient_map.title": {
        "en": "Time spent per gradient band",
        "fr": "Temps passé par catégorie de pente",
    },
    "plot.ltp.gradient_map.x": {"en": "Time", "fr": "Temps"},
    "plot.ltp.gradient_map.y": {"en": "% of moving time", "fr": "% du temps en mouvement"},

    # --- Panels & pages (the composable builder) -----------------------------
    "panel.group": {"en": "Group", "fr": "Groupe"},
    "panel.all": {"en": "All", "fr": "Tout"},
    "panel.selection": {"en": "Selection", "fr": "Sélection"},
    "panel.no_activities": {
        "en": "No activity matches this panel's data source.",
        "fr": "Aucune activité ne correspond à la source de données de ce panneau.",
    },
    "panel.dropped_streamless": {
        "en": "{count} activity(ies) without per-second data were skipped — "
        "this plot needs the full traces.",
        "fr": "{count} activité(s) sans données par seconde ont été ignorées — "
        "ce graphique a besoin des traces complètes.",
    },
    "panel.dropped_cross_sport": {
        "en": "{count} activity(ies) from a different sport were excluded — a "
        "panel can't mix cycling with running.",
        "fr": "{count} activité(s) d'un autre sport ont été exclues — un panneau "
        "ne peut pas mélanger vélo et course à pied.",
    },

    # --- Plot catalogue ------------------------------------------------------
    "plotcat.general": {"en": "General", "fr": "Général"},
    "plotcat.trends": {"en": "Trends over time", "fr": "Évolutions"},
    "plotcat.records": {"en": "Records", "fr": "Records"},
    "plotcat.within": {"en": "Inside one activity", "fr": "Dans une activité"},
    "plotcat.models": {"en": "Models", "fr": "Modèles"},
    "plotcat.explore": {"en": "Exploration", "fr": "Exploration"},
    "plotcat.tables": {"en": "Tables", "fr": "Tableaux"},
    "plotcat.content": {"en": "Text & images", "fr": "Texte et images"},

    "plot.metric_trend.label": {"en": "Metric over time", "fr": "Métrique dans le temps"},
    "plot.metric_trend.description": {
        "en": "Any metric, binned by day/week/month/quarter, per period or "
        "cumulative, on the calendar or aligned to each group's start.",
        "fr": "N'importe quelle métrique, groupée par jour/semaine/mois/trimestre, "
        "par période ou cumulée, sur le calendrier ou alignée sur le début de "
        "chaque groupe.",
    },
    "plot.gradient_map.label": {"en": "Gradient map", "fr": "Carte des pentes"},
    "plot.gradient_map.description": {
        "en": "Share of moving time spent in each gradient band, over time.",
        "fr": "Part du temps en mouvement passée dans chaque catégorie de pente, "
        "dans le temps.",
    },
    "plot.gradient_map.no_stream_data": {
        "en": "No per-second data available to classify gradients.",
        "fr": "Aucune donnée par seconde disponible pour classer les pentes.",
    },
    "plot.pr_progression.label": {
        "en": "Record progression", "fr": "Progression des records",
    },
    "plot.pr_progression.description": {
        "en": "Stepped evolution of your best effort per distance.",
        "fr": "Évolution en escalier de votre meilleur effort par distance.",
    },
    "plot.records_table.label": {"en": "Records table", "fr": "Tableau des records"},
    "plot.records_table.description": {
        "en": "Your current best time and pace for each distance.",
        "fr": "Votre meilleur temps et allure actuels pour chaque distance.",
    },
    "plot.records_table.title": {"en": "Personal records", "fr": "Records personnels"},
    "plot.stream_evolution.label": {
        "en": "Signal inside the activity", "fr": "Signal dans l'activité",
    },
    "plot.stream_evolution.description": {
        "en": "One line per activity for a chosen signal (GAP, pace, HR, power, "
        "altitude, gradient), over time or distance.",
        "fr": "Une courbe par activité pour un signal choisi (GAP, allure, FC, "
        "puissance, altitude, pente), en temps ou en distance.",
    },
    "plot.gap_curve.label": {"en": "GAP curves", "fr": "Courbes GAP"},
    "plot.gap_curve.description": {
        "en": "Fits your personal gradient-adjusted-pace models on the selected "
        "activities and overlays the reference curves.",
        "fr": "Ajuste vos modèles personnels d'allure ajustée à la pente sur les "
        "activités sélectionnées et superpose les courbes de référence.",
    },
    "plot.metric_scatter.label": {"en": "Metric vs metric", "fr": "Métrique vs métrique"},
    "plot.metric_scatter.description": {
        "en": "One point per activity, any metric against any other, with an "
        "optional trendline.",
        "fr": "Un point par activité, n'importe quelle métrique contre une autre, "
        "avec une droite de tendance optionnelle.",
    },
    "plot.metric_distribution.label": {"en": "Distribution", "fr": "Distribution"},
    "plot.metric_distribution.description": {
        "en": "How a metric is spread across the selected activities.",
        "fr": "Comment une métrique se répartit sur les activités sélectionnées.",
    },
    "plot.data_table.label": {"en": "Table", "fr": "Tableau"},
    "plot.data_table.description": {
        "en": "The raw feature table — pick your columns, one row per activity or "
        "per group, downloadable as CSV.",
        "fr": "Le tableau de données brut — choisissez vos colonnes, une ligne par "
        "activité ou par groupe, téléchargeable en CSV.",
    },
    "plot.data_table.title": {"en": "Activities", "fr": "Activités"},

    "plot.text_block.label": {"en": "Text", "fr": "Texte"},
    "plot.text_block.description": {
        "en": "A block of your own text — a title, a comment, what you concluded. "
        "Reads no activity data.",
        "fr": "Un bloc de texte libre — un titre, un commentaire, votre conclusion. "
        "N'utilise aucune donnée d'activité.",
    },
    "plot.image_block.label": {"en": "Image", "fr": "Image"},
    "plot.image_block.description": {
        "en": "An image in the panel: upload one, or point at a URL.",
        "fr": "Une image dans le panneau : téléversez-la, ou indiquez une URL.",
    },

    # --- Shared plot messages ------------------------------------------------
    "plot.no_data": {
        "en": "No data for this selection.", "fr": "Aucune donnée pour cette sélection.",
    },
    "plot.metric_unavailable": {
        "en": "{metric} is not available for these activities.",
        "fr": "{metric} n'est pas disponible pour ces activités.",
    },
    "plot.unknown_type": {
        "en": "Unknown plot type: {type}", "fr": "Type de graphique inconnu : {type}",
    },
    "plot.x.time": {"en": "Time", "fr": "Temps"},
    "plot.x.months_since_start": {
        "en": "Months since the group started", "fr": "Mois depuis le début du groupe",
    },
    "plot.months": {"en": "months", "fr": "mois"},
    "plot.trend.cumulative_ignored": {
        "en": "Cumulative is ignored here: a running total of averages or ratios "
        "has no meaning.",
        "fr": "Le cumul est ignoré ici : un total cumulé de moyennes ou de ratios "
        "n'a pas de sens.",
    },
    "plot.trend.totals": {"en": "Totals per group", "fr": "Totaux par groupe"},
    "plot.records.none": {
        "en": "No record found for the selected distances.",
        "fr": "Aucun record trouvé pour les distances sélectionnées.",
    },
    "plot.distribution.title": {
        "en": "Distribution of {metric}", "fr": "Distribution de {metric}",
    },
    "plot.distribution.count": {"en": "activities", "fr": "activités"},
    "plot.distribution.pct": {"en": "%", "fr": "%"},
    "plot.distribution.y_count": {
        "en": "Number of activities", "fr": "Nombre d'activités",
    },
    "plot.distribution.y_pct": {"en": "% of activities", "fr": "% des activités"},
    "plot.scatter.title": {"en": "{y} vs {x}", "fr": "{y} vs {x}"},
    "plot.scatter.trend": {"en": "trend", "fr": "tendance"},
    "plot.scatter.trend_unavailable": {
        "en": "Not enough spread to fit a trendline for {series}.",
        "fr": "Pas assez de dispersion pour ajuster une tendance sur {series}.",
    },
    "plot.stream.no_stream_data": {
        "en": "None of the selected activities has per-second data.",
        "fr": "Aucune des activités sélectionnées n'a de données par seconde.",
    },
    "plot.stream.truncated": {
        "en": "Showing the first {shown} of {total} activities — raise the limit to "
        "see more.",
        "fr": "Affichage des {shown} premières activités sur {total} — augmentez la "
        "limite pour en voir plus.",
    },

    # --- Activity metrics ----------------------------------------------------
    "metric.distance_km": {"en": "Distance", "fr": "Distance"},
    "metric.elevation_gain_m": {"en": "Elevation gain", "fr": "Dénivelé positif"},
    "metric.moving_time": {"en": "Moving time", "fr": "Temps en mouvement"},
    "metric.activity_count": {"en": "Number of activities", "fr": "Nombre d'activités"},
    "metric.avg_pace": {"en": "Average pace", "fr": "Allure moyenne"},
    "metric.avg_gap_pace": {"en": "Average GAP pace", "fr": "Allure GAP moyenne"},
    "metric.avg_speed_kmh": {"en": "Average speed", "fr": "Vitesse moyenne"},
    "metric.avg_gradient_pct": {"en": "Average gradient", "fr": "Pente moyenne"},
    "metric.elevation_per_km": {"en": "Elevation per km", "fr": "Dénivelé par km"},
    "metric.avg_hr": {"en": "Average heart rate", "fr": "Fréquence cardiaque moyenne"},
    "metric.max_hr": {"en": "Max heart rate", "fr": "Fréquence cardiaque max"},
    "metric.avg_power_w": {"en": "Average power", "fr": "Puissance moyenne"},
    "metric.power_to_hr": {"en": "Power-to-HR", "fr": "Puissance / FC"},
    "metric.relative_effort": {
        "en": "Relative Effort (Strava)", "fr": "Effort relatif (Strava)",
    },
    "metric.best.1_km": {"en": "Best 1 km", "fr": "Meilleur 1 km"},
    "metric.best.3_km": {"en": "Best 3 km", "fr": "Meilleur 3 km"},
    "metric.best.5_km": {"en": "Best 5 km", "fr": "Meilleur 5 km"},
    "metric.best.10_km": {"en": "Best 10 km", "fr": "Meilleur 10 km"},
    "metric.best.semi": {"en": "Best half marathon", "fr": "Meilleur semi"},
    "metric.best.marathon": {"en": "Best marathon", "fr": "Meilleur marathon"},
    "metric.best.50_km": {"en": "Best 50 km", "fr": "Meilleur 50 km"},
    "metric.best.100_km": {"en": "Best 100 km", "fr": "Meilleur 100 km"},
    "metric.best.150_km": {"en": "Best 150 km", "fr": "Meilleur 150 km"},

    # --- Aggregations & granularities ---------------------------------------
    "agg.sum": {"en": "Sum", "fr": "Somme"},
    "agg.mean": {"en": "Average", "fr": "Moyenne"},
    "agg.median": {"en": "Median", "fr": "Médiane"},
    "agg.max": {"en": "Maximum", "fr": "Maximum"},
    "agg.min": {"en": "Minimum", "fr": "Minimum"},
    "agg.count": {"en": "Count", "fr": "Nombre"},
    "gran.activity": {"en": "Per activity", "fr": "Par activité"},
    "gran.day": {"en": "Day", "fr": "Jour"},
    "gran.week": {"en": "Week", "fr": "Semaine"},
    "gran.month": {"en": "Month", "fr": "Mois"},
    "gran.quarter": {"en": "Quarter", "fr": "Trimestre"},
    "gran.year": {"en": "Year", "fr": "Année"},

    # --- Stream signals ------------------------------------------------------
    "signal.gap_pace": {"en": "GAP pace", "fr": "Allure GAP"},
    "signal.pace": {"en": "Pace", "fr": "Allure"},
    "signal.pace.y": {"en": "Pace (min/km)", "fr": "Allure (min/km)"},
    "signal.heartrate": {"en": "Heart rate", "fr": "Fréquence cardiaque"},
    "signal.power": {"en": "Power", "fr": "Puissance"},
    "signal.power_to_hr": {"en": "Power-to-HR", "fr": "Puissance / FC"},
    "signal.altitude": {"en": "Altitude", "fr": "Altitude"},
    "signal.altitude.y": {"en": "Altitude (m)", "fr": "Altitude (m)"},
    "signal.gradient": {"en": "Gradient", "fr": "Pente"},
    "signal.gradient.y": {"en": "Gradient (%)", "fr": "Pente (%)"},

    # --- Plot parameters -----------------------------------------------------
    "param.metric": {"en": "Metric", "fr": "Métrique"},
    "param.metric.help": {
        "en": "Adding a metric to the registry makes it available in every plot "
        "that takes one.",
        "fr": "Ajouter une métrique au registre la rend disponible dans tous les "
        "graphiques qui en acceptent une.",
    },
    "param.aggregation": {"en": "Aggregation", "fr": "Agrégation"},
    "param.metric2": {"en": "Second metric", "fr": "Seconde métrique"},
    "param.metric2.none": {"en": "None", "fr": "Aucune"},
    "param.metric2.help": {
        "en": "Draws a second metric on the same chart, against its own axis on the "
              "right. Useful when the two move together — distance and climb, heart "
              "rate and pace. Because the two axes are scaled independently, where "
              "the series cross means nothing; compare their shapes, not their "
              "crossings.",
        "fr": "Trace une seconde métrique sur le même graphique, avec son propre axe "
              "à droite. Utile quand les deux évoluent ensemble — distance et "
              "dénivelé, fréquence cardiaque et allure. Les deux axes étant mis à "
              "l'échelle indépendamment, les croisements des courbes ne signifient "
              "rien : comparez les formes, pas les intersections.",
    },
    "param.aggregation2": {
        "en": "Second aggregation", "fr": "Agrégation de la seconde",
    },
    "param.chart2": {"en": "Second chart type", "fr": "Type de la seconde"},
    "param.granularity": {"en": "Granularity", "fr": "Granularité"},
    "param.x_mode": {"en": "X axis", "fr": "Axe X"},
    "param.x_mode.calendar": {"en": "Calendar", "fr": "Calendrier"},
    "param.x_mode.elapsed": {"en": "Aligned to group start", "fr": "Aligné sur le début"},
    "param.x_mode.help": {
        "en": "Aligned mode draws every time window from a common zero, so blocks "
        "of different lengths compare directly.",
        "fr": "Le mode aligné trace chaque fenêtre depuis un zéro commun, pour "
        "comparer directement des blocs de durées différentes.",
    },
    "param.cumulative": {"en": "Cumulative", "fr": "Cumulé"},
    "param.chart": {"en": "Chart", "fr": "Graphique"},
    "param.chart.line": {"en": "Line", "fr": "Ligne"},
    "param.chart.step": {"en": "Step", "fr": "Escalier"},
    "param.chart.bar": {"en": "Bars", "fr": "Barres"},
    "param.chart.area": {"en": "Area", "fr": "Aire"},
    "param.markers": {"en": "Show points", "fr": "Afficher les points"},
    "param.split_by": {"en": "Split series by", "fr": "Séparer les séries par"},
    "param.split_by.none": {"en": "Nothing", "fr": "Rien"},
    "param.split_by.sport": {"en": "Sport type", "fr": "Type de sport"},
    "param.smooth_rolling": {"en": "Rolling mean (points)", "fr": "Moyenne glissante (points)"},
    "param.smooth_rolling.help": {
        "en": "0 disables it. Smooths the already-binned curve.",
        "fr": "0 désactive. Lisse la courbe déjà groupée.",
    },
    "param.smooth_savgol": {"en": "Savitzky–Golay (points)", "fr": "Savitzky–Golay (points)"},
    "param.smooth_savgol.help": {"en": "0 disables it.", "fr": "0 désactive."},
    "param.show_totals": {"en": "Show totals table", "fr": "Afficher le tableau des totaux"},
    "param.show_totals.help": {
        "en": "Adds one aggregate per group beside the chart.",
        "fr": "Ajoute un agrégat par groupe à côté du graphique.",
    },
    "param.bands": {"en": "Gradient bands", "fr": "Catégories de pente"},
    "param.bands.help": {
        "en": "Bands are stacked in physical order, descent at the bottom.",
        "fr": "Les catégories sont empilées dans l'ordre physique, descente en bas.",
    },
    "param.bins": {"en": "Bins", "fr": "Classes"},
    "param.normalize": {"en": "As a percentage", "fr": "En pourcentage"},
    "param.normalize.help": {
        "en": "Compare the shape of groups of different sizes.",
        "fr": "Comparer la forme de groupes de tailles différentes.",
    },
    "param.x_metric": {"en": "X metric", "fr": "Métrique X"},
    "param.y_metric": {"en": "Y metric", "fr": "Métrique Y"},
    "param.color_by": {"en": "Colour by", "fr": "Couleur par"},
    "param.color_by.group": {"en": "Group", "fr": "Groupe"},
    "param.color_by.sport": {"en": "Sport type", "fr": "Type de sport"},
    "param.trendline": {"en": "Trendline", "fr": "Droite de tendance"},
    "param.trendline.help": {
        "en": "Least-squares fit across the observed range.",
        "fr": "Régression des moindres carrés sur la plage observée.",
    },
    "param.rows": {"en": "One row per", "fr": "Une ligne par"},
    "param.rows.activity": {"en": "Activity", "fr": "Activité"},
    "param.rows.group": {"en": "Group", "fr": "Groupe"},
    "param.rows.help": {
        "en": "Group rows aggregate every activity of the group.",
        "fr": "Les lignes par groupe agrègent toutes les activités du groupe.",
    },
    "param.columns": {"en": "Columns", "fr": "Colonnes"},
    "param.highlight_best": {"en": "Highlight the best value", "fr": "Mettre en avant la meilleure valeur"},
    "param.highlight_best.help": {
        "en": "Only for metrics that have a meaningful best.",
        "fr": "Uniquement pour les métriques ayant un « meilleur » qui a du sens.",
    },
    "param.sort_by": {"en": "Sort by", "fr": "Trier par"},
    "param.descending": {"en": "Descending", "fr": "Décroissant"},
    "param.limit": {"en": "Row limit", "fr": "Limite de lignes"},
    "param.limit.help": {"en": "0 shows every row.", "fr": "0 affiche toutes les lignes."},
    "param.distances": {"en": "Distances", "fr": "Distances"},
    "param.record_display": {"en": "Show", "fr": "Afficher"},
    "param.record_display.pace": {"en": "Pace", "fr": "Allure"},
    "param.record_display.time": {"en": "Time", "fr": "Temps"},
    "param.record_display.help": {
        "en": "Pace is comparable across distances; time is the raw record.",
        "fr": "L'allure est comparable entre distances ; le temps est le record brut.",
    },
    "param.extend_to_last": {"en": "Extend to the last activity", "fr": "Prolonger jusqu'à la dernière activité"},
    "param.extend_to_last.help": {
        "en": "Carries the current record flat to the edge of the plot.",
        "fr": "Prolonge le record actuel à plat jusqu'au bord du graphique.",
    },
    "param.split_by_group": {"en": "One series per group", "fr": "Une série par groupe"},
    "param.split_by_group.help": {
        "en": "Off means records are all-time across every selected activity.",
        "fr": "Désactivé, les records sont calculés sur toutes les activités.",
    },
    "param.per_group": {"en": "One row per group", "fr": "Une ligne par groupe"},
    "param.per_group.help": {
        "en": "Compare each window's own best.",
        "fr": "Comparer le meilleur de chaque fenêtre.",
    },
    "param.signals": {"en": "Signals", "fr": "Signaux"},
    "param.signals.help": {
        "en": "Any number at once. Signals sharing a unit (GAP and raw pace, say) "
              "share an axis; a second unit gets the right-hand axis; everything "
              "past that folds onto whichever axis is closest.",
        "fr": "Autant que voulu à la fois. Les signaux qui partagent une unité "
              "(GAP et allure brute, par exemple) partagent un axe ; une seconde "
              "unité obtient l'axe de droite ; le reste se replie sur l'axe le "
              "plus proche.",
    },
    "param.x_axis": {"en": "X axis", "fr": "Axe X"},
    "param.as_speed": {"en": "Show as speed", "fr": "Afficher en vitesse"},
    "param.as_speed.help": {
        "en": "km/h instead of min/km — higher is faster.",
        "fr": "km/h au lieu de min/km — plus haut = plus rapide.",
    },
    "param.max_series": {"en": "Max activities shown", "fr": "Activités affichées max"},
    "param.max_series.help": {
        "en": "Keeps a large selection readable; the plot says when it truncates.",
        "fr": "Garde une grande sélection lisible ; le graphique signale la troncature.",
    },
    "param.smoothing": {"en": "Smoothing", "fr": "Lissage"},
    "param.filter.rolling_s": {"en": "Rolling mean (s)", "fr": "Moyenne glissante (s)"},
    "param.filter.savgol_m": {"en": "Savitzky–Golay (m)", "fr": "Savitzky–Golay (m)"},
    "param.gap_models": {"en": "Personal models", "fr": "Modèles personnels"},
    "param.gap_references": {"en": "Reference curves", "fr": "Courbes de référence"},
    "param.hr_bands": {"en": "Heart-rate bands", "fr": "Zones de fréquence cardiaque"},
    "param.hr_bands.help": {
        "en": "Leave empty for one curve per model. Add bands to stratify the same "
        "fit by intensity.",
        "fr": "Laissez vide pour une courbe par modèle. Ajoutez des zones pour "
        "stratifier le même ajustement par intensité.",
    },
    "param.hr_band.name": {"en": "Name", "fr": "Nom"},
    "param.hr_band.min": {"en": "Min bpm", "fr": "FC min"},
    "param.hr_band.max": {"en": "Max bpm", "fr": "FC max"},

    # Content blocks (text, image) — shared alignment and tone vocabularies first.
    "param.align.left": {"en": "Left", "fr": "À gauche"},
    "param.align.center": {"en": "Centered", "fr": "Centré"},
    "param.tone.none": {"en": "None", "fr": "Aucune"},
    "param.tone.forest": {"en": "Green", "fr": "Vert"},
    "param.tone.terracotta": {"en": "Clay", "fr": "Terre cuite"},
    "param.tone.sunrise": {"en": "Amber", "fr": "Ambre"},
    "param.tone.plum": {"en": "Plum", "fr": "Prune"},

    "param.text.body": {"en": "Text", "fr": "Texte"},
    "param.text.body.help": {
        "en": "Line breaks are kept. Your own words, in your own language — this is "
        "the one string in the app that is never translated.",
        "fr": "Les retours à la ligne sont conservés. Vos propres mots, dans votre "
        "langue — c'est le seul texte de l'application qui n'est jamais traduit.",
    },
    "param.text.variant": {"en": "Style", "fr": "Style"},
    "param.text.variant.body": {"en": "Paragraph", "fr": "Paragraphe"},
    "param.text.variant.lede": {"en": "Intro", "fr": "Introduction"},
    "param.text.variant.heading": {"en": "Heading", "fr": "Titre"},
    "param.text.variant.quote": {"en": "Quote", "fr": "Citation"},
    "param.text.align": {"en": "Alignment", "fr": "Alignement"},
    "param.text.tone": {"en": "Highlight", "fr": "Mise en avant"},

    "param.image.src": {"en": "Image", "fr": "Image"},
    "param.image.src.help": {
        "en": "Upload a file (PNG, JPEG, WebP or GIF, up to 4 MB) or paste a URL.",
        "fr": "Téléversez un fichier (PNG, JPEG, WebP ou GIF, jusqu'à 4 Mo) ou "
        "collez une URL.",
    },
    "param.image.caption": {"en": "Caption", "fr": "Légende"},
    "param.image.alt": {"en": "Alt text", "fr": "Texte alternatif"},
    "param.image.alt.help": {
        "en": "What the image shows, for anyone who cannot see it.",
        "fr": "Ce que montre l'image, pour qui ne peut pas la voir.",
    },
    "param.image.width": {"en": "Width (%)", "fr": "Largeur (%)"},
    "param.image.width.help": {
        "en": "Share of the panel's width.",
        "fr": "Part de la largeur du panneau.",
    },
    "param.image.align": {"en": "Alignment", "fr": "Alignement"},

    # --- GAP plot messages ---------------------------------------------------
    "gap.group_no_splits": {
        "en": "{label}: no usable split found — the activities may be too short or "
        "lack heart rate.",
        "fr": "{label} : aucun segment exploitable — les activités sont peut-être "
        "trop courtes ou sans fréquence cardiaque.",
    },
    "gap.curve_unavailable": {
        "en": "{label}: curve unavailable ({error}).",
        "fr": "{label} : courbe indisponible ({error}).",
    },
    "gap.reason.no_calibration": {
        "en": "no flat section shares a heart rate with a climbing section, so the "
        "adjustment cannot be learned",
        "fr": "aucune section plate ne partage une fréquence cardiaque avec une "
        "section en montée, l'ajustement ne peut pas être appris",
    },
    "gap.reason.empty_curve": {
        "en": "no sample falls in this range",
        "fr": "aucun échantillon dans cette plage",
    },

    # --- Built-in example pages ---------------------------------------------
    "dash.window.all_history": {"en": "All history", "fr": "Tout l'historique"},
    "dash.gap.panel.curves": {"en": "GAP curves", "fr": "Courbes GAP"},
    "dash.gap.panel.per_year": {"en": "One curve per year", "fr": "Une courbe par an"},
    "dash.gap.panel.intensity": {"en": "By intensity", "fr": "Par intensité"},
    "dash.races.panel.selection": {
        "en": "Selected workouts", "fr": "Séances sélectionnées",
    },
    "dash.races.selection_label": {"en": "Selection", "fr": "Sélection"},
    "dash.ltp.panel.volume": {
        "en": "Volume: distance and elevation", "fr": "Volume : distance et dénivelé",
    },
    "dash.ltp.panel.terrain": {"en": "Terrain", "fr": "Terrain"},

    # --- Web app chrome ------------------------------------------------------
    # Everything under `ui.` is shipped to the browser in one payload by
    # `ui_strings_payload`, keyed on the part after the prefix. The web app holds no
    # translation table of its own, so this block is the only place its wording
    # lives — see the module docstring.
    "ui.nav.home": {"en": "Home", "fr": "Accueil"},
    "ui.nav.analysis": {"en": "Analysis", "fr": "Analysis"},
    "ui.nav.training": {"en": "Training", "fr": "Entraînement"},
    "ui.nav.sign_out": {"en": "Sign out", "fr": "Se déconnecter"},

    "ui.common.loading": {"en": "Loading…", "fr": "Chargement…"},
    "ui.common.close": {"en": "Close", "fr": "Fermer"},
    "ui.common.not_set": {"en": "Not set", "fr": "Non renseigné"},
    "ui.common.saving": {"en": "saving", "fr": "enregistrement"},
    "ui.common.not_saved": {
        "en": "Not saved — check the value.",
        "fr": "Non enregistré — vérifiez la valeur.",
    },
    "ui.common.km": {"en": "km", "fr": "km"},
    "ui.common.metres": {"en": "m", "fr": "m"},
    "ui.common.kg": {"en": "kg", "fr": "kg"},
    "ui.common.cm": {"en": "cm", "fr": "cm"},
    "ui.common.years": {"en": "years", "fr": "ans"},
    "ui.common.hours": {"en": "h", "fr": "h"},

    # Home — profile card
    "ui.home.profile.title": {"en": "Athlete History", "fr": "Historique de l'athlète"},
    "ui.home.profile.activities": {"en": "Activities", "fr": "Activités"},
    "ui.home.profile.oldest": {"en": "First run", "fr": "Première sortie"},
    "ui.home.profile.newest": {"en": "Latest run", "fr": "Dernière sortie"},
    "ui.home.profile.total_distance": {"en": "Total distance", "fr": "Distance totale"},
    "ui.home.profile.total_elevation": {"en": "Total climb", "fr": "Dénivelé total"},
    "ui.home.profile.total_time": {"en": "Total time on feet", "fr": "Temps total de course"},
    "ui.home.profile.furthest": {"en": "Furthest run", "fr": "Sortie la plus longue"},
    "ui.home.profile.longest": {"en": "Longest run", "fr": "Sortie la plus durable"},
    "ui.home.profile.records": {"en": "Current records", "fr": "Records actuels"},
    "ui.home.profile.records_empty": {
        "en": "No full-distance efforts yet — records appear once an activity covers "
              "the distance.",
        "fr": "Aucun effort complet pour l'instant — les records apparaissent dès "
              "qu'une activité couvre la distance.",
    },

    # Home — health card
    "ui.home.health.title": {
        "en": "Athlete Health Metrics", "fr": "Indicateurs de santé de l'athlète",
    },
    "ui.home.health.age": {"en": "Age", "fr": "Âge"},
    "ui.home.health.experience": {"en": "Years running", "fr": "Années de course"},
    "ui.home.health.weight": {"en": "Weight", "fr": "Poids"},
    "ui.home.health.weight_help": {
        "en": "Unlocks the power metrics.",
        "fr": "Débloque les métriques de puissance.",
    },
    "ui.home.health.height": {"en": "Height", "fr": "Taille"},

    # Home — training zones and VMA pace (display-only, see ZonesCard)
    "ui.home.zones.title": {
        "en": "Athlete Performance Metrics", "fr": "Indicateurs de performance de l'athlète",
    },
    "ui.home.zones.subtitle": {
        "en": "Just for reference — nothing here feeds a calculation.",
        "fr": "Juste pour référence — rien ici n'alimente un calcul.",
    },
    "ui.home.zones.z1": {"en": "Z1max", "fr": "Z1max"},
    "ui.home.zones.z2": {"en": "Z2max", "fr": "Z2max"},
    "ui.home.zones.z3": {"en": "Z3max", "fr": "Z3max"},
    "ui.home.zones.z4": {"en": "Z4max", "fr": "Z4max"},
    "ui.home.zones.hr_max": {"en": "HRmax", "fr": "FCmax"},
    "ui.home.zones.vma": {"en": "VMA pace", "fr": "Allure VMA"},
    "ui.home.zones.pace_z2": {"en": "Easy endurance", "fr": "Endurance fondamentale"},
    "ui.home.zones.pace_endurance": {"en": "Active Endurance", "fr": "Endurance active"},
    "ui.home.zones.pace_threshold": {"en": "Threshold", "fr": "Seuil"},
    "ui.home.zones.pace_intervals": {"en": "Intervals", "fr": "Intervalles"},
    "ui.home.zones.pace_reps": {"en": "Reps", "fr": "Répétitions"},
    "ui.home.zones.unlocked_by_vma": {
        "en": "Unlocked by giving VMA", "fr": "Débloqué en renseignant la VMA",
    },
    "ui.home.zones.unlocked_by_hrmax": {
        "en": "Derived from HRmax", "fr": "Déduit de la FCmax",
    },
    "ui.home.zones.hr_map_title": {
        "en": "Pace zones, mapped onto heart rate",
        "fr": "Allures, projetées sur la fréquence cardiaque",
    },
    "ui.home.zones.hr_map_needs_hrmax": {
        "en": "Set your HRmax above to see where each pace zone falls.",
        "fr": "Renseignez votre FCmax ci-dessus pour voir où se situe chaque allure.",
    },

    # Home — last activity and the weekly volume chart
    "ui.home.last.title": {"en": "Last activity", "fr": "Dernière activité"},
    "ui.home.last.empty": {
        "en": "Nothing imported yet.", "fr": "Rien d'importé pour l'instant.",
    },
    "ui.home.last.distance": {"en": "Distance", "fr": "Distance"},
    "ui.home.last.climb": {"en": "Climb", "fr": "Dénivelé"},
    "ui.home.last.time": {"en": "Moving time", "fr": "Temps en mouvement"},
    "ui.home.last.pace": {"en": "Pace", "fr": "Allure"},
    "ui.home.last.heart_rate": {"en": "Avg HR", "fr": "FC moyenne"},
    "ui.home.last.map_loading": {
        "en": "Loading the route…", "fr": "Chargement du parcours…",
    },
    "ui.home.last.map_none": {
        "en": "This activity has no GPS route — a treadmill run or a manual entry.",
        "fr": "Cette activité n'a pas de tracé GPS — tapis de course ou saisie "
              "manuelle.",
    },
    "ui.home.last.map_unavailable": {
        "en": "The route could not be fetched from Strava just now.",
        "fr": "Le parcours n'a pas pu être récupéré depuis Strava pour le moment.",
    },
    "ui.home.form.title": {"en": "Recent form", "fr": "Forme récente"},
    "ui.home.form.subtitle": {
        "en": "Power per heartbeat, weekly, smoothed over four weeks. It rises when "
              "the same effort buys you more pace — and unlike raw pace it does not "
              "care whether the week was hilly or flat.",
        "fr": "Puissance par battement, par semaine, lissée sur quatre semaines. "
              "Elle monte quand le même effort vous rapporte plus d'allure — et "
              "contrairement à l'allure brute, elle ne dépend pas du relief de la "
              "semaine.",
    },
    "ui.home.form.needs_weight": {
        "en": "Power is modelled from your body mass, so this chart needs your "
              "weight — set it on the Health card above.",
        "fr": "La puissance est modélisée à partir de votre masse corporelle : ce "
              "graphique a besoin de votre poids — renseignez-le dans la carte "
              "Santé ci-dessus.",
    },
    "ui.home.recent.title": {"en": "Recent History", "fr": "Historique récent"},
    "ui.home.recent.subtitle": {
        "en": "Distance and climb per week over the last 30 weeks. Each has its own "
              "axis — distance on the left, climb on the right — so compare the "
              "shapes rather than where the two meet.",
        "fr": "Distance et dénivelé par semaine sur les 30 dernières semaines. "
              "Chacun a son axe — distance à gauche, dénivelé à droite — comparez "
              "donc les formes plutôt que les points de rencontre.",
    },

    # Home — importing from Strava
    "ui.home.import.title": {"en": "Your data", "fr": "Vos données"},
    "ui.home.import.first": {"en": "Import my activities", "fr": "Importer mes activités"},
    "ui.home.import.more": {
        "en": "Import new activities", "fr": "Importer les nouvelles activités",
    },
    "ui.home.import.again": {"en": "Re-import everything", "fr": "Tout réimporter"},
    "ui.home.import.again_help": {
        "en": "Re-fetch and recompute everything. Slow, and spends the Strava rate "
              "limit.",
        "fr": "Tout retélécharger et recalculer. Lent, et consomme le quota Strava.",
    },
    "ui.home.import.running": {"en": "Importing from Strava…", "fr": "Import depuis Strava…"},
    "ui.home.import.failed": {"en": "Last import failed", "fr": "Dernier import échoué"},
    "ui.home.import.last": {"en": "Last import", "fr": "Dernier import"},
    "ui.home.import.empty": {
        "en": "Import your activities to start building pages — every plot works off "
              "that data.",
        "fr": "Importez vos activités pour commencer à construire des pages — tous "
              "les graphiques s'appuient sur ces données.",
    },

    # My Pages
    "ui.pages.title": {"en": "Analysis", "fr": "Analysis"},
    "ui.pages.how.title": {
        "en": "How an analysis works", "fr": "Comment fonctionne une analyse",
    },
    "ui.pages.how.body": {
        "en": "An analysis is yours to assemble. You add panels; each panel takes one "
              "data source and as many plots as you like over it.",
        "fr": "Une analyse se construit. Vous ajoutez des panneaux ; chaque panneau "
              "prend une source de données et autant de graphiques que vous voulez.",
    },
    "ui.pages.how.step1.title": {"en": "1. Pick a data source", "fr": "1. Choisir des données"},
    "ui.pages.how.step1.body": {
        "en": "Specific activities, one date range, or several named periods to "
              "compare side by side.",
        "fr": "Des activités précises, une période, ou plusieurs périodes nommées à "
              "comparer côte à côte.",
    },
    "ui.pages.how.step2.title": {"en": "2. Add plots", "fr": "2. Ajouter des graphiques"},
    "ui.pages.how.step2.body": {
        "en": "Any metric, at any granularity, as a trend, distribution, scatter or "
              "table. Each plot brings its own form.",
        "fr": "N'importe quelle métrique, à n'importe quelle granularité : tendance, "
              "distribution, nuage de points ou tableau. Chaque graphique amène son "
              "propre formulaire.",
    },
    "ui.pages.how.step3.title": {"en": "3. Keep it", "fr": "3. La conserver"},
    "ui.pages.how.step3.body": {
        "en": "An analysis is saved as a document, so it reopens exactly as you left "
              "it. The three you start with work the same way — edit them freely.",
        "fr": "Une analyse est enregistrée comme un document : elle se rouvre "
              "exactement comme vous l'avez laissée. Les trois analyses fournies "
              "fonctionnent pareil — modifiez-les librement.",
    },
    "ui.pages.new.button": {"en": "New analysis", "fr": "Nouvelle analyse"},
    "ui.pages.new.hint": {
        "en": "Start from an empty analysis and add your first panel.",
        "fr": "Partez d'une analyse vide et ajoutez votre premier panneau.",
    },
    "ui.pages.new.prompt": {"en": "Name your analysis", "fr": "Nommez votre analyse"},
    "ui.pages.new.default_name": {"en": "My analysis", "fr": "Mon analyse"},
    # The header of one analysis.
    "ui.page.recompute": {"en": "Recompute", "fr": "Recalculer"},
    "ui.page.duplicate": {"en": "Duplicate", "fr": "Dupliquer"},
    "ui.page.delete": {"en": "Delete", "fr": "Supprimer"},
    "ui.page.add_panel": {"en": "Add a panel", "fr": "Ajouter un panneau"},
    "ui.page.default": {"en": "Default", "fr": "Par défaut"},
    "ui.page.default_help": {
        "en": "This analysis ships with the app, so it cannot be deleted. Everything "
              "else about it is editable — duplicate it if you want a version you can "
              "remove.",
        "fr": "Cette analyse est fournie avec l'application : elle ne peut pas être "
              "supprimée. Tout le reste est modifiable — dupliquez-la si vous voulez "
              "une version que vous pouvez supprimer.",
    },

    "ui.pages.panel_count.one": {"en": "{count} panel", "fr": "{count} panneau"},
    "ui.pages.panel_count.many": {"en": "{count} panels", "fr": "{count} panneaux"},
    "ui.pages.plot_count.one": {"en": "{count} plot", "fr": "{count} graphique"},
    "ui.pages.plot_count.many": {"en": "{count} plots", "fr": "{count} graphiques"},

    # Email — asked for once, right after the first sign-in.
    "ui.email.title": {
        "en": "One last thing: your email", "fr": "Une dernière chose : votre email",
    },
    "ui.email.body": {
        "en": "Strava does not share email addresses, so we have to ask. It is how "
              "we reach you about your account and about what changes in the app.",
        "fr": "Strava ne communique pas les adresses email, nous devons donc vous la "
              "demander. C'est ainsi que nous vous joignons au sujet de votre compte "
              "et des évolutions de l'application.",
    },
    "ui.email.label": {"en": "Email address", "fr": "Adresse email"},
    "ui.email.placeholder": {"en": "you@example.com", "fr": "vous@exemple.com"},
    "ui.email.submit": {"en": "Continue", "fr": "Continuer"},
    "ui.email.invalid": {
        "en": "That does not look like an email address.",
        "fr": "Cela ne ressemble pas à une adresse email.",
    },
    "ui.email.missing": {
        "en": "We still need your email address.",
        "fr": "Il nous manque encore votre adresse email.",
    },
    "ui.email.provide": {"en": "Add it now", "fr": "L'ajouter maintenant"},
    "ui.home.health.email": {"en": "Email", "fr": "Email"},

    # Import — the automatic pass that runs when you connect.
    "ui.home.import.auto": {
        "en": "Checking Strava for new activities…",
        "fr": "Recherche de nouvelles activités sur Strava…",
    },
    "ui.home.import.auto_help": {
        "en": "New activities are imported automatically when you open the app. The "
              "buttons are there for when you want to force it.",
        "fr": "Les nouvelles activités sont importées automatiquement à l'ouverture "
              "de l'application. Les boutons sont là si vous voulez forcer l'import.",
    },

    # Background computation of the expensive plots.
    "ui.precompute.title": {"en": "Models", "fr": "Modèles"},
    "ui.precompute.running": {
        "en": "Fitting your GAP models in the background…",
        "fr": "Ajustement de vos modèles GAP en arrière-plan…",
    },
    "ui.precompute.help": {
        "en": "The GAP curves are model fits over your per-second data. They are "
              "computed once, per year of history, and kept — so the example page "
              "opens already drawn.",
        "fr": "Les courbes GAP sont des ajustements de modèles sur vos données "
              "seconde par seconde. Elles sont calculées une fois, par année "
              "d'historique, puis conservées — la page d'exemple s'ouvre donc déjà "
              "tracée.",
    },
    "ui.precompute.done": {"en": "Models ready", "fr": "Modèles prêts"},
    "ui.precompute.failed": {
        "en": "Could not finish fitting the models",
        "fr": "Impossible de terminer l'ajustement des modèles",
    },

    # Training: the calendar of planned workouts/goals/notes and completed sessions.
    "ui.training.add_plan": {"en": "+ Plan", "fr": "+ Planifier"},
    "ui.training.new_plan_title": {"en": "New plan", "fr": "Nouveau plan"},
    "ui.training.kind.workout": {"en": "Workout", "fr": "Séance"},
    "ui.training.kind.goal": {"en": "Goal", "fr": "Objectif"},
    "ui.training.kind.note": {"en": "Note", "fr": "Note"},
    "ui.training.form.title_placeholder": {"en": "Title", "fr": "Titre"},
    "ui.training.form.body_placeholder": {
        "en": "Notes — shown when opened",
        "fr": "Notes — visibles à l'ouverture",
    },
    "ui.training.form.save": {"en": "Save", "fr": "Enregistrer"},
    "ui.training.form.delete": {"en": "Delete", "fr": "Supprimer"},
    "ui.training.form.duplicate": {"en": "Duplicate", "fr": "Dupliquer"},
    "ui.training.form.importance_primary": {"en": "Primary", "fr": "Principal"},
    "ui.training.form.importance_secondary": {"en": "Secondary", "fr": "Secondaire"},
    "ui.training.form.end_date_label": {"en": "Until", "fr": "Jusqu'au"},
    "ui.training.badge.planned": {"en": "Planned", "fr": "Prévu"},
    "ui.training.badge.note": {"en": "Note", "fr": "Note"},
    "ui.training.badge.completed": {"en": "Completed", "fr": "Terminé"},
    "ui.training.week.running": {"en": "Running", "fr": "Course"},
    "ui.training.week.cycling": {"en": "Cycling", "fr": "Vélo"},

}

UI_PREFIX = "ui."


def translate(key: str, lang: str = DEFAULT_LANG) -> str:
    """Return the ``lang`` string for ``key``.

    Falls back to English, then to the raw key, so a missing translation degrades
    gracefully instead of raising.
    """
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(lang) or entry.get("en") or key


def ui_strings(lang: str = DEFAULT_LANG) -> dict:
    """Every ``ui.*`` string for ``lang``, keyed without the prefix.

    The web app's whole vocabulary in one object, so it can look up
    ``strings["nav.home"]`` without shipping a translation table of its own.
    """
    return {
        key[len(UI_PREFIX):]: translate(key, lang)
        for key in TRANSLATIONS
        if key.startswith(UI_PREFIX)
    }
