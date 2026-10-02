"""Central translation table for TAGG.

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
    "plot.races.speed.y": {
        "en": "Speed (km/h, higher = faster)",
        "fr": "Vitesse (km/h, plus haut = plus rapide)",
    },
    "plot.races.power.y": {"en": "Power (W)", "fr": "Puissance (W)"},
    "plot.races.power_per_kg.y": {"en": "Power (W/kg)", "fr": "Puissance (W/kg)"},
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

    "plot.metric_trend.label": {"en": "Metrics over time", "fr": "Métriques dans le temps"},
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
    "plot.fitness_fatigue.label": {
        "en": "Fitness & Fatigue", "fr": "Fitness & Fatigue",
    },
    "plot.fitness_fatigue.description": {
        "en": "Daily training load (Strava's Relative Effort, every sport "
        "combined), split into a slow-building Fitness curve and a "
        "fast-reacting Fatigue curve — the classic Banister model.",
        "fr": "Charge d'entraînement quotidienne (Effort relatif Strava, tous "
        "sports confondus), décomposée en une courbe de Fitness à évolution "
        "lente et une courbe de Fatigue à réaction rapide — le modèle de "
        "Banister.",
    },
    "plot.fitness_fatigue.fitness": {"en": "Fitness", "fr": "Fitness"},
    "plot.fitness_fatigue.fatigue": {"en": "Fatigue", "fr": "Fatigue"},
    "plot.fitness_fatigue.y": {
        "en": "Training load (Relative Effort)",
        "fr": "Charge d'entraînement (Effort relatif)",
    },
    "plot.weekly_feel.label": {
        "en": "Weekly effort & feel", "fr": "Effort et ressenti par semaine",
    },
    "plot.weekly_feel.description": {
        "en": "Your own weekly ratings against the model: average RPE as the "
        "curve, average feeling as the week's background colour, and the "
        "Fitness trend as the same tag the week summary shows. Answers whether "
        "the load you feel is paying off, and at what cost.",
        "fr": "Vos propres notes, semaine par semaine, face au modèle : le RPE "
        "moyen en courbe, le ressenti moyen en couleur de fond, et la tendance "
        "du Fitness sous forme du même tag que le résumé de la semaine. Répond "
        "à : est-ce que la charge que je ressens paie, et à quel prix ?",
    },
    "plot.weekly_feel.rpe": {"en": "Average RPE", "fr": "RPE moyen"},
    "plot.weekly_feel.y": {
        "en": "Average RPE (1-10)", "fr": "RPE moyen (1-10)",
    },
    "plot.weekly_feel.caption": {
        "en": "Background: the week's average feeling (red weak, gold ok, green "
        "strong). Tags: Fitness over the week — ↑ up, → stable, ↓ down. Weeks "
        "you did not rate leave a gap.",
        "fr": "Fond : ressenti moyen de la semaine (rouge faible, gold ok, vert "
        "fort). Tags : évolution du Fitness sur la semaine — ↑ hausse, → stable, "
        "↓ baisse. Les semaines non notées laissent un trou.",
    },
    "plot.weekly_feel.hover_rpe": {
        "en": "{label} {value}/10 · {count} rated",
        "fr": "{label} {value}/10 · {count} notée(s)",
    },
    "plot.weekly_feel.no_entries": {
        "en": "No RPE or feeling recorded over this period — rate your sessions "
        "on the Training screen and this chart fills in.",
        "fr": "Aucun RPE ni ressenti saisi sur cette période — notez vos séances "
        "dans l'écran Entraînement et ce graphique se remplira.",
    },
    "plot.fitness_fatigue.missing_relative_effort": {
        "en": "{count} activities without heart-rate data were not counted "
        "toward training load.",
        "fr": "{count} activités sans fréquence cardiaque n'ont pas été "
        "comptées dans la charge d'entraînement.",
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
    "metric.power_per_kg": {"en": "Power (W/kg)", "fr": "Puissance (W/kg)"},
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
    "signal.power_per_kg": {"en": "Power (W/kg)", "fr": "Puissance (W/kg)"},
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
    "ui.nav.analysis": {"en": "Analysis", "fr": "Analyses"},
    "ui.nav.training": {"en": "Training", "fr": "Entraînement"},
    "ui.nav.blog": {"en": "Blog", "fr": "Blog"},
    "ui.nav.sign_in_required": {
        "en": "Sign in to access this",
        "fr": "Connectez-vous pour y accéder",
    },
    "ui.nav.sign_out": {"en": "Sign out", "fr": "Se déconnecter"},
    "ui.nav.group_open": {"en": "Open", "fr": "Ouvert"},
    "ui.nav.group_strava": {"en": "With Strava", "fr": "Avec Strava"},
    "ui.nav.connect": {"en": "Connect Strava", "fr": "Connecter Strava"},

    # --- Visitor (no account): design/tagg/visitor.md -------------------------
    # Two free tiers, never sold as such: "Free · now" and "Free · with Strava".
    # No "Pro", "Premium" or "Unlock" anywhere.
    "ui.visitor.lede": {
        "en": "TAGG analyses your runs and helps you improve. Two tools are open to "
              "everyone; the rest opens when you connect Strava.",
        "fr": "TAGG analyse vos sorties et vous aide à progresser. Deux outils sont "
              "ouverts à tous ; le reste s’ouvre en connectant Strava.",
    },
    "ui.visitor.open.title": {"en": "Without an account", "fr": "Sans compte"},
    "ui.visitor.open.tier": {"en": "Free · now", "fr": "Gratuit · maintenant"},
    "ui.visitor.strava.title": {"en": "With Strava", "fr": "Avec Strava"},
    "ui.visitor.strava.tier": {"en": "Free · 1 click", "fr": "Gratuit · 1 clic"},
    "ui.visitor.race_plan": {
        "en": "The pace to hold on every stretch of your race, from its GPX.",
        "fr": "L’allure à tenir sur chaque portion de votre course, à partir de son GPX.",
    },
    "ui.visitor.blog": {
        "en": "Training methods and analyses, free to read.",
        "fr": "Méthodes d’entraînement et analyses, en libre lecture.",
    },
    "ui.visitor.home": {
        "en": "Your profile, your records and your latest runs in one place.",
        "fr": "Votre profil, vos records et vos dernières sorties au même endroit.",
    },
    "ui.visitor.analysis": {
        "en": "Charts built on your own activities: trends, models, comparisons.",
        "fr": "Des graphiques construits sur vos propres activités : évolutions, "
              "modèles, comparaisons.",
    },
    "ui.visitor.training": {
        "en": "Your week, your sessions and your goals on one calendar.",
        "fr": "Votre semaine, vos séances et vos objectifs sur un calendrier.",
    },
    "ui.visitor.connect": {"en": "Connect with Strava", "fr": "Se connecter avec Strava"},
    "ui.visitor.trust": {
        "en": "Free, no card. TAGG reads your Strava activities to analyse them; your "
              "Strava tokens are encrypted and never leave the server.",
        "fr": "Gratuit, sans carte. TAGG lit vos activités Strava pour les analyser ; vos "
              "jetons Strava sont chiffrés et ne quittent jamais le serveur.",
    },
    "ui.visitor.fine": {
        "en": "Free, no card. TAGG only reads your activities.",
        "fr": "Gratuit, sans carte. TAGG ne lit que vos activités.",
    },
    "ui.visitor.tier": {"en": "Free · with Strava", "fr": "Gratuit · avec Strava"},
    "ui.visitor.teaser.home.title": {
        "en": "Your runner’s dashboard",
        "fr": "Votre tableau de bord de coureur",
    },
    "ui.visitor.teaser.home.1": {
        "en": "Distance, climbing and time since your very first run",
        "fr": "Distance, dénivelé et temps cumulés depuis votre première sortie",
    },
    "ui.visitor.teaser.home.2": {
        "en": "Your records over 5 km, 10 km, half and full marathon",
        "fr": "Vos records sur 5 km, 10 km, semi et marathon",
    },
    "ui.visitor.teaser.home.3": {
        "en": "Your zones and paces, set from your max heart rate and VMA",
        "fr": "Vos zones et allures, calées sur votre FCmax et votre VMA",
    },
    "ui.visitor.teaser.analysis.title": {
        "en": "Analyses built on your own runs",
        "fr": "Des analyses construites sur vos sorties",
    },
    "ui.visitor.teaser.analysis.1": {
        "en": "A personal GAP curve: what a gradient costs you, specifically",
        "fr": "Une courbe GAP personnelle : ce que la pente vous coûte, à vous",
    },
    "ui.visitor.teaser.analysis.2": {
        "en": "Your durability: what the effort costs after two, three, five hours",
        "fr": "Votre durabilité : ce que l’effort coûte après deux, trois, cinq heures",
    },
    "ui.visitor.teaser.analysis.3": {
        "en": "Your seasons compared, and your races side by side",
        "fr": "Vos saisons comparées, et vos courses côte à côte",
    },
    "ui.visitor.teaser.training.title": {
        "en": "Your training week at a glance",
        "fr": "Votre semaine d’entraînement d’un coup d’œil",
    },
    "ui.visitor.teaser.training.1": {
        "en": "Your Strava sessions on a calendar, rated by RPE and feel",
        "fr": "Vos séances Strava sur un calendrier, notées en RPE et en ressenti",
    },
    "ui.visitor.teaser.training.2": {
        "en": "Your planned sessions, notes and race goals",
        "fr": "Vos séances prévues, vos notes et vos objectifs de course",
    },
    "ui.visitor.teaser.training.3": {
        "en": "Each week’s summary: volume, climbing, fitness",
        "fr": "Le bilan de chaque semaine : volume, dénivelé, forme",
    },
    "ui.visitor.more.race_plan": {
        "en": "With Strava, TAGG learns this plan from your own climbs.",
        "fr": "Avec Strava, TAGG apprend ce plan à partir de vos propres montées.",
    },
    "ui.visitor.more.blog": {
        "en": "With Strava, TAGG applies these methods to your own runs.",
        "fr": "Avec Strava, TAGG applique ces méthodes à vos propres sorties.",
    },
    "ui.visitor.more.link": {"en": "Connect Strava", "fr": "Connecter Strava"},

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
    # Card kickers (design/tagg/contrast.md § 4): the time window, or "you".
    "ui.home.kicker.you": {"en": "You", "fr": "Vous"},
    "ui.home.kicker.all_time": {"en": "All time", "fr": "Tout l'historique"},
    "ui.home.kicker.weeks": {"en": "Last {count} weeks", "fr": "{count} dernières semaines"},
    # The hero (design/tagg/components/Hero.md): the current week in three numbers.
    "ui.home.hero.week": {"en": "Week {number} · {range}", "fr": "Semaine {number} · {range}"},
    "ui.home.hero.volume": {"en": "Volume", "fr": "Volume"},
    "ui.home.hero.climb": {"en": "Climb", "fr": "D+"},
    "ui.home.hero.form": {"en": "Form", "fr": "Forme"},
    "ui.home.hero.import": {"en": "Import", "fr": "Importer"},
    "ui.home.hero.this_week": {"en": "This week", "fr": "Cette semaine"},
    "ui.home.recent.weekly_average": {"en": "Weekly average", "fr": "Moyenne par semaine"},
    "ui.home.efficiency.latest": {"en": "Latest week", "fr": "Dernière semaine"},
    "ui.home.form.fitness": {"en": "Fitness", "fr": "Fitness"},
    "ui.home.form.fatigue": {"en": "Fatigue", "fr": "Fatigue"},
    "ui.home.feel.latest": {"en": "Last rated week", "fr": "Dernière semaine notée"},
    "ui.home.feel.rpe": {"en": "Average RPE", "fr": "RPE moyen"},
    "ui.home.records.new": {"en": "New", "fr": "Nouveau"},
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
    "ui.home.last.title": {"en": "Last Run", "fr": "Dernière sortie"},
    "ui.home.last.empty": {
        "en": "Nothing imported yet.", "fr": "Rien d'importé pour l'instant.",
    },
    "ui.home.last.distance": {"en": "Distance", "fr": "Distance"},
    "ui.home.last.climb": {"en": "Climb", "fr": "Dénivelé"},
    "ui.home.last.time": {"en": "Moving time", "fr": "Temps en mouvement"},
    "ui.home.last.pace": {"en": "Pace", "fr": "Allure"},
    "ui.home.last.speed": {"en": "Speed", "fr": "Vitesse"},
    "ui.home.last.power": {"en": "Avg power", "fr": "Puissance moyenne"},
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

    # Session detail — comments
    "ui.session.comments.placeholder": {
        "en": "Add a comment…", "fr": "Ajouter un commentaire…",
    },
    "ui.session.comments.add": {"en": "Add", "fr": "Ajouter"},
    "ui.session.comments.save": {"en": "Save", "fr": "Enregistrer"},
    "ui.session.comments.cancel": {"en": "Cancel", "fr": "Annuler"},
    "ui.session.comments.edit": {"en": "Edit", "fr": "Modifier"},
    "ui.session.comments.delete": {"en": "Delete", "fr": "Supprimer"},
    "ui.home.progress.title": {"en": "Recent Progress", "fr": "Progrès récents"},
    "ui.home.form.title": {"en": "Recent Form", "fr": "Forme récente"},
    "ui.home.form.subtitle": {
        "en": "Fitness and fatigue over the last 30 weeks, from the Banister "
              "training-load model. Fitness builds and fades slowly; fatigue reacts "
              "to the last few days — the gap between them is what a hard week "
              "costs before it turns into fitness.",
        "fr": "Fitness et fatigue sur les 30 dernières semaines, d'après le modèle de "
              "charge d'entraînement de Banister. Le Fitness se construit et s'efface "
              "lentement ; la fatigue réagit aux derniers jours — l'écart entre les "
              "deux, c'est ce qu'une semaine dure coûte avant de se transformer en "
              "Fitness.",
    },
    "ui.home.feel.title": {"en": "Effort & Feel", "fr": "Effort et ressenti"},
    "ui.home.feel.subtitle": {
        "en": "Your last 12 weeks as you rated them: average RPE per week as the curve, "
              "the week's average feeling as its background colour, and the same "
              "Fitness tag the week summary shows. Read together: a hard week that "
              "felt strong and moved Fitness up is working; the same week felt weak "
              "with Fitness flat is not.",
        "fr": "Vos 12 dernières semaines telles que vous les avez notées : le RPE "
              "moyen en courbe, "
              "le ressenti moyen en couleur de fond, et le même tag de Fitness que "
              "le résumé de la semaine. À lire ensemble : une semaine dure, bien "
              "vécue et qui fait monter le Fitness, ça paie ; la même semaine mal "
              "vécue avec un Fitness plat, non.",
    },
    "ui.home.feel.empty": {
        "en": "Rate a few sessions — effort and how they felt — on the Training "
              "screen, and this chart fills in week by week.",
        "fr": "Notez quelques séances — effort et ressenti — dans l'écran "
              "Entraînement, et ce graphique se remplira semaine après semaine.",
    },
    "ui.home.efficiency.title": {"en": "Recent Efficiency", "fr": "Efficacité récente"},
    "ui.home.efficiency.subtitle": {
        "en": "Power per heartbeat, weekly, smoothed with a five-week rolling "
              "average and a Savitzky–Golay filter. It rises when the same effort "
              "buys you more pace — and unlike raw pace it does not care whether "
              "the week was hilly or flat.",
        "fr": "Puissance par battement, par semaine, lissée par une moyenne "
              "glissante de cinq semaines et un filtre de Savitzky–Golay. Elle "
              "monte quand le même effort vous rapporte plus d'allure — et "
              "contrairement à l'allure brute, elle ne dépend pas du relief de la "
              "semaine.",
    },
    "ui.home.efficiency.needs_weight": {
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
    "ui.home.trend.increasing": {"en": "Increasing", "fr": "En hausse"},
    "ui.home.trend.stable": {"en": "Stable", "fr": "Stable"},
    "ui.home.trend.decreasing": {"en": "Decreasing", "fr": "En baisse"},
    "ui.home.trend.short_term": {"en": "Recent", "fr": "Récent"},
    "ui.home.trend.long_term": {"en": "Sustained", "fr": "Durable"},

    # Home — importing from Strava
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
    "ui.training.week.running": {"en": "Run", "fr": "Course"},
    "ui.training.week.cycling": {"en": "Ride", "fr": "Vélo"},
    "ui.training.week.hiking": {"en": "Hike", "fr": "Rando"},
    "ui.training.week.swimming": {"en": "Swim", "fr": "Nage"},
    "ui.training.week.other": {"en": "Other", "fr": "Autres"},
    "ui.training.week.summary_title": {"en": "Week summary", "fr": "Résumé de la semaine"},
    "ui.training.week.fitness_label": {"en": "Fitness", "fr": "Fitness"},
    "ui.training.week.fitness_increasing": {
        "en": "Increasing Fitness", "fr": "Fitness en hausse",
    },
    "ui.training.week.fitness_stable": {
        "en": "Stable Fitness", "fr": "Fitness stable",
    },
    "ui.training.week.fitness_decreasing": {
        "en": "Decreasing Fitness", "fr": "Fitness en baisse",
    },
    "ui.training.session.rpe_short": {"en": "RPE", "fr": "RPE"},
    "ui.training.session.rpe_title": {
        "en": "Rate perceived exertion", "fr": "Effort ressenti (RPE)",
    },
    "ui.training.session.feeling_short": {"en": "Feel", "fr": "Ressenti"},
    "ui.training.session.feeling_title": {
        "en": "How did it feel?", "fr": "Comment s'est passée la séance ?",
    },
    "ui.training.session.feeling_faible": {"en": "Weak", "fr": "Faible"},
    "ui.training.session.feeling_ok": {"en": "OK", "fr": "Ok"},
    "ui.training.session.feeling_fort": {"en": "Strong", "fr": "Fort"},

    # --- Race plan ("Plan de course") -------------------------------------------
    "race_plan.series.elevation": {"en": "Elevation", "fr": "Altitude"},
    "race_plan.series.target_pace": {"en": "Target pace", "fr": "Allure cible"},
    "race_plan.series.gap_pace": {"en": "Constant GAP pace", "fr": "Allure GAP constante"},
    "race_plan.series.section_pace": {"en": "Section pace", "fr": "Allure par section"},
    "race_plan.series.leg_pace": {"en": "Leg pace", "fr": "Allure par tronçon"},
    "race_plan.series.aid_stations": {"en": "Aid stations", "fr": "Ravitaillements"},
    "race_plan.axis.elevation": {"en": "Elevation (m)", "fr": "Altitude (m)"},
    "race_plan.axis.pace": {"en": "Pace (min/km)", "fr": "Allure (min/km)"},
    "race_plan.axis.distance": {"en": "Distance (km)", "fr": "Distance (km)"},
    "race_plan.hover.elapsed": {"en": "elapsed", "fr": "écoulé"},
    "race_plan.hover.arrival": {"en": "arrival", "fr": "arrivée"},
    "race_plan.chart.profile": {
        "en": "Target pace along the course", "fr": "Allure cible sur le parcours",
    },
    "race_plan.chart.sections": {
        "en": "Pace per climb, descent and flat",
        "fr": "Allure par montée, descente et plat",
    },
    "race_plan.chart.aid_stations": {
        "en": "Pace between aid stations", "fr": "Allure entre ravitaillements",
    },
    "race_plan.caption.profile": {
        "en": "Holding a constant effort — a GAP pace of {gap} — lands exactly on the "
              "target time. Average pace over the course: {avg}.",
        "fr": "Tenir un effort constant — une allure GAP de {gap} — donne exactement le "
              "temps visé. Allure moyenne sur le parcours : {avg}.",
    },
    "race_plan.caption.sections": {
        "en": "Sections are detected from the smoothed profile: a climb or descent "
              "averages more than 3 % and changes elevation by at least 25 m; "
              "anything else is flat. Numbers match the table below.",
        "fr": "Les sections sont détectées sur le profil lissé : une montée ou une "
              "descente dépasse 3 % de pente moyenne et au moins 25 m de dénivelé ; "
              "le reste est du plat. Les numéros renvoient au tableau ci-dessous.",
    },
    "race_plan.caption.aid_stations": {
        "en": "Tags show the planned elapsed time at each aid station and at the finish.",
        "fr": "Les étiquettes indiquent le temps de course prévu à chaque ravitaillement "
              "et à l'arrivée.",
    },
    "race_plan.table.sections": {"en": "Sections", "fr": "Sections"},
    "race_plan.table.aid_stations": {"en": "Aid stations", "fr": "Ravitaillements"},
    "race_plan.section.climb": {"en": "Climb", "fr": "Montée"},
    "race_plan.section.descent": {"en": "Descent", "fr": "Descente"},
    "race_plan.section.flat": {"en": "Flat", "fr": "Plat"},
    "race_plan.col.type": {"en": "Type", "fr": "Type"},
    "race_plan.col.start_km": {"en": "From (km)", "fr": "Début (km)"},
    "race_plan.col.end_km": {"en": "To (km)", "fr": "Fin (km)"},
    "race_plan.col.distance": {"en": "Distance", "fr": "Distance"},
    "race_plan.col.grade": {"en": "Avg. grade", "fr": "Pente moy."},
    "race_plan.col.pace": {"en": "Avg. pace", "fr": "Allure moy."},
    "race_plan.col.duration": {"en": "Time", "fr": "Durée"},
    "race_plan.col.elapsed_end": {"en": "Elapsed at end", "fr": "Temps à la fin"},
    "race_plan.col.station": {"en": "Aid station", "fr": "Ravitaillement"},
    "race_plan.col.km": {"en": "km", "fr": "km"},
    "race_plan.col.leg_distance": {"en": "Leg distance", "fr": "Distance tronçon"},
    "race_plan.col.leg_time": {"en": "Leg time", "fr": "Temps tronçon"},
    "race_plan.col.arrival": {"en": "Arrival (elapsed)", "fr": "Arrivée (temps de course)"},
    "race_plan.col.clock": {"en": "Arrival (clock)", "fr": "Arrivée (heure)"},
    "race_plan.finish": {"en": "Finish", "fr": "Arrivée"},
    "race_plan.aid_station_n": {"en": "Aid station {n}", "fr": "Ravito {n}"},
    "race_plan.next_day": {"en": "(+{n}d)", "fr": "(+{n}j)"},
    "race_plan.curve.personal_efficiency": {
        "en": "My curve (efficiency model)", "fr": "Ma courbe (modèle d'efficacité)",
    },
    "race_plan.curve.personal_auto": {
        "en": "My curve (auto-learning model)", "fr": "Ma courbe (modèle auto-apprenant)",
    },
    "race_plan.note.ignored_stations": {
        "en": "Ignored aid stations outside the course (0–{total} km): {km}.",
        "fr": "Ravitaillements ignorés, hors du parcours (0–{total} km) : {km}.",
    },
    "race_plan.note.personal_fallback": {
        "en": "Your personal GAP curve could not be used ({reason}), so this plan uses "
              "the balanced-runner reference curve.",
        "fr": "Votre courbe GAP personnelle n'a pas pu être utilisée ({reason}) : ce plan "
              "utilise la courbe de référence du coureur équilibré.",
    },
    "race_plan.reason.no_runs": {
        "en": "no run with per-second data yet",
        "fr": "aucune course avec données détaillées pour l'instant",
    },
    "race_plan.reason.not_enough_data": {
        "en": "not enough runs with heart rate", "fr": "pas assez de sorties avec cardio",
    },
    "race_plan.reason.fit_failed": {
        "en": "the model could not be fitted", "fr": "le modèle n'a pas pu être ajusté",
    },
    "race_plan.error.gpx_invalid": {
        "en": "This file is not a readable GPX.", "fr": "Ce fichier n'est pas un GPX lisible.",
    },
    "race_plan.error.gpx_no_points": {
        "en": "This GPX has no track or route points.",
        "fr": "Ce GPX ne contient aucun point de trace ou d'itinéraire.",
    },
    "race_plan.error.gpx_no_elevation": {
        "en": "This GPX has no elevation data — export it with elevation to plan on it.",
        "fr": "Ce GPX n'a pas de données d'altitude — exportez-le avec l'altitude pour "
              "pouvoir le planifier.",
    },
    "race_plan.error.gpx_too_short": {
        "en": "This course is too short to plan.", "fr": "Ce parcours est trop court.",
    },
    "race_plan.error.gpx_too_large": {
        "en": "This GPX is too large (15 MB max).", "fr": "Ce GPX est trop lourd (15 Mo max).",
    },
    "race_plan.error.no_gpx": {"en": "Choose a GPX file.", "fr": "Choisissez un fichier GPX."},
    "race_plan.error.target_time": {
        "en": "Enter a target finish time.", "fr": "Indiquez un temps d'arrivée visé.",
    },

    "ui.nav.race_plan": {"en": "Race plan", "fr": "Plan de course"},
    # --- Durability (cost drift over a long effort) ---------------------------
    # "Durability", never "fatigue": fatigue is the Banister acute-load series.
    "race_plan.series.gap_pace_durability": {
        "en": "Effort-equivalent GAP pace (with durability)",
        "fr": "Allure GAP à effort égal (avec durabilité)",
    },
    "race_plan.series.durability_total": {"en": "Extra cost", "fr": "Surcoût"},
    "race_plan.chart.durability": {
        "en": "Durability — extra energetic cost along the course",
        "fr": "Durabilité — surcoût énergétique le long du parcours",
    },
    "race_plan.axis.durability": {"en": "Extra cost (%)", "fr": "Surcoût (%)"},
    "race_plan.caption.durability": {
        "en": "Running gets costlier as the race goes on: the same GAP speed costs +{pct} % "
              "more energy at the finish. At constant effort your GAP pace therefore drifts "
              "from {start} at the start to {finish} at the finish, for the same target time. "
              "Coloured areas split the extra cost into its causes.",
        "fr": "Courir coûte de plus en plus cher au fil de la course : la même vitesse GAP "
              "coûte +{pct} % d'énergie à l'arrivée. À effort constant, votre allure GAP "
              "passe donc de {start} au départ à {finish} à l'arrivée, pour le même temps "
              "visé. Les aires colorées répartissent ce surcoût par cause.",
    },
    "durability.component.duration": {"en": "Duration", "fr": "Durée"},
    "durability.component.severe_intensity": {
        "en": "Time above threshold", "fr": "Temps au-dessus du seuil",
    },
    "durability.component.downhill": {
        "en": "Downhill (muscle damage)", "fr": "Descente (dommages musculaires)",
    },
    "durability.component.thermal": {"en": "Heat", "fr": "Chaleur"},
    "durability.component.pre_race_load": {
        "en": "Pre-race load", "fr": "Charge avant course",
    },
    "durability.table.title": {"en": "Durability model", "fr": "Modèle de durabilité"},
    "durability.table.caption": {
        "en": "Coefficients multiply each exposure in log-cost. Personal share: how much of "
              "the applied value comes from your own runs rather than the population prior.",
        "fr": "Chaque coefficient multiplie son exposition, en log-coût. Part personnelle : "
              "la part de la valeur appliquée qui vient de vos sorties plutôt que de l'a "
              "priori population.",
    },
    "durability.col.component": {"en": "Cause", "fr": "Cause"},
    "durability.col.unit": {"en": "Unit", "fr": "Unité"},
    "durability.col.population": {"en": "Population", "fr": "Population"},
    "durability.col.applied": {"en": "Applied", "fr": "Appliqué"},
    "durability.col.weight": {"en": "Personal share", "fr": "Part personnelle"},
    "durability.col.posterior_sd": {"en": "Uncertainty (±1σ)", "fr": "Incertitude (±1σ)"},
    "durability.col.exposure": {"en": "Exposure at finish", "fr": "Exposition à l'arrivée"},
    "durability.col.contribution": {"en": "Extra cost at finish", "fr": "Surcoût à l'arrivée"},
    "durability.confidence.population_only": {
        "en": "Durability: population model (not personalized).",
        "fr": "Durabilité : modèle population (non personnalisé).",
    },
    "durability.confidence.partially_personalized": {
        "en": "Durability: partially personalized from {runs} long runs ({hours} h) of the "
              "past year — still close to the population prior.",
        "fr": "Durabilité : partiellement personnalisée à partir de {runs} sorties longues "
              "({hours} h) de l'année écoulée — encore proche de l'a priori population.",
    },
    "durability.confidence.personalized": {
        "en": "Durability: personalized from {runs} long runs ({hours} h) of the past year.",
        "fr": "Durabilité : personnalisée à partir de {runs} sorties longues ({hours} h) de "
              "l'année écoulée.",
    },
    "durability.reason.no_history": {
        "en": "No usable long run with heart rate in the past year.",
        "fr": "Aucune sortie longue exploitable avec cardio sur l'année écoulée.",
    },
    "durability.reason.no_reference_speed": {
        "en": "No best effort in the past year to set your reference speed.",
        "fr": "Aucun meilleur effort sur l'année écoulée pour fixer votre vitesse de référence.",
    },
    "durability.reason.too_few_runs": {
        "en": "Too few usable long runs to personalize.",
        "fr": "Trop peu de sorties longues exploitables pour personnaliser.",
    },
    "durability.reason.weak_evidence": {
        "en": "Your runs do not yet separate your durability clearly from the population's, "
              "so the prior still carries most of the weight.",
        "fr": "Vos sorties ne distinguent pas encore clairement votre durabilité de celle de "
              "la population : l'a priori garde l'essentiel du poids.",
    },
    "durability.reason.clipped_at_zero": {
        "en": "A coefficient fitted below zero was set to zero (cost cannot fall with effort).",
        "fr": "Un coefficient ajusté sous zéro a été ramené à zéro (le coût ne peut pas "
              "baisser avec l'effort).",
    },
    "durability.reason.signed_out": {
        "en": "Sign in to personalize durability from your own runs.",
        "fr": "Connectez-vous pour personnaliser la durabilité à partir de vos sorties.",
    },
    "durability.reason.fit_failed": {
        "en": "Your durability model could not be fitted; the population model is used.",
        "fr": "Votre modèle de durabilité n'a pas pu être ajusté : le modèle population est "
              "utilisé.",
    },
    "durability.note.placeholder": {
        "en": "Population coefficients are conservative product defaults ({version}), not "
              "validated individual physiology.",
        "fr": "Les coefficients population sont des valeurs produit prudentes ({version}), "
              "pas une physiologie individuelle validée.",
    },
    "durability.note.target_reference": {
        "en": "Intensity is inferred from the target time itself (treated as a full race "
              "effort), not from your best efforts.",
        "fr": "L'intensité est déduite du temps visé lui-même (considéré comme un effort de "
              "course complet), pas de vos meilleurs efforts.",
    },
    "durability.note.clamped": {
        "en": "The extra cost reached the model's safety ceiling and was capped.",
        "fr": "Le surcoût a atteint le plafond de sécurité du modèle et a été plafonné.",
    },
    "durability.note.not_converged": {
        "en": "The pacing did not fully converge; the last iterate is shown.",
        "fr": "Le calcul d'allure n'a pas complètement convergé : la dernière itération est "
              "affichée.",
    },
    "durability.note.fresh_fallback": {
        "en": "Durability could not be applied numerically; this plan ignores it.",
        "fr": "La durabilité n'a pas pu être appliquée numériquement : ce plan l'ignore.",
    },
    "durability.note.past_year": {
        "en": "Only runs from the last {days} days are used.",
        "fr": "Seules les sorties des {days} derniers jours sont utilisées.",
    },
    "durability.note.reference": {
        "en": "Reference (critical) speed: {pace} GAP, from your best {distance} of the year.",
        "fr": "Vitesse de référence (critique) : {pace} GAP, d'après votre meilleur {distance} "
              "de l'année.",
    },
    "durability.note.reference_gap": {
        "en": "Reference (critical) speed: {pace} GAP, from your best gradient-adjusted "
              "{distance} of the year.",
        "fr": "Vitesse de référence (critique) : {pace} GAP, d'après votre meilleur {distance} "
              "ajusté à la pente de l'année.",
    },
    "durability.note.outliers": {
        "en": "{count} best effort(s) ignored as implausible next to your other distances "
              "(GPS glitch, tunnel…): {distances}.",
        "fr": "{count} meilleur(s) effort(s) ignoré(s) car invraisemblable(s) au regard de vos "
              "autres distances (erreur GPS, tunnel…) : {distances}.",
    },
    "durability.note.excluded": {
        "en": "Runs left out — {details}.", "fr": "Sorties écartées — {details}.",
    },
    "durability.excluded.sport": {"en": "treadmill / other sport", "fr": "tapis / autre sport"},
    "durability.excluded.too_short": {"en": "too short", "fr": "trop courtes"},
    "durability.excluded.no_heart_rate": {"en": "no heart rate", "fr": "sans cardio"},
    "durability.excluded.poor_elevation": {"en": "poor elevation", "fr": "altitude incomplète"},
    "durability.excluded.few_valid_segments": {
        "en": "too few steady stretches", "fr": "trop peu de portions régulières",
    },
    "durability.excluded.intermittent": {
        "en": "intermittent (intervals)", "fr": "fractionnées",
    },
    "durability.excluded.no_stream": {"en": "no detailed data", "fr": "sans données détaillées"},
    "durability.nothing": {
        "en": "No long run in the selection.", "fr": "Aucune sortie longue dans la sélection.",
    },
    "durability.series.observed": {"en": "Observed (median, IQR)", "fr": "Observé (médiane, IQR)"},
    "durability.series.population": {"en": "Population model", "fr": "Modèle population"},
    "durability.series.personal": {"en": "Your model", "fr": "Votre modèle"},
    "durability.chart.drift": {
        "en": "Cost drift within your long runs", "fr": "Dérive du coût pendant vos sorties longues",
    },
    "durability.chart.projection": {
        "en": "Projected durability", "fr": "Durabilité projetée",
    },
    "durability.axis.elapsed": {"en": "Elapsed time (h)", "fr": "Temps écoulé (h)"},
    "durability.axis.drift": {
        "en": "Cost drift since the start (%)", "fr": "Dérive du coût depuis le début (%)",
    },
    "durability.axis.extra_cost": {"en": "Extra cost (%)", "fr": "Surcoût (%)"},
    "durability.caption.drift": {
        "en": "Each steady 5-minute stretch of your long runs: heart-rate reserve per unit of "
              "GAP speed, relative to the start of that run, after removing typical cardiac "
              "drift. Lines are what each model predicts for the same stretches.",
        "fr": "Chaque portion régulière de 5 minutes de vos sorties longues : réserve "
              "cardiaque par unité de vitesse GAP, relative au début de la sortie, une fois "
              "retirée la dérive cardiaque typique. Les courbes sont ce que prédit chaque "
              "modèle pour ces mêmes portions.",
    },
    "durability.caption.projection": {
        "en": "Extra cost of a steady, flat run at your typical long-run intensity "
              "({intensity} of critical speed), neutral weather. The band is ±1σ on your "
              "duration coefficient.",
        "fr": "Surcoût d'une sortie régulière sur le plat à votre intensité habituelle de "
              "sortie longue ({intensity} de la vitesse critique), météo neutre. La bande "
              "représente ±1σ sur votre coefficient de durée.",
    },
    "plot.durability_curve.label": {"en": "Durability curve", "fr": "Courbe de durabilité"},
    "plot.durability_curve.description": {
        "en": "How your running cost drifts over long efforts, fitted on your past year of "
              "long runs — the model the race plan paces with.",
        "fr": "Comment votre coût de course dérive sur les efforts longs, ajusté sur vos "
              "sorties longues de l'année écoulée — le modèle utilisé par le plan de course.",
    },
    "param.durability.lookback": {"en": "History (days)", "fr": "Historique (jours)"},
    "param.durability.lookback.help": {
        "en": "At most the past year, whatever the data source selects.",
        "fr": "Au plus l'année écoulée, quelle que soit la source de données.",
    },
    "param.durability.min_run": {
        "en": "Minimum run length (min)", "fr": "Durée minimale de sortie (min)",
    },
    "param.durability.bin": {"en": "Time bins (min)", "fr": "Pas de temps (min)"},
    "param.durability.show_observed": {
        "en": "Show observed drift", "fr": "Afficher la dérive observée",
    },
    "page.durability.title": {"en": "Durability", "fr": "Durabilité"},
    "durability.intro": {
        "en": "How much more running costs you after two, three, five hours — measured on "
              "your long runs of the past year, compared with a typical runner.",
        "fr": "Combien la course vous coûte en plus après deux, trois, cinq heures — mesuré "
              "sur vos sorties longues de l'année écoulée, comparé à un coureur typique.",
    },
    "dash.durability.window": {"en": "Past 12 months", "fr": "12 derniers mois"},
    "dash.durability.panel.method": {"en": "What is measured", "fr": "Ce qui est mesuré"},
    "dash.durability.text.what": {
        "en": "Durability is how well you hold your efficiency as an effort gets long. Late "
              "in a long race the same pace costs more energy: that extra cost is what this "
              "analysis estimates.",
        "fr": "La durabilité, c'est votre capacité à garder votre efficacité quand l'effort "
              "s'allonge. Tard dans une longue course, la même allure coûte plus d'énergie : "
              "c'est ce surcoût que cette analyse estime.",
    },
    "dash.durability.text.how": {
        "en": "Only your runs of the past year are used. Each steady 5-minute stretch of a "
              "long run compares your heart-rate reserve with your gradient-adjusted speed; "
              "typical cardiac drift is removed first, because a rising heart rate alone is "
              "not a rising cost. Intervals, treadmill runs, pauses and stretches without "
              "heart rate are left out. The result starts from a population model and moves "
              "toward yours only as far as your data support.",
        "fr": "Seules vos sorties de l'année écoulée sont utilisées. Chaque portion régulière "
              "de 5 minutes d'une sortie longue compare votre réserve cardiaque à votre "
              "vitesse ajustée à la pente ; la dérive cardiaque typique est retirée d'abord, "
              "car un cardio qui monte ne signifie pas à lui seul un coût qui monte. Les "
              "fractionnés, le tapis, les pauses et les portions sans cardio sont écartés. "
              "Le résultat part d'un modèle population et ne s'en éloigne qu'autant que vos "
              "données le justifient.",
    },
    "dash.durability.panel.long_runs": {"en": "Your long runs", "fr": "Vos sorties longues"},
    "dash.durability.panel.long_runs.help": {
        "en": "Durability is only visible on long runs: the longer and more regular they "
              "are, the more personal the curve below.",
        "fr": "La durabilité ne se voit que sur les sorties longues : plus elles sont longues "
              "et régulières, plus la courbe ci-dessous est personnelle.",
    },
    "dash.durability.panel.curve": {"en": "Your durability", "fr": "Votre durabilité"},
    "dash.durability.panel.curve.help": {
        "en": "The same model the race plan uses to pace your long races.",
        "fr": "Le même modèle que celui qu'utilise le plan de course pour vos longues courses.",
    },
    "ui.race_plan.conditions": {"en": "Durability & conditions", "fr": "Durabilité et conditions"},
    "ui.race_plan.durability": {
        "en": "Account for durability (pace drifts as the race gets long)",
        "fr": "Tenir compte de la durabilité (l'allure dérive quand la course s'allonge)",
    },
    "ui.race_plan.temperature_start": {"en": "Start temperature (°C)", "fr": "Température au départ (°C)"},
    "ui.race_plan.temperature_end": {"en": "Finish temperature (°C)", "fr": "Température à l'arrivée (°C)"},
    "ui.race_plan.humidity": {"en": "Relative humidity (%)", "fr": "Humidité relative (%)"},
    "ui.race_plan.weather_help": {
        "en": "Optional. Heat and humidity add cost only above mild conditions.",
        "fr": "Optionnel. Chaleur et humidité n'ajoutent un coût qu'au-delà de conditions "
              "douces.",
    },
    "ui.race_plan.error.weather": {
        "en": "Temperatures must be numbers between −40 and 55 °C, humidity between 0 and 100 %.",
        "fr": "Les températures doivent être entre −40 et 55 °C, l'humidité entre 0 et 100 %.",
    },
    "ui.race_plan.summary.durability_finish": {
        "en": "Extra cost at finish", "fr": "Surcoût à l'arrivée",
    },
    "ui.race_plan.summary.gap_finish": {"en": "GAP pace at finish", "fr": "Allure GAP à l'arrivée"},
    "ui.race_plan.summary.durability_model": {"en": "Durability model", "fr": "Modèle de durabilité"},
    "ui.race_plan.confidence.population_only": {"en": "Population", "fr": "Population"},
    "ui.race_plan.confidence.partially_personalized": {
        "en": "Partly personal", "fr": "Partiellement personnel",
    },
    "ui.race_plan.confidence.personalized": {"en": "Personal", "fr": "Personnel"},
    "ui.race_plan.section.durability": {"en": "Durability", "fr": "Durabilité"},
    "ui.race_plan.title": {"en": "Race plan", "fr": "Plan de course"},
    "ui.race_plan.intro": {
        "en": "Upload the course GPX, list the aid stations and set your target time: "
              "you get the pace to hold at every point of the course — the one "
              "constant effort, adjusted for the gradient, that lands exactly on your time.",
        "fr": "Importez le GPX du parcours, indiquez les ravitaillements et votre temps "
              "visé : vous obtenez l'allure à tenir en chaque point du parcours — l'effort "
              "constant, ajusté à la pente, qui donne exactement votre temps.",
    },
    "ui.race_plan.gpx": {"en": "Course GPX", "fr": "GPX du parcours"},
    "ui.race_plan.target_time": {"en": "Target finish time", "fr": "Temps d'arrivée visé"},
    "ui.race_plan.target_time_help": {
        "en": "h:mm or h:mm:ss", "fr": "h:mm ou h:mm:ss",
    },
    "ui.race_plan.start_time": {
        "en": "Start time (optional)", "fr": "Heure de départ (optionnel)",
    },
    "ui.race_plan.start_time_help": {
        "en": "Adds the time of day at each aid station.",
        "fr": "Ajoute l'heure de passage à chaque ravitaillement.",
    },
    "ui.race_plan.aid_stations": {"en": "Aid stations", "fr": "Ravitaillements"},
    "ui.race_plan.aid_station_km": {"en": "km", "fr": "km"},
    "ui.race_plan.aid_station_name": {"en": "Name (optional)", "fr": "Nom (optionnel)"},
    "ui.race_plan.add_aid_station": {"en": "+ Add an aid station", "fr": "+ Ajouter un ravito"},
    "ui.race_plan.remove": {"en": "Remove", "fr": "Retirer"},
    "ui.race_plan.curve": {"en": "GAP curve", "fr": "Courbe GAP"},
    "ui.race_plan.curve_sign_in": {"en": "sign in", "fr": "connexion requise"},
    "ui.race_plan.submit": {"en": "Compute the plan", "fr": "Calculer le plan"},
    "ui.race_plan.computing": {"en": "Computing…", "fr": "Calcul en cours…"},
    "ui.race_plan.computing_personal": {
        "en": "Fitting your GAP curve on your runs — the first plan can take a minute…",
        "fr": "Ajustement de votre courbe GAP sur vos sorties — le premier plan peut "
              "prendre une minute…",
    },
    "ui.race_plan.error.no_gpx": {"en": "Choose a GPX file.", "fr": "Choisissez un fichier GPX."},
    "ui.race_plan.error.target_time": {
        "en": "Enter the target time as h:mm or h:mm:ss.",
        "fr": "Indiquez le temps visé au format h:mm ou h:mm:ss.",
    },
    "ui.race_plan.error.start_time": {
        "en": "Enter the start time as hh:mm.", "fr": "Indiquez l'heure de départ au format hh:mm.",
    },
    "ui.race_plan.new.button": {"en": "New race plan", "fr": "Nouveau plan de course"},
    "ui.race_plan.new.hint": {
        "en": "A GPX, the aid stations, a target time",
        "fr": "Un GPX, les ravitos, un temps visé",
    },
    "ui.race_plan.empty": {
        "en": "No saved plan yet.", "fr": "Aucun plan enregistré pour l'instant.",
    },
    "ui.race_plan.plan_title": {"en": "Plan title", "fr": "Titre du plan"},
    "ui.race_plan.title_placeholder": {
        "en": "e.g. UTMB 2026", "fr": "ex. UTMB 2026",
    },
    "ui.race_plan.untitled": {"en": "Untitled plan", "fr": "Plan sans titre"},
    "ui.race_plan.save": {"en": "Save", "fr": "Enregistrer"},
    "ui.race_plan.saving": {"en": "Saving…", "fr": "Enregistrement…"},
    "ui.race_plan.saved": {"en": "Saved", "fr": "Enregistré"},
    "ui.race_plan.delete": {"en": "Delete", "fr": "Supprimer"},
    "ui.race_plan.delete_confirm": {
        "en": "Delete “{title}”? This cannot be undone.",
        "fr": "Supprimer « {title} » ? Cette action est définitive.",
    },
    "ui.race_plan.back": {"en": "← My race plans", "fr": "← Mes plans de course"},
    "ui.race_plan.gpx_current": {"en": "Current file: {name}", "fr": "Fichier actuel : {name}"},
    "ui.race_plan.gpx_replace": {
        "en": "Choose another file to replace it.",
        "fr": "Choisissez un autre fichier pour le remplacer.",
    },
    "ui.race_plan.updated": {"en": "Updated {date}", "fr": "Modifié le {date}"},
    "ui.race_plan.summary.distance": {"en": "Distance", "fr": "Distance"},
    "ui.race_plan.summary.elevation": {"en": "Elevation", "fr": "Dénivelé"},
    "ui.race_plan.summary.target": {"en": "Target time", "fr": "Temps visé"},
    "ui.race_plan.summary.gap_pace": {"en": "Constant GAP pace", "fr": "Allure GAP constante"},
    "ui.race_plan.summary.avg_pace": {"en": "Average pace", "fr": "Allure moyenne"},
    "ui.race_plan.summary.curve": {"en": "Curve used", "fr": "Courbe utilisée"},
    "ui.race_plan.section.profile": {"en": "Pace profile", "fr": "Profil d'allure"},
    "ui.race_plan.section.sections": {
        "en": "By climb, descent and flat", "fr": "Par montée, descente et plat",
    },
    "ui.race_plan.section.aid_stations": {
        "en": "Between aid stations", "fr": "Entre ravitaillements",
    },

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
