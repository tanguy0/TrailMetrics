"""Central UI translation table for TrailMetrics.

One source of truth for every user-facing string, in English (``en``) and French
(``fr``). Pure Python with no Streamlit / framework dependency, so both the
Streamlit app (``app/``) and the domain plot helpers (``src/domain/...``) can
import it.

Usage:
    from src.translations import translate
    translate("home.connect.header", "fr")

The app reads the active language from ``st.session_state["lang"]`` (set by the
selector on the Home page) through the ``t()`` helper in ``app/_helpers.py``;
domain plot functions receive an explicit ``lang`` argument.

Strings may contain ``{placeholder}`` fields — format them at the call site,
e.g. ``translate("home.status.loaded", lang).format(count=12)``.
"""

LANGUAGES = {"fr": "Français", "en": "English"}
DEFAULT_LANG = "fr"


# key -> {"en": ..., "fr": ...}
TRANSLATIONS = {
    # --- Common --------------------------------------------------------------
    "common.you": {"en": "You", "fr": "Vous"},
    "common.lang_label": {"en": "Language / Langue", "fr": "Langue / Language"},
    "gate.no_data": {
        "en": "No data loaded yet. Go to the **Home** page, connect Strava and "
        "click **Load my data** to unlock this analysis.",
        "fr": "Aucune donnée chargée pour l'instant. Allez sur la page "
        "**Accueil**, connectez-vous à Strava et cliquez sur **Charger mes "
        "données** pour débloquer cette analyse.",
    },

    # --- Page titles (browser tab + h1) -------------------------------------
    "page.howitworks.tab": {"en": "How it works", "fr": "Comment ça marche"},
    "page.howitworks.title": {
        "en": "❓ How TrailMetrics works",
        "fr": "❓ Comment fonctionne TrailMetrics",
    },
    "page.gap.title": {
        "en": "Personalized GAP Simulator",
        "fr": "Simulateur GAP personnalisé",
    },
    "page.races.title": {"en": "Race Comparator", "fr": "Comparateur de courses"},

    # --- Home ----------------------------------------------------------------
    "home.subheader": {
        "en": "Connect Strava, load your history, then explore your running data",
        "fr": "Connectez Strava, chargez votre historique, puis explorez vos "
        "données de course",
    },
    "home.intro": {
        "en": """
    Set up the connection below and load your activity history **once**. After
    that, open any analysis from the **left sidebar** — they all reuse this data,
    so you can tweak parameters freely without re-fetching.

    New here? See **How it works** in the sidebar.
    """,
        "fr": """
    Configurez la connexion ci-dessous et chargez votre historique d'activités
    **une seule fois**. Ensuite, ouvrez n'importe quelle analyse depuis la
    **barre latérale gauche** — elles réutilisent toutes ces données, vous pouvez
    donc ajuster les paramètres librement sans nouvelle récupération.

    Nouveau ici ? Consultez **Comment ça marche** dans la barre latérale.
    """,
    },
    "home.connect.header": {"en": "1. Connect Strava", "fr": "1. Connexion à Strava"},
    "home.client_id.help": {
        "en": "Your Strava application client ID.",
        "fr": "L'identifiant client de votre application Strava.",
    },
    "home.token_exchange_failed": {
        "en": "Token exchange failed: {error}",
        "fr": "Échec de l'échange du jeton : {error}",
    },
    "home.missing_creds": {
        "en": "Missing credentials — re-enter STRAVA_CLIENT_ID / SECRET above.",
        "fr": "Identifiants manquants — saisissez à nouveau STRAVA_CLIENT_ID / "
        "SECRET ci-dessus.",
    },
    "home.connected": {"en": "✅ Connected to Strava.", "fr": "✅ Connecté à Strava."},
    "home.connect_button": {
        "en": "Connect with Strava",
        "fr": "Se connecter avec Strava",
    },
    "home.enter_creds_info": {
        "en": "Enter your STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET above to connect.",
        "fr": "Saisissez vos STRAVA_CLIENT_ID et STRAVA_CLIENT_SECRET ci-dessus "
        "pour vous connecter.",
    },
    "home.runner.header": {"en": "2. Runner", "fr": "2. Coureur"},
    "home.runner.name_label": {"en": "Your name", "fr": "Votre nom"},
    "home.runner.weight_label": {"en": "Your weight (kg)", "fr": "Votre poids (kg)"},
    "home.runner.weight_help": {
        "en": "Used to estimate running power on the Race Comparator. Leave empty "
        "to skip.",
        "fr": "Sert à estimer la puissance de course dans le Comparateur de "
        "courses. Laissez vide pour ignorer.",
    },
    "home.load.header": {"en": "3. Load your data", "fr": "3. Charger vos données"},
    "home.load.caption": {
        "en": "Fetches every run (all types) from the date below up to today. This "
        "can take a while the first time — it only runs once per session.",
        "fr": "Récupère toutes les courses (tous types) depuis la date ci-dessous "
        "jusqu'à aujourd'hui. Cela peut prendre un moment la première fois — "
        "exécuté une seule fois par session.",
    },
    "home.load.from_label": {
        "en": "Fetch data back to",
        "fr": "Récupérer les données jusqu'au",
    },
    "home.load.from_help": {
        "en": "The oldest date to fetch initially. Older activities are ignored.",
        "fr": "La date la plus ancienne à récupérer initialement. Les activités "
        "plus anciennes sont ignorées.",
    },
    "home.load.button": {"en": "Load my data", "fr": "Charger mes données"},
    "home.load.scouting": {
        "en": "Scouting your activities on Strava…",
        "fr": "Repérage de vos activités sur Strava…",
    },
    "home.load.fetching": {
        "en": "Fetching activity {done} of {total}…",
        "fr": "Récupération de l'activité {done} sur {total}…",
    },
    "home.status.loaded": {
        "en": "✅ Loaded {count} activities — analyses are unlocked.",
        "fr": "✅ {count} activités chargées — les analyses sont débloquées.",
    },
    "home.metric.activities": {"en": "Activities", "fr": "Activités"},
    "home.metric.oldest": {"en": "Oldest session", "fr": "Séance la plus ancienne"},
    "home.metric.newest": {"en": "Most recent session", "fr": "Séance la plus récente"},
    "home.status.open_analysis": {
        "en": "Open an analysis from the **left sidebar** to get started.",
        "fr": "Ouvrez une analyse depuis la **barre latérale gauche** pour "
        "commencer.",
    },
    "home.status.no_data": {
        "en": "No data loaded yet. Complete the steps above to unlock the analyses.",
        "fr": "Aucune donnée chargée pour l'instant. Complétez les étapes "
        "ci-dessus pour débloquer les analyses.",
    },

    # --- How it works --------------------------------------------------------
    "howitworks.body": {
        "en": """
    TrailMetrics is a personal lab for running-data simulations. The flow is
    designed around **loading your data once** and then exploring it freely.

    ### 1. Connect & load (Home page)
    1. Provide your Strava `client_id` / `client_secret`, then click
       **Connect with Strava** — you'll be redirected to authorize and brought
       straight back with a token (no codes to copy). Credentials are remembered
       for next time.
    2. Enter your name and weight.
    3. Click **Load my data**. TrailMetrics fetches the maximum history
       available (all run types) and keeps it in memory for the whole session.

    Once loaded, you'll see how many activities were fetched and the date range
    they span (oldest → most recent session).

    ### 2. Run an analysis (sidebar pages)
    Each analysis lives on its own page and unlocks only after data is loaded.
    All of its inputs — **date range**, **session types**, model parameters,
    intensity ranges — are *filters applied to the data already in memory*. That
    means you can tweak any parameter and re-run instantly, without re-fetching
    from Strava.

    - The selectable date range is bounded by your fetched history
      (oldest session → today).

    ### Available analyses
    - **Personalized GAP Simulator** — builds personalized GAP (Gradient
      Adjusted Pace) curves from your history and compares them against the
      *Balanced Runner* and *Kilian Jornet* reference curves, including
      intensity-stratified panels. Every figure has a **Download data** button
      (PNG + CSV bundled in a ZIP).

    ### Notes
    - Your Strava token lives only in this browser session and is never stored
      on disk.
    - Closing the app clears the loaded data; you'll load it again next time.
    """,
        "fr": """
    TrailMetrics est un labo personnel de simulations sur vos données de course.
    Le principe : **charger vos données une seule fois**, puis les explorer
    librement.

    ### 1. Connexion & chargement (page Accueil)
    1. Renseignez vos `client_id` / `client_secret` Strava, puis cliquez sur
       **Se connecter avec Strava** — vous serez redirigé pour autoriser puis
       ramené directement avec un jeton (aucun code à copier). Les identifiants
       sont mémorisés pour la prochaine fois.
    2. Saisissez votre nom et votre poids.
    3. Cliquez sur **Charger mes données**. TrailMetrics récupère le maximum
       d'historique disponible (tous types de course) et le garde en mémoire
       pendant toute la session.

    Une fois chargé, vous verrez le nombre d'activités récupérées et la plage de
    dates qu'elles couvrent (séance la plus ancienne → la plus récente).

    ### 2. Lancer une analyse (pages de la barre latérale)
    Chaque analyse a sa propre page et ne se débloque qu'une fois les données
    chargées. Toutes ses entrées — **plage de dates**, **types de séance**,
    paramètres de modèle, plages d'intensité — sont des *filtres appliqués aux
    données déjà en mémoire*. Vous pouvez donc ajuster n'importe quel paramètre
    et relancer instantanément, sans nouvelle récupération depuis Strava.

    - La plage de dates sélectionnable est bornée par votre historique récupéré
      (séance la plus ancienne → aujourd'hui).

    ### Analyses disponibles
    - **Simulateur GAP personnalisé** — construit des courbes GAP (allure
      ajustée à la pente) personnalisées à partir de votre historique et les
      compare aux courbes de référence *Coureur équilibré* et *Kilian Jornet*,
      avec des panneaux par intensité. Chaque graphique dispose d'un bouton
      **Télécharger les données** (PNG + CSV regroupés dans un ZIP).

    ### Remarques
    - Votre jeton Strava ne vit que dans cette session de navigateur et n'est
      jamais stocké sur le disque.
    - Fermer l'application efface les données chargées ; vous les rechargerez la
      prochaine fois.
    """,
    },

    # --- GAP simulator -------------------------------------------------------
    "gap.intro": {
        "en": """
    Build personalized GAP (Gradient Adjusted Pace) curves from your loaded
    history and compare them against reference curves. Define one or more
    **time scales** (named date ranges), choose which models and reference
    curves to plot in the sidebar, then re-run — no re-fetching required.
    """,
        "fr": """
    Construisez des courbes GAP (allure ajustée à la pente) personnalisées à
    partir de votre historique et comparez-les à des courbes de référence.
    Définissez une ou plusieurs **échelles de temps** (plages de dates nommées),
    choisissez les modèles et les courbes de référence à afficher dans la barre
    latérale, puis relancez — aucune nouvelle récupération nécessaire.
    """,
    },
    "gap.filters.header": {"en": "Filters", "fr": "Filtres"},
    "gap.session_types": {"en": "Session types", "fr": "Types de séance"},
    "gap.timescales.header": {"en": "Time scales", "fr": "Échelles de temps"},
    "gap.timescales.caption": {
        "en": "Compare up to five time scales — each is a named date range. At "
        "least one named scale is required; every figure overlays them.",
        "fr": "Comparez jusqu'à cinq échelles de temps — chacune est une plage de "
        "dates nommée. Au moins une échelle nommée est requise ; chaque graphique "
        "les superpose.",
    },
    "gap.timescales.name_label": {"en": "Name", "fr": "Nom"},
    "gap.timescales.name_default": {"en": "Scale {n}", "fr": "Échelle {n}"},
    "gap.timescales.name_placeholder": {
        "en": "e.g. 2025 season",
        "fr": "ex. saison 2025",
    },
    "gap.timescales.remove_help": {
        "en": "Remove this time scale",
        "fr": "Supprimer cette échelle de temps",
    },
    "gap.timescales.from": {"en": "From", "fr": "Du"},
    "gap.timescales.to": {"en": "To", "fr": "Au"},
    "gap.timescales.add": {
        "en": "➕ Add time scale",
        "fr": "➕ Ajouter une échelle de temps",
    },
    "gap.models.header": {"en": "GAP models", "fr": "Modèles GAP"},
    "gap.models.caption": {
        "en": "Pick at least one model to plot.",
        "fr": "Choisissez au moins un modèle à afficher.",
    },
    "gap.models.efficiency": {"en": "Efficiency model", "fr": "Modèle d'efficacité"},
    "gap.models.auto": {"en": "Auto-Learning model", "fr": "Modèle auto-apprenant"},
    "gap.refs.header": {"en": "Reference curves", "fr": "Courbes de référence"},
    "gap.refs.caption": {
        "en": "Optional overlays — leave both off to hide them.",
        "fr": "Superpositions optionnelles — décochez les deux pour les masquer.",
    },
    "gap.refs.balanced": {"en": "Balanced runner", "fr": "Coureur équilibré"},
    "gap.refs.kilian": {"en": "Kilian curve", "fr": "Courbe Kilian"},
    "gap.display.header": {"en": "Display options", "fr": "Options d'affichage"},
    "gap.display.show_std": {
        "en": "Show standard deviation bands",
        "fr": "Afficher les bandes d'écart-type",
    },
    "gap.display.show_std_help": {
        "en": "Shade ±1 std around each curve. Turn off for a cleaner overlay.",
        "fr": "Ombre ±1 écart-type autour de chaque courbe. Décochez pour une "
        "superposition plus épurée.",
    },
    "gap.params.header": {"en": "Simulation parameters", "fr": "Paramètres de simulation"},
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
    "gap.intensity.header": {"en": "Intensity ranges", "fr": "Plages d'intensité"},
    "gap.intensity.low_min": {"en": "Low intensity: min HR", "fr": "Intensité basse : FC min"},
    "gap.intensity.low_max": {"en": "Low intensity: max HR", "fr": "Intensité basse : FC max"},
    "gap.intensity.high_min": {"en": "High intensity: min HR", "fr": "Intensité haute : FC min"},
    "gap.intensity.high_max": {"en": "High intensity: max HR", "fr": "Intensité haute : FC max"},
    "gap.intensity.low": {"en": "Low", "fr": "Basse"},
    "gap.intensity.high": {"en": "High", "fr": "Haute"},
    "gap.validate.no_sport": {
        "en": "Select at least one session type.",
        "fr": "Sélectionnez au moins un type de séance.",
    },
    "gap.validate.no_scale": {
        "en": "Name at least one time scale.",
        "fr": "Nommez au moins une échelle de temps.",
    },
    "gap.validate.inverted": {
        "en": "Some time scales have From after To — fix the dates.",
        "fr": "Certaines échelles ont une date « Du » postérieure au « Au » — "
        "corrigez les dates.",
    },
    "gap.validate.no_model": {
        "en": "Select at least one GAP model.",
        "fr": "Sélectionnez au moins un modèle GAP.",
    },
    "gap.run_button": {"en": "Run simulation", "fr": "Lancer la simulation"},
    "gap.loader": {
        "en": "Filtering activities and fitting models…",
        "fr": "Filtrage des activités et ajustement des modèles…",
    },
    "gap.summary.item": {
        "en": "{label}: {n} splits",
        "fr": "{label} : {n} segments",
    },
    "gap.summary": {
        "en": "Simulation complete — {summary}.",
        "fr": "Simulation terminée — {summary}.",
    },
    "gap.subheader.curves": {"en": "GAP curves", "fr": "Courbes GAP"},
    "gap.nothing_to_plot": {
        "en": "Nothing to plot — select a model or a reference curve in the sidebar.",
        "fr": "Rien à tracer — sélectionnez un modèle ou une courbe de référence "
        "dans la barre latérale.",
    },
    "gap.caption.main": {
        "en": "Color = time scale · solid = Efficiency model · dash-dot = "
        "Auto-Learning model · dashed = reference curves.",
        "fr": "Couleur = échelle de temps · trait plein = modèle d'efficacité · "
        "trait mixte = modèle auto-apprenant · tirets = courbes de référence.",
    },
    "gap.subheader.intensity": {
        "en": "Intensity-stratified GAP curves",
        "fr": "Courbes GAP par intensité",
    },
    "gap.caption.intensity": {
        "en": "Color = time scale · solid = low intensity · dashed = high intensity.",
        "fr": "Couleur = échelle de temps · trait plein = intensité basse · "
        "tirets = intensité haute.",
    },
    "gap.col.efficiency_heading": {
        "en": "{runner} (Efficiency model)",
        "fr": "{runner} (Modèle d'efficacité)",
    },
    "gap.col.auto_heading": {
        "en": "{runner} (Auto-Learning model)",
        "fr": "{runner} (Modèle auto-apprenant)",
    },
    "gap.intensity.unavailable": {
        "en": "{label} {intensity}-intensity curve unavailable: {error}",
        "fr": "Courbe d'intensité {intensity} pour {label} indisponible : {error}",
    },
    "gap.run_hint": {
        "en": "Define your time scales, choose models and reference curves in the "
        "sidebar, then click 'Run simulation'.",
        "fr": "Définissez vos échelles de temps, choisissez les modèles et les "
        "courbes de référence dans la barre latérale, puis cliquez sur "
        "« Lancer la simulation ».",
    },

    # --- Race comparator -----------------------------------------------------
    "races.intro": {
        "en": """
    Compare any number of races side by side — from a single workout up to
    **{max}** at once. Start by picking your workouts below. Use the date
    search to narrow things down; each option shows its duration, distance and
    sport type so you can tell similar sessions apart.
    """,
        "fr": """
    Comparez autant de courses que vous voulez côte à côte — d'une seule séance
    jusqu'à **{max}** à la fois. Commencez par choisir vos séances ci-dessous.
    Utilisez la recherche par date pour affiner ; chaque option affiche sa
    durée, sa distance et son type de sport pour distinguer les séances
    similaires.
    """,
    },
    "races.select.subheader": {"en": "Select workouts", "fr": "Sélection des séances"},
    "races.search_by_date": {"en": "Search by date", "fr": "Rechercher par date"},
    "races.search_by_date_help": {
        "en": "Narrow the workout list to a single day.",
        "fr": "Restreindre la liste des séances à un seul jour.",
    },
    "races.workout_date": {"en": "Workout date", "fr": "Date de la séance"},
    "races.matches_caption": {
        "en": "{n} workout(s) on {date}:",
        "fr": "{n} séance(s) le {date} :",
    },
    "races.unknown_date": {"en": "unknown date", "fr": "date inconnue"},
    "races.no_workouts_on_date": {
        "en": "No workouts on that date.",
        "fr": "Aucune séance à cette date.",
    },
    "races.no_dated": {
        "en": "No dated workouts in the loaded history.",
        "fr": "Aucune séance datée dans l'historique chargé.",
    },
    "races.multiselect.label": {
        "en": "Workouts to compare (up to {max})",
        "fr": "Séances à comparer (jusqu'à {max})",
    },
    "races.multiselect.help": {
        "en": "Pick between 1 and {max} workouts.",
        "fr": "Choisissez entre 1 et {max} séances.",
    },
    "races.selected.subheader": {
        "en": "Selected ({n}/{max})",
        "fr": "Sélection ({n}/{max})",
    },
    "races.pick_one": {
        "en": "Pick at least one workout above to get started.",
        "fr": "Choisissez au moins une séance ci-dessus pour commencer.",
    },
    "races.col.date": {"en": "Date", "fr": "Date"},
    "races.col.sport": {"en": "Sport", "fr": "Sport"},
    "races.col.distance": {"en": "Distance", "fr": "Distance"},
    "races.col.duration": {"en": "Duration", "fr": "Durée"},
    "races.smoothing.expander": {
        "en": "⚙️ Smoothing settings",
        "fr": "⚙️ Paramètres de lissage",
    },
    "races.smoothing.caption": {
        "en": "Each signal can pass through a time-domain rolling average and/or a "
        "distance-domain Savitzky–Golay filter (applied in that order). Altitude "
        "smoothing drives the gradient and elevation gain.",
        "fr": "Chaque signal peut passer par une moyenne glissante (domaine "
        "temporel) et/ou un filtre Savitzky–Golay (domaine distance), appliqués "
        "dans cet ordre. Le lissage de l'altitude pilote la pente et le dénivelé.",
    },
    "races.filter.rolling": {"en": "Rolling avg", "fr": "Moyenne glissante"},
    "races.filter.window_s": {"en": "Window (s)", "fr": "Fenêtre (s)"},
    "races.filter.savgol": {"en": "Savitzky–Golay", "fr": "Savitzky–Golay"},
    "races.filter.window_m": {"en": "Window (m)", "fr": "Fenêtre (m)"},
    "races.signal.pace": {"en": "Pace / GAP", "fr": "Allure / GAP"},
    "races.signal.altitude": {"en": "Altitude", "fr": "Altitude"},
    "races.signal.hr": {"en": "Heart rate", "fr": "Fréquence cardiaque"},
    "races.signal.power": {"en": "Power", "fr": "Puissance"},
    "races.analysis.header": {"en": "Analysis", "fr": "Analyse"},
    "races.summary.subheader": {"en": "Summary stats", "fr": "Statistiques récapitulatives"},
    "races.col.elev_gain": {"en": "Elevation gain", "fr": "Dénivelé"},
    "races.col.time": {"en": "Time", "fr": "Temps"},
    "races.col.avg_pace": {"en": "Avg pace", "fr": "Allure moy."},
    "races.col.avg_gap_pace": {"en": "Avg GAP pace", "fr": "Allure GAP moy."},
    "races.col.avg_power": {"en": "Avg power", "fr": "Puissance moy."},
    "races.power_blank": {
        "en": "Avg power is blank — set **your weight** on the Home page to enable "
        "power (and the power graphs below).",
        "fr": "La puissance moy. est vide — renseignez **votre poids** sur la page "
        "Accueil pour activer la puissance (et les graphiques de puissance "
        "ci-dessous).",
    },
    "races.evolution.subheader": {
        "en": "Evolution across the race",
        "fr": "Évolution sur la course",
    },
    "races.xaxis.label": {"en": "X axis", "fr": "Axe X"},
    "races.xaxis.time": {"en": "Time", "fr": "Temps"},
    "races.xaxis.distance": {"en": "Distance", "fr": "Distance"},
    "races.xaxis.help": {
        "en": "Switch every graph below between elapsed time and distance covered.",
        "fr": "Basculer tous les graphiques ci-dessous entre le temps écoulé et "
        "la distance parcourue.",
    },
    "races.gap_display.label": {"en": "Show GAP as", "fr": "Afficher le GAP en"},
    "races.gap_display.pace": {"en": "Pace", "fr": "Allure"},
    "races.gap_display.speed": {"en": "Speed", "fr": "Vitesse"},
    "races.gap_display.help": {
        "en": "Display the gradient-adjusted-pace graph as pace (min/km) or speed "
        "(km/h).",
        "fr": "Afficher le graphique d'allure ajustée à la pente en allure "
        "(min/km) ou en vitesse (km/h).",
    },
    "races.plot.gap_pace": {
        "en": "Gradient-adjusted pace",
        "fr": "Allure ajustée à la pente",
    },
    "races.plot.power": {"en": "Power", "fr": "Puissance"},
    "races.plot.heartrate": {"en": "Heart rate", "fr": "Fréquence cardiaque"},
    "races.plot.power_to_hr": {"en": "Power-to-HR", "fr": "Puissance / FC"},
    "races.weight_needed": {
        "en": "Set **your weight** on the Home page to enable this graph.",
        "fr": "Renseignez **votre poids** sur la page Accueil pour activer ce "
        "graphique.",
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
    "plot.races.gap_pace.title": {
        "en": "Gradient-Adjusted Pace",
        "fr": "Allure ajustée à la pente",
    },
    "plot.races.gap_speed.y": {
        "en": "GAP speed (km/h, higher = faster)",
        "fr": "Vitesse GAP (km/h, plus haut = plus rapide)",
    },
    "plot.races.power.y": {"en": "Power (W)", "fr": "Puissance (W)"},
    "plot.races.power.title": {"en": "Power", "fr": "Puissance"},
    "plot.races.hr.y": {"en": "Heart rate (bpm)", "fr": "Fréquence cardiaque (bpm)"},
    "plot.races.hr.title": {"en": "Heart Rate", "fr": "Fréquence cardiaque"},
    "plot.races.p2hr.y": {"en": "Power / HR (W/bpm)", "fr": "Puissance / FC (W/bpm)"},
    "plot.races.p2hr.title": {
        "en": "Power-to-Heart-Rate",
        "fr": "Puissance / Fréquence cardiaque",
    },
    "plot.races.x.time": {"en": "Time (min)", "fr": "Temps (min)"},
    "plot.races.x.distance": {"en": "Distance (km)", "fr": "Distance (km)"},
    "plot.races.title_suffix": {"en": "across the race", "fr": "sur la course"},

    # --- Long-Term Progress page --------------------------------------------
    "page.ltp.title": {
        "en": "📈 Long-Term Progress",
        "fr": "📈 Progression long terme",
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
    "ltp.computing": {
        "en": "Crunching your whole history (best efforts + gradients)…",
        "fr": "Analyse de tout votre historique (meilleurs efforts + pentes)…",
    },
    "ltp.no_data": {
        "en": "No dated activities to analyse in the loaded history.",
        "fr": "Aucune activité datée à analyser dans l'historique chargé.",
    },
    "ltp.col.season": {"en": "Season", "fr": "Saison"},
    "ltp.bin_label": {"en": "Aggregate by", "fr": "Agréger par"},
    "ltp.bin.week": {"en": "Week", "fr": "Semaine"},
    "ltp.bin.month": {"en": "Month", "fr": "Mois"},

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
    "ltp.records.metric_label": {"en": "Show as", "fr": "Afficher en"},
    "ltp.records.metric.pace": {"en": "Pace (min/km)", "fr": "Allure (min/km)"},
    "ltp.records.metric.time": {"en": "Time", "fr": "Temps"},
    "ltp.records.col.distance": {"en": "Distance", "fr": "Distance"},
    "ltp.records.col.record": {"en": "Record", "fr": "Record"},
    "ltp.records.col.pace": {"en": "Pace", "fr": "Allure"},
    "ltp.records.col.date": {"en": "Date", "fr": "Date"},
    "ltp.records.none": {
        "en": "No records yet — no activity is long enough for these distances.",
        "fr": "Aucun record — aucune activité n'est assez longue pour ces distances.",
    },

    # Section 2 — Annual mileage
    "ltp.section.mileage": {
        "en": "Evolution of annual mileage",
        "fr": "Évolution du kilométrage annuel",
    },
    "ltp.mileage.col.total": {"en": "Total", "fr": "Total"},

    # Section 3 — Annual elevation gain
    "ltp.section.elevation": {
        "en": "Evolution of annual elevation gain",
        "fr": "Évolution du dénivelé annuel",
    },
    "ltp.elevation.col.total": {"en": "Total D+", "fr": "Total D+"},

    # Section 4 — Average gradient per season
    "ltp.section.gradient": {
        "en": "Evolution of average gradient per season",
        "fr": "Évolution de la pente moyenne par saison",
    },
    "ltp.section.gradient.help": {
        "en": "Average gradient over each bin = total elevation gain ÷ total "
        "distance for that bin (always positive). One line per season — click a "
        "season in the legend to show or hide it.",
        "fr": "Pente moyenne sur chaque période = dénivelé total ÷ distance totale "
        "de la période (toujours positive). Une ligne par saison — cliquez sur une "
        "saison dans la légende pour l'afficher ou la masquer.",
    },
    "ltp.gradient.col.avg": {"en": "Season avg", "fr": "Moy. saison"},

    # Section 5 — Gradient map
    "ltp.section.gradient_map": {"en": "Gradient map", "fr": "Carte des pentes"},
    "ltp.gradient_map.help": {
        "en": "Share of moving time spent in each gradient band, per bin. Each bar "
        "sums to 100%. Click a band in the legend to show or hide it.",
        "fr": "Part du temps en mouvement passé dans chaque catégorie de pente, par "
        "période. Chaque barre totalise 100 %. Cliquez sur une catégorie dans la "
        "légende pour l'afficher ou la masquer.",
    },
    "ltp.gradient_map.range_label": {"en": "Time span", "fr": "Période"},
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
    "plot.ltp.x.month": {"en": "Month", "fr": "Mois"},
    "plot.ltp.power_hr.title": {
        "en": "Power-to-HR efficiency over time",
        "fr": "Efficacité puissance / FC au fil du temps",
    },
    "plot.ltp.power_hr.x": {"en": "Time", "fr": "Temps"},
    "plot.ltp.power_hr.y": {"en": "Power / HR (W/bpm)", "fr": "Puissance / FC (W/bpm)"},
    "plot.ltp.records.title": {
        "en": "Personal-record evolution",
        "fr": "Évolution des records personnels",
    },
    "plot.ltp.records.x": {"en": "Date", "fr": "Date"},
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
    "plot.ltp.mileage.title": {
        "en": "Cumulative distance by season",
        "fr": "Distance cumulée par saison",
    },
    "plot.ltp.mileage.y": {
        "en": "Cumulative distance (km)", "fr": "Distance cumulée (km)",
    },
    "plot.ltp.elevation.title": {
        "en": "Cumulative elevation gain by season",
        "fr": "Dénivelé cumulé par saison",
    },
    "plot.ltp.elevation.y": {
        "en": "Cumulative elevation gain (m)", "fr": "Dénivelé cumulé (m)",
    },
    "plot.ltp.gradient.title": {
        "en": "Average gradient by season",
        "fr": "Pente moyenne par saison",
    },
    "plot.ltp.gradient.y": {"en": "Average gradient (%)", "fr": "Pente moyenne (%)"},
    "plot.ltp.gradient_map.title": {
        "en": "Time spent per gradient band",
        "fr": "Temps passé par catégorie de pente",
    },
    "plot.ltp.gradient_map.x": {"en": "Time", "fr": "Temps"},
    "plot.ltp.gradient_map.y": {"en": "% of moving time", "fr": "% du temps en mouvement"},
}


def translate(key: str, lang: str = DEFAULT_LANG) -> str:
    """Return the ``lang`` string for ``key``.

    Falls back to English, then to the raw key, so a missing translation degrades
    gracefully instead of raising.
    """
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(lang) or entry.get("en") or key
