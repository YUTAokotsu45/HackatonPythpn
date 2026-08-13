import flask
import pandas as pd
import model as mdl


app = flask.Flask(__name__)

app.secret_key = "HelloWorld_1234509"


weight = None

# ============================================================
# INDEX
# ============================================================

@app.route("/", methods=["GET"])
def index():

    return flask.render_template(
        "index.html",
        message=""
    )


# ============================================================
# FILE
# ============================================================

@app.route("/file", methods=["GET", "POST"])
def upload():

    if flask.request.method == "POST":

        # ----------------------------------------------------
        # Récupération du fichier et des paramètres
        # ----------------------------------------------------

        fichier = flask.request.files.get(
            "fichier"
        )

        target = flask.request.form.get(
            "target",
            ""
        )

        type_problem = flask.request.form.get(
            "type_problem",
            ""
        )

        # ----------------------------------------------------
        # Vérification du fichier
        # ----------------------------------------------------

        if not fichier or fichier.filename == "":

            return flask.render_template(
                "index.html",
                message="Aucun fichier sélectionné"
            )

        # ----------------------------------------------------
        # Lecture du fichier
        # ----------------------------------------------------

        extension = fichier.filename[
            fichier.filename.rfind("."):
        ].lower()

        if extension == ".csv":

            df = pd.read_csv(
                fichier
            )

        elif extension == ".xlsx":

            df = pd.read_excel(
                fichier
            )

        else:

            return flask.render_template(
                "index.html",
                message="Format de fichier non supporté"
            )

        # ----------------------------------------------------
        # Nettoyage automatique du Dataset
        # ----------------------------------------------------

        # 1. Supprime les colonnes de type Datetime (évite l'erreur NumPy)
        df = df.select_dtypes(
            exclude=['datetime', 'datetime64']
        )

        # 2. Remplit les valeurs manquantes (évite d'avoir 0 ligne pour IsolationForest)
        df = df.ffill().bfill().fillna(0)

        # ----------------------------------------------------
        # Vérification et ajustement de la target
        # ----------------------------------------------------

        if target not in df.columns:

            return flask.render_template(
                "index.html",
                message=f"La colonne '{target}' n'existe pas dans le fichier."
            )

        # Si la target contient des valeurs continues/décimales, forcer la Régression
        if pd.api.types.is_float_dtype(df[target]):
            type_problem = "Régression"

        # ----------------------------------------------------
        # Récupération de toutes les features
        # ----------------------------------------------------

        df_train = df.drop(
            columns=target
        )

        features = df_train.columns.tolist()

        # ----------------------------------------------------
        # Sauvegarde du dataset
        # ----------------------------------------------------

        df.to_pickle(
            "temp_data.pkl"
        )

        # ----------------------------------------------------
        # ENTRAÎNEMENT À BLANC POUR TROUVER LES MEILLEURES COLONNES
        # ----------------------------------------------------

        temp_eptus = mdl.Eptus(df, target, features)
        temp_algo = mdl.CelerPlex(temp_eptus.df, target, features)

        if type_problem in ["Régression", "Regression", "regression"]:
            temp_algo.Regress()
        else:
            temp_algo.Classify()

        best_dummy_cols = temp_algo.best_cols

        global best_features

        best_features = []
        for orig_col in features:
            if orig_col in best_dummy_cols:
                if orig_col not in best_features:
                    best_features.append(orig_col)
            else:
                for dummy_col in best_dummy_cols:
                    if str(dummy_col).startswith(str(orig_col) + "_"):
                        if orig_col not in best_features:
                            best_features.append(orig_col)
                        break

        # ----------------------------------------------------
        # Sauvegarde des informations de session
        # ----------------------------------------------------

        flask.session["features"] = features
        flask.session["target"] = target
        flask.session["type_problem"] = type_problem

        # ----------------------------------------------------
        # Page de sélection des features (AFFICHAGE FILTRÉ)
        # ----------------------------------------------------

        return flask.render_template(
            "file.html",
            features=best_features
        )

    return flask.render_template(
        "index.html",
        message=""
    )

@app.route("/predict", methods=["POST"])
def predict():

    type_problem = flask.request.form.get(
        "type_problem",
        flask.session.get("type_problem", "")
    ).strip()

    features = flask.session.get(
        "features",
        []
    )

    target = flask.session.get(
        "target"
    )

    df = pd.read_pickle(
        "temp_data.pkl"
    )

    if target in df.columns and pd.api.types.is_float_dtype(df[target]):
        type_problem = "Régression"

    valeurs = []

    for feature in features:

        val_brute = flask.request.form.get(feature, "").strip()

        # Si la colonne était affichée ET remplie par l'utilisateur
        if feature in flask.request.form and val_brute != "":
            try:
                valeurs.append(float(val_brute))
            except ValueError:
                valeurs.append(val_brute)

        # Si la colonne était masquée OU laissée vide par l'utilisateur
        else:
            if pd.api.types.is_numeric_dtype(df[feature]):
                valeurs.append(0.0)
            else:
                s_clean = df[feature].dropna()
                valeurs.append(str(s_clean.iloc[0]) if not s_clean.empty else "inconnu")

    df_train = df.drop(
        columns=target
    )

    model = mdl.Eptus(
        df,
        target,
        df_train.columns.tolist()
    )

    if type_problem in ["Régression", "Regression", "regression"]:

        prediction, acc, used_model = model.Regress(
            valeurs
        )

    elif type_problem in ["Classification", "classification"]:

        prediction, acc, used_model = model.Classify(
            valeurs
        )

    else:

        return f"Type de problème invalide (reçu : '{type_problem}')"


    list_of = model.DictWeight(used_model, best_features)

    return flask.render_template(
        "predict.html",
        prediction=prediction,
        acc =acc * 100, 
        list_of = list_of
    )


    
if __name__ == "__main__":

    app.run(
        debug=True
    )