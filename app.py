import streamlit as st
import pandas as pd
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from streamlit_authenticator import Authenticate
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
import os


# --- CONFIGURATION DE LA PAGE ---
# Cette commande doit obligatoirement être la PREMIÈRE ligne Streamlit du script.
# page_title : Définit le nom dans l'onglet du navigateur.
# layout="wide" : Force l'affichage à utiliser toute la largeur de l'écran.
st.set_page_config(page_title="Cinéma Recommandation", layout="wide")

# . GESTION DE L'AUTHENTIFICATION (LOGIN) ---
# On vérifie si le fichier "exo.csv" existe physiquement sur l'ordinateur
if os.path.exists("exo.csv"):
    # Si le fichier existe, on le charge dans un tableau (DataFrame) avec Pandas
    # Cela permet de récupérer les comptes utilisateurs déjà créés précédemment
    df_users = pd.read_csv("exo.csv")
    # : INITIALISATION DE LA STRUCTURE ---
    # On crée un dictionnaire vide avec une clé racine 'usernames'. 
    # C'est le format EXACT exigé par la bibliothèque d'authentification.
    lesDonneesDesComptes = {'usernames': {}}
    # .iterrows() permet de parcourir le tableau Pandas (df_users) ligne par ligne.
    # index : le numéro de la ligne / row : le contenu de la ligne.
    for index, row in df_users.iterrows():
        # On utilise le nom d'utilisateur (colonne 'username') comme "clé" pour identifier la fiche.
        # Pour chaque utilisateur, on crée un sous-dictionnaire contenant ses infos personnelles.
        lesDonneesDesComptes['usernames'][row['username']] = {
            'name': row['name'],
            'password': str(row['password']),
            'email': row['email'],
            # int(...) convertit le nombre d'essais ratés en nombre entier pour permettre les calculs plus tard.
            'failed_login_attemps': int(row['failed_login_attemps']),
            # Indique si l'utilisateur est déjà connecté.
            'logged_in': row['logged_in'],
            # Indique les droits (ex: 'admin' ou 'user').
            'role': row['role']
        }
    # On configure le module Authenticate avec :
    #  Les données transformées (lesDonneesDesComptes).
    #  Le nom et la clé du cookie (pour que l'utilisateur n'ait pas à se reconnecter à chaque clic).
    #  La durée de validité (30 jours).
    authenticator = Authenticate(lesDonneesDesComptes, "cookie_name", "cookie_key", 30)
    # Cette ligne génère l'interface graphique (les champs de saisie) sur le site.
    # Elle récupère les entrées en temps réel de l'utilisateur (Identifiant / Mot de passe).
    # Elle compare INSTANTANÉMENT ces entrées avec les données issues du CSV.
    # Elle gère seule le succès ou l'échec (création de session ou message d'erreur).
    #Elle initialise la variable 'authentication_status' dans st.session_state.
    authenticator.login()

    # Si l'utilisateur est bien connecté
    if st.session_state.get("authentication_status"):
        
        # ---  CHARGEMENT DE LA BASE DE DONNÉES FILMS ---
        if os.path.exists("ma_base_films.csv"):
            df_complet = pd.read_csv("ma_base_films.csv")
        else:
            st.error("⚠️ Fichier 'ma_base_films.csv' introuvable !")
            st.stop()

        # --- 4. BARRE LATÉRALE (SIDEBAR) ---
        with st.sidebar:
            authenticator.logout('Déconnexion', 'sidebar')
            # Cette ligne sert uniquement à afficher un message visuel en vert.
            st.sidebar.success(f"Bienvenue : {st.session_state['name']}")
            st.write("---")
            
            # Cases à cocher et réglages pour filtrer les films
            st.write("### 🔍 Configuration")
            filtre_fr = st.checkbox("🇫🇷 Uniquement films français")
            note_min = st.slider("⭐ Note minimale", 0.0, 10.0, 5.0)
            
            st.write("---")
            # Menu pour changer de page
            selected = option_menu(
                menu_title="Menu Principal",
                options=["Recommandation", "Stats: Genres", "Stats: Langues", "Stats: Popularité"],
                icons=["film", "tags", "translate", "graph-up-arrow"],
                menu_icon="cast",
                default_index=0,
            )

        #  FILTRAGE DU DATAFRAME 
        # On garde les films avec la note minimum choisie
        df_final = df_complet[df_complet['vote_average'] >= note_min].copy()
        # Si la case "Français" est cochée, on filtre par langue
        if filtre_fr and 'original_language' in df_final.columns:
            df_final = df_final[df_final['original_language'] == 'fr']
        df_final = df_final.reset_index(drop=True)

        # --- AFFICHAGE DU NOMBRE DE FILMS DANS LA SIDEBAR ---
        with st.sidebar:
            st.write("---")
            # Cette ligne affiche un bloc statistique (métrique).
            # Elle compte automatiquement le nombre de lignes dans le tableau 'df_final'
            # et l'affiche sous le titre "Nombre de films disponibles".
            st.metric(label="Nombre de films disponibles", value=len(df_final))

        # --- PAGE RECOMMANDATION ---
        if selected == "Recommandation":
            
            # Fonction pour enlever les accents et mettre en minuscule
            # --- PRÉPARATION DU TEXTE POUR LA RECHERCHE ---
            # Cette fonction transforme n'importe quel texte en version "simplifiée" :
            # Enlève les accents (é -> e).
            # Met tout en minuscules (A -> a).
            # Supprime les espaces inutiles.
            # Cela permet de trouver un film même si l'utilisateur fait des fautes d'accents.
            def nettoyer(texte):
                if not isinstance(texte, str): return ""
                texte_propre = "".join(c for c in unicodedata.normalize('NFD', texte) if unicodedata.category(c) != 'Mn')
                return texte_propre.lower().strip()

            # Préparation des données pour le moteur de recommandation(c'est un modèle non supervisé)
            # On crée une colonne 'soup' qui regroupe les mots-clés importants.
            # On multiplie les genres (*3) et les acteurs (*2) pour leur donner plus de "poids".
            # Si un film est un "Action", ce mot sera plus important que les mots de la description.
            df_final['titre_clean'] = df_final['title'].apply(nettoyer)
            df_final['soup'] = (
                (df_final['genre'].fillna('') + " ") * 3 + 
                (df_final['acteurs'].fillna('') + " ") * 2 + 
                df_final['overview'].fillna('')
            ).apply(nettoyer)
            # : VECTORISATION (TF-IDF) ---
            # Le modèle transforme le texte en chiffres (vecteurs).
            # TF-IDF réduit l'importance des mots trop communs (comme "le", "un") 
            # et augmente l'importance des mots uniques qui définissent le film

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(df_final['soup'])
            # CALCUL DE SIMILARITÉ (COSINE SIMILARITY) ---
            # On calcule la "distance" entre chaque film. 
            # Le résultat est une matrice où chaque film a un score de ressemblance avec tous les autres.
            sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

            st.write(f"## 🍿 Bonjour {st.session_state['name']}, voici des suggestions au hasard")
            
            # Affichage de 3 films au hasard au début
            # --- AFFICHAGE ALÉATOIRE (VITRINE) ---
            # SÉLECTION : On vérifie qu'il y a assez de films, puis on en pioche 3 au hasard (.sample).
            # PRÉPARATION : On définit l'adresse racine (URL) pour récupérer les images de TMDB.
            # MISE EN PAGE : On crée 3 colonnes côte à côte pour un affichage propre.
            # BOUCLE : Pour chaque colonne, on affiche l'affiche, le titre et la note du film choisi.
            if len(df_final) >= 3:
                df_random = df_final.sample(n=3)
                URL_IMG = "https://image.tmdb.org/t/p/w500"
                cols_rdm = st.columns(3)
                for i in range(3):
                    with cols_rdm[i]:
                        st.image(URL_IMG + str(df_random.iloc[i]['poster_path']), width=150)
                        st.write(f"**{df_random.iloc[i]['title']}**")
                        st.write(f"⭐ Note : {df_random.iloc[i]['vote_average']}")

            # Barre de recherche pour l'utilisateur
            recherche = st.chat_input("Entrez le titre d'un film que vous avez aimé")
            # Cette condition vérifie si la variable 'recherche' n'est pas vide.
            # Si l'utilisateur n'a rien tapé, Python ignore tout le bloc suivant.
            if recherche:
                # On utilise notre fonction 'nettoyer' sur ce que l'utilisateur a écrit.
                #  transformer "BatmAn" ou "Batman " en "batman" pour éviter les erreurs.
                rech_propre = nettoyer(recherche)
                # On regarde si 'rech_propre' existe dans la colonne 'titre_clean' du CSV.
                # .values transforme la colonne en une liste de mots pour la recherche.
                if rech_propre in df_final['titre_clean'].values:
                    # Trouver l'index du film cherché
                    #On cherche le numéro de la ligne (l'index) où se trouve le film.
                    #.index[0] récupère le numéro exact de la première ligne correspondante.
                    idx = df_final[df_final['titre_clean'] == rech_propre].index[0]
                    # .iloc[idx] va chercher TOUTES les colonnes de cette ligne précise.
                    # 'film_choisi' devient un objet contenant le titre, l'image et le résumé du film.
                    film_choisi = df_final.iloc[idx]
                    
                    st.write(f"### ✅ Vous avez choisi : **{film_choisi['title']}**")
                    
                    # Encadré bleu avec les infos du film choisi (Inclus les ACTEURS)
                    st.markdown(f"""
                    <div style="background-color: #1E3A8A; padding: 25px; border-radius: 15px; border-left: 10px solid #3B82F6; margin: 20px 0;">
                        <p style="color: white; font-size: 20px; margin: 0;">
                            🎬 Genre : <b>{film_choisi['genre']}</b> | 👥 Acteurs : <b>{film_choisi['acteurs']}</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Calcul des scores de ressemblance
                    # On va chercher dans la matrice de similarité la ligne 'idx' (le film choisi).
                    # sim_matrix[idx] contient les scores de ressemblance entre ce film et TOUS les autres.
                    # enumerate() ajoute l'index (0, 1, 2...) devant chaque score pour savoir à quel film il appartient.
                    # list() transforme le tout en une liste de couples : [(index, score), (index, score)...].
                    scores = list(enumerate(sim_matrix[idx]))
                    # On range la liste pour mettre les films les plus ressemblants en premier.
                    # key=lambda x: x[1] : dit à Python de trier selon le 'score' (le 2ème élément du couple).
                    # reverse=True : trie par ordre décroissant (du plus gros score au plus petit).
                    scores = sorted(scores, key=lambda x: x[1], reverse=True)
                    
                    st.subheader('Voici les 5 films recommandés pour vous :')
                    cols_res = st.columns(5)
                    
                    j = 0
                    for i in range(1, 6): # On prend les 5 premiers films après le film lui-même
                        film_reco = df_final.iloc[scores[i][0]]
                        
                        with cols_res[j]:
                            st.image(URL_IMG + str(film_reco['poster_path']))
                            st.write(f"### {film_reco['title']}")
                            st.write(f"🎭 le genre est : {film_reco['genre']}")
                            st.write(f"⭐ note est {film_reco['vote_average']}")
                            st.write(f"📅 date de sortie est {film_reco['release_date']}")
                            st.write(f"👥 le nombre de vote est {film_reco['vote_count']}")
                            st.write(f"🎭 Acteurs : {film_reco['acteurs']}") 
                            j += 1
                else:
                    st.error(f"Le film '{recherche}' n'est pas disponible avec vos filtres actuels.")

        
        elif selected == "Stats: Genres":
            st.title("🏷️ Analyse des Genres")
            # On sépare les genres et on compte
            genres_series = df_final['genre'].str.split(',').explode().str.strip()
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            sns.countplot(y=genres_series, palette="viridis", ax=ax1, order=genres_series.value_counts().index)
            ax1.bar_label(ax1.containers[0], padding=3)
            st.pyplot(fig1)
            st.info("Cette page montre quels genres sont les plus présents dans ta base actuelle.")

        elif selected == "Stats: Langues":
            st.title("🌐 Répartition par Langue")
            lang_data = df_final['original_language'].value_counts().head(10)
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.pie(lang_data, labels=lang_data.index, autopct='%1.1f%%', startangle=140)
            st.pyplot(fig2)
            st.info("Ici on voit la diversité des langues originales des films.")

        elif selected == "Stats: Popularité":
            st.title("📈 Popularité vs Qualité")
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df_final, x='vote_count', y='vote_average', alpha=0.4, color='teal', ax=ax3)
            if len(df_final) > 0 and df_final['vote_count'].min() > 0:
                ax3.set_xscale('log') # Échelle log pour mieux voir les données
            st.pyplot(fig3)
            st.info("Ce graphique montre si les films très populaires sont mieux notés que les autres.")

    elif st.session_state.get("authentication_status") is False:
        st.error('Utilisateur ou mot de passe incorrect')
    elif st.session_state.get("authentication_status") is None:
        st.warning('Veuillez vous connecter')
else:
    st.error("⚠️ Fichier 'exo.csv' introuvable !")