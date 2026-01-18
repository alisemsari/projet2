import requests # Bibliothèque pour envoyer des requêtes HTTP (communiquer avec l'API)
import pandas as pd

# On prépare le "headers" pour accéder à l'API de TMDB.

headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI2YjhhZTkxODE5NDczZWFjMDMyMzQ0YTI5ZGM2MDk1YyIsIm5iZiI6MTc2NjQ4MzU0Mi4xMzIsInN1YiI6IjY5NGE2NjU2ZWE5OWU0ZGZkYjhlYzI2ZSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.K1g9VO9XjrLEmfRkIShVqtbIY7kORtJjEXTmfEMRLwY",
    "accept": "application/json"
}

tous_les_films = []

print("⏳ Téléchargement de 10 000 films en cours... Patientez quelques minutes.")

# L'API TMDB limite les résultats à 20 films par page. 
# Pour avoir 10 000 films, on doit parcourir 500 pages (500 * 20 = 10 000)
for i in range(1, 501):
    try:
        # On demande tous les films populaires (toutes langues confondues) depuis 1996
        url = f'https://api.themoviedb.org/3/discover/movie?include_adult=false&language=fr-FR&page={i}&sort_by=popularity.desc&vote_count.gte=50&primary_release_date.gte=1996-01-01'
        # Envoi de la requête GET et conversion immédiate de la réponse brute en dictionnaire Python (.json())
        reponse = requests.get(url, headers=headers).json()
        # Extraction de la liste de films présente dans la clé 'results'. 
        # Si 'results' n'existe pas, on renvoie une liste vide [] pour éviter que le code ne plante.
        films_page = reponse.get('results', [])
        # 'films_page' est une LISTE contenant les 20 films de la page actuelle.
        for f in films_page:
            # On cherche les acteurs pour chaque film
            
            # 'movie_id' Elle contient l'identifiant unique du film que l'on est en train d'analyser.
            movie_id = f['id']
            # On utilise 'movie_id' pour construire l'adresse exacte du casting du film.
            url_cast = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?language=fr-FR"
            data_cast = requests.get(url_cast, headers=headers).json()
            cast = data_cast.get('cast', [])
            
            # Boucle simple pour prendre les 3 premiers noms
            noms_acteurs = []
            compteur = 0
            for acteur in cast:
                if compteur < 3:
                    noms_acteurs.append(acteur['name'])
                    compteur += 1
            # DÉCLARATION D'UNE NOUVELLE VARIABLE DANS L'OBJET FILM :
            # 'f' est le dictionnaire original du film. Ici, on crée une nouvelle "colonne"
            # nommée 'acteurs' à l'intérieur de 'f' pour y stocker les noms trouvés.
            # Cela enrichit la base de données avant de la sauvegarder.
            f['acteurs'] = ", ".join(noms_acteurs)
            tous_les_films.append(f)
            
        if i % 50 == 0:
            print(f"✅ {len(tous_les_films)} films récupérés...")
            
    except:
        continue

# Sauvegarde finale
df = pd.DataFrame(tous_les_films)
df.to_csv("ma_base_films.csv", index=False)
print("✨ Terminé ! Le fichier 'ma_base_films.csv' est prêt.")