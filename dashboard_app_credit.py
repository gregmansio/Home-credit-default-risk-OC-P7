# Une boite de sélection principale pour choisir un client dans une liste de plein de clients (+ potentiellement un bouton qui va chercher automatiquement un client au hasard...
# ... parmi l'ensemble de l'échantillon, OU parmi le sous-échantillon filtré grâce à des box sur le côté) 
# Sélection de quelques variables sous forme de filtres pour indentifier plusieurs sous-groupes d'utilisateurs (cf point du dessus): âge, sexe, tranche revenu.
# Grâce au point 1, on obtient déjà: 
#    - Un tableau (qui montre les infos de ce client sur telle et telle variable - on va en sélectionner entre 5 et 10) 
#    - Un graphique sur la répartition d'une de ces variables importantes
#    - Un graphique sur la répartition d'un ratio de 2 variables
#    - Une prédiction (score + traduction "Accordé/Non-accordé") accompagnée d'un graphique SHAP qui montre les 3 variables qui influencent le + le résultat pour ce client, donc une interprétation facile
# Grâce au point 2, on filtre les points affichés sur ces graphiques.
# (dans les deux cas, notre invidivu est clairement représenté sur chaque graphique pour pouvoir le situer visuellement)
# JAJOUTE CETTE LIGNE

import pandas as pd
import numpy as np
import streamlit as st
import requests
import time

DATA_URL = 'https://raw.githubusercontent.com/gregmansio/Home-credit-default-risk-OC-P7/main/app/df_test_sample'


data_load_state = st.text('Chargement des données... Veuillez-patienter')
data = pd.read_csv(DATA_URL)
time.sleep(0)
data_load_state.text('Chargement terminé')
time.sleep(0)
#def request_prediction(model_uri, data):
#    headers = {"Content-Type": "application/json"}
#
#    data_json = {'data': data}
#    response = requests.request(
#        method='POST', headers=headers, url=model_uri, json=data_json)
#
#    if response.status_code != 200:
#        raise Exception(
#            "Request failed with status {}, {}".format(response.status_code, response.text))
#
#    return response.json()

# Travail sur les données à sortir visuellement

# Âge - servira à filtrer les infos en dataframe et les deux graphhiques (pareil pour sexe et revenu)
age = 

def main():
    data_load_state.text('')
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
    
    # Sidebar de sélection de filtres (ou d'aucun filtre!)
    sidebar_title = st.sidebar.title('Filtres')
    sidebar_helper = st.sidebar.text('Sélectionner les filtres à appliquer')
    gender_choice = st.sidebar.radio( # CODE_GENDER
        label = 'Filtrer selon le sexe :', options = ('Tous', 'Femme', 'Homme'))

    age_choice = st.sidebar.radio( # DAYS_BIRTH - âge en jours au moment de la demande de crédit
        label = 'Filtrer selon l\'âge :', options = ('Tous', '18-30', '30-40', '40-50', '50-60', '60+'))

    income_choice = st.sidebar.radio( # AMT_INCOME_TOTAL - 
        label = 'Filtrer selon le revenu :', options = ('Tous', '18-30', '30-40', '40-50', '50-60', '60+'))

    # Titre de l'app
    st.title('Home Credit Scoring')

    # Sélection du client
    client = st.selectbox(label = 'Sélectionner un client', options = data.index, key='client')
    
    def client_aleatoire(): # On va chercher aléatoirement un client dans l'index,
        id_random = np.random.randint(1, len(data.index), 1)
        st.session_state.client = id_random
    st.button("Tirage aléatoire d'un client", on_click=client_aleatoire)    

    # Prédiction
    #X = request_prediction(MLFLOW_URI, data)[0] * 100000 # réponse à l'appel de l'API mlflow
    #Z = va_chercher_treshhold(MLFLOW_model) # fonction à créer pour récupérer le treshold en dessous duquel on accepte la demande de crédit
    #if X < Z : 
    #    Y = "Congratulations! Your loan request has been approved."
    #else Y = "Unfortunately, we have to decline your credit application."  
    Y_temp = "Yes"
    col1, col2 = st.columns(2)
    col1.metric(label = "Credit score", value = "X", help = "Le résultat de notre prediction. 0 est le meilleur score.")
    with col2:
        st.metric(label = "Credit accepté?", value = Y_temp)
         
    # Informations essentielles du client sélectionné (1 ligne, plusieurs colonnes)     
    df_client = data.iloc[st.session_state.client]
    st.dataframe(df_client, use_container_width = True)

    # Graphique de gauche

    # Graphique de droite

    
    



    #predict_btn = st.button('Predict')
    #if predict_btn:
    #    data = [[revenu_med, age_med, nb_piece_med, nb_chambre_moy,
    #             taille_pop, occupation_moy, latitude, longitude]]
    #    score = st.metric(label = "Credit score", value = X, help = "The resulting score of our prediction. Closer to zero is better.")         
    #    
    #    pred = request_prediction(MLFLOW_URI, data)[0] * 100000
    #    
    #    st.write(
    #        'Le prix médian d\'une habitation est de {:.2f}'.format(pred))


if __name__ == '__main__':
    main()
