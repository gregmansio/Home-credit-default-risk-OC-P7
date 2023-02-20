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
# J'AJOUTE CETTE LIGNE

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import requests
import time
import ast
import urllib.request
import json
import os
import ssl
import re

st.set_page_config(layout="wide")

DATA_URL = 'https://raw.githubusercontent.com/gregmansio/Home-credit-default-risk-OC-P7/main/data/app/df_test_sample'

data_load_state = st.text('Chargement des données... Veuillez-patienter')
data_csv = pd.read_csv(DATA_URL, index_col=0)
cols_to_drop = ['TARGET']
data_no_target = data_csv.drop(cols_to_drop, axis = 1)
time.sleep(1)
data_load_state.text('Chargement terminé')
data_csv.loc[:, 'age'] = round(abs(data_csv['DAYS_BIRTH']/365), 1)


    
# Travail sur les données à sortir visuellement

# Âge - servira à filtrer les infos en dataframe et les deux graphhiques (pareil pour sexe et revenu)
def categorize_age(age):
    if age < 30:
        return "18-30"
    elif age < 40:
        return "30-40"
    elif age < 50:
        return "40-50"
    elif age < 60:
        return "50-60"
    else:
        return "60+" 
data_csv['cat_age'] = data_csv['age'].apply(categorize_age)

# Revenu
def categorize_revenu(x):
    if x < 80000:
        return '0 à 80 000'
    elif x < 160000:
        return '80 000 à 160 000'
    elif x < 240000:
        return '160 000 à 240 000'
    else:
        return '240 000 et +'   
data_csv['cat_revenu'] = data_csv['AMT_INCOME_TOTAL'].apply(categorize_revenu)

# Sexe
def categorize_gender(x):
    if x == 0:
        return 'Homme'
    else :
        return 'Femme'
data_csv['cat_sexe'] = data_csv['CODE_GENDER'].apply(categorize_gender)


# Fonction de création du dashboard
def main():
    data_load_state.text('')

    
    # Sidebar de sélection de filtres (ou d'aucun filtre!)
    sidebar_title = st.sidebar.title('Filtres')
    sidebar_helper = st.sidebar.text('Sélectionner les filtres à appliquer')
    gender_choice = st.sidebar.radio( # CODE_GENDER
        label = 'Filtrer selon le sexe :', options = ('Tous', 'Femme', 'Homme'), key = 'gender', index=0)

    age_choice = st.sidebar.radio( # DAYS_BIRTH - âge en jours au moment de la demande de crédit
        label = 'Filtrer selon l\'âge :', options = ('Tous', '18-30', '30-40', '40-50', '50-60', '60+'), key = 'age', index=0)

    income_choice = st.sidebar.radio( # AMT_INCOME_TOTAL - 
        label = 'Filtrer selon le revenu :', options = ('Tous', '0 à 80 000', '80 000 à 160 000', '160 000 à 240 000', '240 000 et +'), key = 'income', index=0)
    
    # Titre de l'app
    st.title('Home Credit Scoring')

    # Sélection manuelle du client
    client = st.selectbox(label = 'Sélectionner un client', options = data_no_target.index, key='client')

    # Ou selection aléatoire d'un client
    #def client_aleatoire(): # On va chercher aléatoirement un client dans l'index,
        # Tirage aléatoire parmi l'index de notre dataset
    #    id_random = np.random.randint(1, len(data_no_target.index), 1)
        # Mise à jour de 'client' grâce au résultat du tirage aléatoire de la ligne juste au dessus
     #   st.session_state.client = id_random
        # Mise à jour de data_client et data_x aussi
        #st.session_state.data_client = data_no_target.iloc[id_random,]
        #data_x = np.asarray(data_client).tolist()

    #st.button("Tirage aléatoire d'un client", on_click=client_aleatoire)    
    
     # Filtrage des données clients
    data_client = data_no_target.loc[client]
    data_x = np.asarray(data_client).tolist()

    # Prédiction*
    # Fonction proposée de base par Azure ML pour gérer les certificats
    def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context

        allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

    # Definition de la fonction d'appel à l'API et d'obtention de la prédiction
    def predict(donnees, index): 
        data =  {
        "input_data": {
            "columns": [
                "index_x",
                "SK_ID_CURR",
                "CODE_GENDER",
                "FLAG_OWN_REALTY",
                "AMT_INCOME_TOTAL",
                "AMT_CREDIT",
                "AMT_ANNUITY",
                "AMT_GOODS_PRICE",
                "REGION_POPULATION_RELATIVE",
                "DAYS_BIRTH",
                "DAYS_EMPLOYED",
                "DAYS_REGISTRATION",
                "DAYS_ID_PUBLISH",
                "OWN_CAR_AGE",
                "FLAG_WORK_PHONE",
                "FLAG_PHONE",
                "CNT_FAM_MEMBERS",
                "REGION_RATING_CLIENT",
                "REGION_RATING_CLIENT_W_CITY",
                "HOUR_APPR_PROCESS_START",
                "REG_CITY_NOT_LIVE_CITY",
                "LIVE_CITY_NOT_WORK_CITY",
                "EXT_SOURCE_1",
                "EXT_SOURCE_2",
                "EXT_SOURCE_3",
                "APARTMENTS_AVG",
                "BASEMENTAREA_AVG",
                "YEARS_BEGINEXPLUATATION_AVG",
                "YEARS_BUILD_AVG",
                "COMMONAREA_AVG",
                "ELEVATORS_AVG",
                "ENTRANCES_AVG",
                "FLOORSMAX_AVG",
                "FLOORSMIN_AVG",
                "LANDAREA_AVG",
                "LIVINGAPARTMENTS_AVG",
                "LIVINGAREA_AVG",
                "NONLIVINGAPARTMENTS_AVG",
                "NONLIVINGAREA_AVG",
                "APARTMENTS_MODE",
                "BASEMENTAREA_MODE",
                "YEARS_BEGINEXPLUATATION_MODE",
                "YEARS_BUILD_MODE",
                "COMMONAREA_MODE",
                "FLOORSMIN_MODE",
                "LANDAREA_MODE",
                "LIVINGAPARTMENTS_MODE",
                "LIVINGAREA_MODE",
                "NONLIVINGAPARTMENTS_MODE",
                "NONLIVINGAREA_MODE",
                "APARTMENTS_MEDI",
                "BASEMENTAREA_MEDI",
                "YEARS_BEGINEXPLUATATION_MEDI",
                "YEARS_BUILD_MEDI",
                "COMMONAREA_MEDI",
                "ENTRANCES_MEDI",
                "FLOORSMAX_MEDI",
                "FLOORSMIN_MEDI",
                "LANDAREA_MEDI",
                "LIVINGAPARTMENTS_MEDI",
                "LIVINGAREA_MEDI",
                "NONLIVINGAPARTMENTS_MEDI",
                "NONLIVINGAREA_MEDI",
                "TOTALAREA_MODE",
                "OBS_30_CNT_SOCIAL_CIRCLE",
                "DEF_30_CNT_SOCIAL_CIRCLE",
                "OBS_60_CNT_SOCIAL_CIRCLE",
                "DEF_60_CNT_SOCIAL_CIRCLE",
                "DAYS_LAST_PHONE_CHANGE",
                "FLAG_DOCUMENT_3",
                "FLAG_DOCUMENT_8",
                "FLAG_DOCUMENT_18",
                "AMT_REQ_CREDIT_BUREAU_MON",
                "AMT_REQ_CREDIT_BUREAU_QRT",
                "AMT_REQ_CREDIT_BUREAU_YEAR",
                "NEW_CREDIT_TO_ANNUITY_RATIO",
                "NEW_CREDIT_TO_GOODS_RATIO",
                "NEW_DOC_IND_AVG",
                "NEW_DOC_IND_STD",
                "NEW_DOC_IND_KURT",
                "NEW_LIVE_IND_SUM",
                "NEW_LIVE_IND_STD",
                "NEW_LIVE_IND_KURT",
                "NEW_INC_PER_CHLD",
                "NEW_INC_BY_ORG",
                "NEW_EMPLOY_TO_BIRTH_RATIO",
                "NEW_ANNUITY_TO_INCOME_RATIO",
                "NEW_SOURCES_PROD",
                "NEW_EXT_SOURCES_MEAN",
                "NEW_SCORES_STD",
                "NEW_CAR_TO_BIRTH_RATIO",
                "NEW_CAR_TO_EMPLOY_RATIO",
                "NEW_PHONE_TO_BIRTH_RATIO",
                "NEW_PHONE_TO_EMPLOY_RATIO",
                "NEW_CREDIT_TO_INCOME_RATIO",
                "NAME_CONTRACT_TYPE_Cash loans",
                "NAME_INCOME_TYPE_Commercial associate",
                "NAME_INCOME_TYPE_State servant",
                "NAME_INCOME_TYPE_Working",
                "NAME_EDUCATION_TYPE_Higher education",
                "NAME_EDUCATION_TYPE_Incomplete higher",
                "NAME_EDUCATION_TYPE_Lower secondary",
                "NAME_EDUCATION_TYPE_Secondary / secondary special",
                "NAME_FAMILY_STATUS_Married",
                "NAME_FAMILY_STATUS_Separated",
                "NAME_HOUSING_TYPE_House / apartment",
                "NAME_HOUSING_TYPE_Municipal apartment",
                "NAME_HOUSING_TYPE_Office apartment",
                "NAME_HOUSING_TYPE_Rented apartment",
                "OCCUPATION_TYPE_Accountants",
                "OCCUPATION_TYPE_Core staff",
                "OCCUPATION_TYPE_Drivers",
                "OCCUPATION_TYPE_High skill tech staff",
                "OCCUPATION_TYPE_Laborers",
                "OCCUPATION_TYPE_Medicine staff",
                "WEEKDAY_APPR_PROCESS_START_MONDAY",
                "WEEKDAY_APPR_PROCESS_START_SATURDAY",
                "WEEKDAY_APPR_PROCESS_START_SUNDAY",
                "WEEKDAY_APPR_PROCESS_START_WEDNESDAY",
                "ORGANIZATION_TYPE_Bank",
                "ORGANIZATION_TYPE_Business Entity Type 3",
                "ORGANIZATION_TYPE_Construction",
                "ORGANIZATION_TYPE_Industry: type 9",
                "ORGANIZATION_TYPE_Kindergarten",
                "ORGANIZATION_TYPE_Medicine",
                "ORGANIZATION_TYPE_Military",
                "ORGANIZATION_TYPE_Police",
                "ORGANIZATION_TYPE_School",
                "ORGANIZATION_TYPE_Self-employed",
                "ORGANIZATION_TYPE_Transport: type 3",
                "WALLSMATERIAL_MODE_Stone, brick",
                "BURO_DAYS_CREDIT_MIN",
                "BURO_DAYS_CREDIT_MAX",
                "BURO_DAYS_CREDIT_MEAN",
                "BURO_DAYS_CREDIT_VAR",
                "BURO_DAYS_CREDIT_ENDDATE_MIN",
                "BURO_DAYS_CREDIT_ENDDATE_MAX",
                "BURO_DAYS_CREDIT_ENDDATE_MEAN",
                "BURO_DAYS_CREDIT_UPDATE_MEAN",
                "BURO_AMT_CREDIT_MAX_OVERDUE_MEAN",
                "BURO_AMT_CREDIT_SUM_MAX",
                "BURO_AMT_CREDIT_SUM_MEAN",
                "BURO_AMT_CREDIT_SUM_SUM",
                "BURO_AMT_CREDIT_SUM_DEBT_MAX",
                "BURO_AMT_CREDIT_SUM_DEBT_MEAN",
                "BURO_AMT_CREDIT_SUM_DEBT_SUM",
                "BURO_AMT_CREDIT_SUM_OVERDUE_MEAN",
                "BURO_AMT_CREDIT_SUM_LIMIT_MEAN",
                "BURO_AMT_CREDIT_SUM_LIMIT_SUM",
                "BURO_AMT_ANNUITY_MAX",
                "BURO_AMT_ANNUITY_MEAN",
                "BURO_MONTHS_BALANCE_MIN_MIN",
                "BURO_MONTHS_BALANCE_SIZE_MEAN",
                "BURO_MONTHS_BALANCE_SIZE_SUM",
                "BURO_CREDIT_ACTIVE_Active_MEAN",
                "BURO_CREDIT_ACTIVE_Closed_MEAN",
                "BURO_CREDIT_ACTIVE_Sold_MEAN",
                "BURO_CREDIT_TYPE_Another type of loan_MEAN",
                "BURO_CREDIT_TYPE_Car loan_MEAN",
                "BURO_CREDIT_TYPE_Consumer credit_MEAN",
                "BURO_CREDIT_TYPE_Credit card_MEAN",
                "BURO_CREDIT_TYPE_Microloan_MEAN",
                "BURO_CREDIT_TYPE_Mortgage_MEAN",
                "BURO_STATUS_0_MEAN_MEAN",
                "BURO_STATUS_1_MEAN_MEAN",
                "BURO_STATUS_C_MEAN_MEAN",
                "BURO_STATUS_X_MEAN_MEAN",
                "ACTIVE_DAYS_CREDIT_MIN",
                "ACTIVE_DAYS_CREDIT_MAX",
                "ACTIVE_DAYS_CREDIT_MEAN",
                "ACTIVE_DAYS_CREDIT_VAR",
                "ACTIVE_DAYS_CREDIT_ENDDATE_MIN",
                "ACTIVE_DAYS_CREDIT_ENDDATE_MAX",
                "ACTIVE_DAYS_CREDIT_ENDDATE_MEAN",
                "ACTIVE_DAYS_CREDIT_UPDATE_MEAN",
                "ACTIVE_CREDIT_DAY_OVERDUE_MAX",
                "ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN",
                "ACTIVE_AMT_CREDIT_SUM_MAX",
                "ACTIVE_AMT_CREDIT_SUM_MEAN",
                "ACTIVE_AMT_CREDIT_SUM_SUM",
                "ACTIVE_AMT_CREDIT_SUM_DEBT_MAX",
                "ACTIVE_AMT_CREDIT_SUM_DEBT_MEAN",
                "ACTIVE_AMT_CREDIT_SUM_DEBT_SUM",
                "ACTIVE_AMT_CREDIT_SUM_OVERDUE_MEAN",
                "ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN",
                "ACTIVE_AMT_CREDIT_SUM_LIMIT_SUM",
                "ACTIVE_AMT_ANNUITY_MAX",
                "ACTIVE_AMT_ANNUITY_MEAN",
                "ACTIVE_MONTHS_BALANCE_MIN_MIN",
                "ACTIVE_MONTHS_BALANCE_MAX_MAX",
                "ACTIVE_MONTHS_BALANCE_SIZE_MEAN",
                "ACTIVE_MONTHS_BALANCE_SIZE_SUM",
                "CLOSED_DAYS_CREDIT_MIN",
                "CLOSED_DAYS_CREDIT_MAX",
                "CLOSED_DAYS_CREDIT_MEAN",
                "CLOSED_DAYS_CREDIT_VAR",
                "CLOSED_DAYS_CREDIT_ENDDATE_MIN",
                "CLOSED_DAYS_CREDIT_ENDDATE_MAX",
                "CLOSED_DAYS_CREDIT_ENDDATE_MEAN",
                "CLOSED_DAYS_CREDIT_UPDATE_MEAN",
                "CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN",
                "CLOSED_AMT_CREDIT_SUM_MAX",
                "CLOSED_AMT_CREDIT_SUM_MEAN",
                "CLOSED_AMT_CREDIT_SUM_SUM",
                "CLOSED_AMT_CREDIT_SUM_DEBT_MAX",
                "CLOSED_AMT_CREDIT_SUM_DEBT_MEAN",
                "CLOSED_AMT_CREDIT_SUM_DEBT_SUM",
                "CLOSED_AMT_ANNUITY_MAX",
                "CLOSED_AMT_ANNUITY_MEAN",
                "CLOSED_MONTHS_BALANCE_MIN_MIN",
                "CLOSED_MONTHS_BALANCE_SIZE_MEAN",
                "CLOSED_MONTHS_BALANCE_SIZE_SUM",
                "NEW_RATIO_BURO_DAYS_CREDIT_MIN",
                "NEW_RATIO_BURO_DAYS_CREDIT_MAX",
                "NEW_RATIO_BURO_DAYS_CREDIT_MEAN",
                "NEW_RATIO_BURO_DAYS_CREDIT_VAR",
                "NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MIN",
                "NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MAX",
                "NEW_RATIO_BURO_DAYS_CREDIT_ENDDATE_MEAN",
                "NEW_RATIO_BURO_DAYS_CREDIT_UPDATE_MEAN",
                "NEW_RATIO_BURO_AMT_CREDIT_MAX_OVERDUE_MEAN",
                "NEW_RATIO_BURO_AMT_CREDIT_SUM_MAX",
                "NEW_RATIO_BURO_AMT_CREDIT_SUM_MEAN",
                "NEW_RATIO_BURO_AMT_CREDIT_SUM_SUM",
                "NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_MAX",
                "NEW_RATIO_BURO_AMT_CREDIT_SUM_DEBT_SUM",
                "NEW_RATIO_BURO_AMT_CREDIT_SUM_LIMIT_MEAN",
                "NEW_RATIO_BURO_AMT_ANNUITY_MAX",
                "NEW_RATIO_BURO_AMT_ANNUITY_MEAN",
                "NEW_RATIO_BURO_MONTHS_BALANCE_MIN_MIN",
                "NEW_RATIO_BURO_MONTHS_BALANCE_SIZE_MEAN",
                "NEW_RATIO_BURO_MONTHS_BALANCE_SIZE_SUM",
                "index_y",
                "PREV_AMT_ANNUITY_MIN",
                "PREV_AMT_ANNUITY_MAX",
                "PREV_AMT_ANNUITY_MEAN",
                "PREV_AMT_APPLICATION_MIN",
                "PREV_AMT_APPLICATION_MAX",
                "PREV_AMT_APPLICATION_MEAN",
                "PREV_AMT_CREDIT_MIN",
                "PREV_AMT_CREDIT_MAX",
                "PREV_AMT_CREDIT_MEAN",
                "PREV_APP_CREDIT_PERC_MIN",
                "PREV_APP_CREDIT_PERC_MAX",
                "PREV_APP_CREDIT_PERC_MEAN",
                "PREV_APP_CREDIT_PERC_VAR",
                "PREV_AMT_DOWN_PAYMENT_MIN",
                "PREV_AMT_DOWN_PAYMENT_MAX",
                "PREV_AMT_DOWN_PAYMENT_MEAN",
                "PREV_AMT_GOODS_PRICE_MIN",
                "PREV_AMT_GOODS_PRICE_MAX",
                "PREV_AMT_GOODS_PRICE_MEAN",
                "PREV_HOUR_APPR_PROCESS_START_MIN",
                "PREV_HOUR_APPR_PROCESS_START_MAX",
                "PREV_HOUR_APPR_PROCESS_START_MEAN",
                "PREV_RATE_DOWN_PAYMENT_MIN",
                "PREV_RATE_DOWN_PAYMENT_MAX",
                "PREV_RATE_DOWN_PAYMENT_MEAN",
                "PREV_DAYS_DECISION_MIN",
                "PREV_DAYS_DECISION_MAX",
                "PREV_DAYS_DECISION_MEAN",
                "PREV_CNT_PAYMENT_MEAN",
                "PREV_CNT_PAYMENT_SUM",
                "PREV_NAME_CONTRACT_TYPE_Cash loans_MEAN",
                "PREV_NAME_CONTRACT_TYPE_Consumer loans_MEAN",
                "PREV_NAME_CONTRACT_TYPE_Revolving loans_MEAN",
                "PREV_WEEKDAY_APPR_PROCESS_START_FRIDAY_MEAN",
                "PREV_WEEKDAY_APPR_PROCESS_START_MONDAY_MEAN",
                "PREV_WEEKDAY_APPR_PROCESS_START_SATURDAY_MEAN",
                "PREV_WEEKDAY_APPR_PROCESS_START_SUNDAY_MEAN",
                "PREV_WEEKDAY_APPR_PROCESS_START_THURSDAY_MEAN",
                "PREV_WEEKDAY_APPR_PROCESS_START_TUESDAY_MEAN",
                "PREV_WEEKDAY_APPR_PROCESS_START_WEDNESDAY_MEAN",
                "PREV_NAME_CASH_LOAN_PURPOSE_Car repairs_MEAN",
                "PREV_NAME_CASH_LOAN_PURPOSE_Medicine_MEAN",
                "PREV_NAME_CASH_LOAN_PURPOSE_Repairs_MEAN",
                "PREV_NAME_CASH_LOAN_PURPOSE_Urgent needs_MEAN",
                "PREV_NAME_CASH_LOAN_PURPOSE_XNA_MEAN",
                "PREV_NAME_CONTRACT_STATUS_Approved_MEAN",
                "PREV_NAME_CONTRACT_STATUS_Canceled_MEAN",
                "PREV_NAME_CONTRACT_STATUS_Refused_MEAN",
                "PREV_NAME_PAYMENT_TYPE_Cash through the bank_MEAN",
                "PREV_NAME_PAYMENT_TYPE_XNA_MEAN",
                "PREV_CODE_REJECT_REASON_HC_MEAN",
                "PREV_CODE_REJECT_REASON_LIMIT_MEAN",
                "PREV_CODE_REJECT_REASON_SCO_MEAN",
                "PREV_CODE_REJECT_REASON_SCOFR_MEAN",
                "PREV_CODE_REJECT_REASON_XAP_MEAN",
                "PREV_NAME_TYPE_SUITE_Children_MEAN",
                "PREV_NAME_TYPE_SUITE_Family_MEAN",
                "PREV_NAME_TYPE_SUITE_Other_A_MEAN",
                "PREV_NAME_TYPE_SUITE_Other_B_MEAN",
                "PREV_NAME_TYPE_SUITE_Spouse, partner_MEAN",
                "PREV_NAME_TYPE_SUITE_Unaccompanied_MEAN",
                "PREV_NAME_TYPE_SUITE_nan_MEAN",
                "PREV_NAME_CLIENT_TYPE_New_MEAN",
                "PREV_NAME_CLIENT_TYPE_Refreshed_MEAN",
                "PREV_NAME_CLIENT_TYPE_Repeater_MEAN",
                "PREV_NAME_GOODS_CATEGORY_Audio/Video_MEAN",
                "PREV_NAME_GOODS_CATEGORY_Computers_MEAN",
                "PREV_NAME_GOODS_CATEGORY_Consumer Electronics_MEAN",
                "PREV_NAME_GOODS_CATEGORY_Furniture_MEAN",
                "PREV_NAME_GOODS_CATEGORY_Mobile_MEAN",
                "PREV_NAME_GOODS_CATEGORY_Photo / Cinema Equipment_MEAN",
                "PREV_NAME_GOODS_CATEGORY_Sport and Leisure_MEAN",
                "PREV_NAME_PORTFOLIO_Cards_MEAN",
                "PREV_NAME_PORTFOLIO_Cash_MEAN",
                "PREV_NAME_PORTFOLIO_POS_MEAN",
                "PREV_NAME_PORTFOLIO_XNA_MEAN",
                "PREV_NAME_PRODUCT_TYPE_XNA_MEAN",
                "PREV_NAME_PRODUCT_TYPE_walk-in_MEAN",
                "PREV_NAME_PRODUCT_TYPE_x-sell_MEAN",
                "PREV_CHANNEL_TYPE_AP+ (Cash loan)_MEAN",
                "PREV_CHANNEL_TYPE_Channel of corporate sales_MEAN",
                "PREV_CHANNEL_TYPE_Contact center_MEAN",
                "PREV_CHANNEL_TYPE_Country-wide_MEAN",
                "PREV_CHANNEL_TYPE_Credit and cash offices_MEAN",
                "PREV_CHANNEL_TYPE_Regional / Local_MEAN",
                "PREV_CHANNEL_TYPE_Stone_MEAN",
                "PREV_NAME_SELLER_INDUSTRY_Auto technology_MEAN",
                "PREV_NAME_SELLER_INDUSTRY_Clothing_MEAN",
                "PREV_NAME_SELLER_INDUSTRY_Connectivity_MEAN",
                "PREV_NAME_SELLER_INDUSTRY_Consumer electronics_MEAN",
                "PREV_NAME_SELLER_INDUSTRY_XNA_MEAN",
                "PREV_NAME_YIELD_GROUP_XNA_MEAN",
                "PREV_NAME_YIELD_GROUP_high_MEAN",
                "PREV_NAME_YIELD_GROUP_low_action_MEAN",
                "PREV_NAME_YIELD_GROUP_low_normal_MEAN",
                "PREV_NAME_YIELD_GROUP_middle_MEAN",
                "PREV_PRODUCT_COMBINATION_Card Street_MEAN",
                "PREV_PRODUCT_COMBINATION_Card X-Sell_MEAN",
                "PREV_PRODUCT_COMBINATION_Cash_MEAN",
                "PREV_PRODUCT_COMBINATION_Cash Street: high_MEAN",
                "PREV_PRODUCT_COMBINATION_Cash Street: low_MEAN",
                "PREV_PRODUCT_COMBINATION_Cash Street: middle_MEAN",
                "PREV_PRODUCT_COMBINATION_Cash X-Sell: high_MEAN",
                "PREV_PRODUCT_COMBINATION_Cash X-Sell: low_MEAN",
                "PREV_PRODUCT_COMBINATION_Cash X-Sell: middle_MEAN",
                "PREV_PRODUCT_COMBINATION_POS household with interest_MEAN",
                "PREV_PRODUCT_COMBINATION_POS household without interest_MEAN",
                "PREV_PRODUCT_COMBINATION_POS industry with interest_MEAN",
                "PREV_PRODUCT_COMBINATION_POS mobile with interest_MEAN",
                "PREV_PRODUCT_COMBINATION_POS other with interest_MEAN",
                "APPROVED_AMT_ANNUITY_MIN",
                "APPROVED_AMT_ANNUITY_MAX",
                "APPROVED_AMT_ANNUITY_MEAN",
                "APPROVED_AMT_APPLICATION_MIN",
                "APPROVED_AMT_APPLICATION_MAX",
                "APPROVED_AMT_APPLICATION_MEAN",
                "APPROVED_AMT_CREDIT_MIN",
                "APPROVED_AMT_CREDIT_MAX",
                "APPROVED_AMT_CREDIT_MEAN",
                "APPROVED_APP_CREDIT_PERC_MIN",
                "APPROVED_APP_CREDIT_PERC_MAX",
                "APPROVED_APP_CREDIT_PERC_MEAN",
                "APPROVED_APP_CREDIT_PERC_VAR",
                "APPROVED_AMT_DOWN_PAYMENT_MIN",
                "APPROVED_AMT_DOWN_PAYMENT_MAX",
                "APPROVED_AMT_DOWN_PAYMENT_MEAN",
                "APPROVED_AMT_GOODS_PRICE_MIN",
                "APPROVED_AMT_GOODS_PRICE_MAX",
                "APPROVED_AMT_GOODS_PRICE_MEAN",
                "APPROVED_HOUR_APPR_PROCESS_START_MIN",
                "APPROVED_HOUR_APPR_PROCESS_START_MAX",
                "APPROVED_HOUR_APPR_PROCESS_START_MEAN",
                "APPROVED_RATE_DOWN_PAYMENT_MIN",
                "APPROVED_RATE_DOWN_PAYMENT_MAX",
                "APPROVED_RATE_DOWN_PAYMENT_MEAN",
                "APPROVED_DAYS_DECISION_MIN",
                "APPROVED_DAYS_DECISION_MAX",
                "APPROVED_DAYS_DECISION_MEAN",
                "APPROVED_CNT_PAYMENT_MEAN",
                "APPROVED_CNT_PAYMENT_SUM",
                "REFUSED_AMT_ANNUITY_MIN",
                "REFUSED_AMT_ANNUITY_MAX",
                "REFUSED_AMT_ANNUITY_MEAN",
                "REFUSED_AMT_APPLICATION_MIN",
                "REFUSED_AMT_APPLICATION_MAX",
                "REFUSED_AMT_APPLICATION_MEAN",
                "REFUSED_AMT_CREDIT_MIN",
                "REFUSED_AMT_CREDIT_MAX",
                "REFUSED_AMT_CREDIT_MEAN",
                "REFUSED_APP_CREDIT_PERC_MIN",
                "REFUSED_APP_CREDIT_PERC_MAX",
                "REFUSED_APP_CREDIT_PERC_MEAN",
                "REFUSED_APP_CREDIT_PERC_VAR",
                "REFUSED_AMT_DOWN_PAYMENT_MIN",
                "REFUSED_AMT_GOODS_PRICE_MIN",
                "REFUSED_AMT_GOODS_PRICE_MAX",
                "REFUSED_AMT_GOODS_PRICE_MEAN",
                "REFUSED_HOUR_APPR_PROCESS_START_MIN",
                "REFUSED_HOUR_APPR_PROCESS_START_MAX",
                "REFUSED_HOUR_APPR_PROCESS_START_MEAN",
                "REFUSED_RATE_DOWN_PAYMENT_MAX",
                "REFUSED_RATE_DOWN_PAYMENT_MEAN",
                "REFUSED_DAYS_DECISION_MIN",
                "REFUSED_DAYS_DECISION_MAX",
                "REFUSED_DAYS_DECISION_MEAN",
                "REFUSED_CNT_PAYMENT_MEAN",
                "REFUSED_CNT_PAYMENT_SUM",
                "NEW_RATIO_PREV_AMT_ANNUITY_MIN",
                "NEW_RATIO_PREV_AMT_ANNUITY_MAX",
                "NEW_RATIO_PREV_AMT_ANNUITY_MEAN",
                "NEW_RATIO_PREV_AMT_APPLICATION_MIN",
                "NEW_RATIO_PREV_AMT_APPLICATION_MAX",
                "NEW_RATIO_PREV_AMT_APPLICATION_MEAN",
                "NEW_RATIO_PREV_AMT_CREDIT_MIN",
                "NEW_RATIO_PREV_AMT_CREDIT_MAX",
                "NEW_RATIO_PREV_AMT_CREDIT_MEAN",
                "NEW_RATIO_PREV_APP_CREDIT_PERC_MIN",
                "NEW_RATIO_PREV_APP_CREDIT_PERC_MAX",
                "NEW_RATIO_PREV_APP_CREDIT_PERC_MEAN",
                "NEW_RATIO_PREV_APP_CREDIT_PERC_VAR",
                "NEW_RATIO_PREV_AMT_DOWN_PAYMENT_MAX",
                "NEW_RATIO_PREV_AMT_DOWN_PAYMENT_MEAN",
                "NEW_RATIO_PREV_AMT_GOODS_PRICE_MIN",
                "NEW_RATIO_PREV_AMT_GOODS_PRICE_MAX",
                "NEW_RATIO_PREV_AMT_GOODS_PRICE_MEAN",
                "NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MIN",
                "NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MAX",
                "NEW_RATIO_PREV_HOUR_APPR_PROCESS_START_MEAN",
                "NEW_RATIO_PREV_RATE_DOWN_PAYMENT_MIN",
                "NEW_RATIO_PREV_RATE_DOWN_PAYMENT_MEAN",
                "NEW_RATIO_PREV_DAYS_DECISION_MIN",
                "NEW_RATIO_PREV_DAYS_DECISION_MAX",
                "NEW_RATIO_PREV_DAYS_DECISION_MEAN",
                "NEW_RATIO_PREV_CNT_PAYMENT_MEAN",
                "NEW_RATIO_PREV_CNT_PAYMENT_SUM",
                "POS_MONTHS_BALANCE_MAX",
                "POS_MONTHS_BALANCE_MEAN",
                "POS_MONTHS_BALANCE_SIZE",
                "POS_SK_DPD_MAX",
                "POS_SK_DPD_MEAN",
                "POS_SK_DPD_DEF_MAX",
                "POS_SK_DPD_DEF_MEAN",
                "POS_NAME_CONTRACT_STATUS_Active_MEAN",
                "POS_NAME_CONTRACT_STATUS_Completed_MEAN",
                "POS_NAME_CONTRACT_STATUS_Returned to the store_MEAN",
                "POS_NAME_CONTRACT_STATUS_Signed_MEAN",
                "POS_COUNT",
                "INSTAL_NUM_INSTALMENT_VERSION_NUNIQUE",
                "INSTAL_DPD_MAX",
                "INSTAL_DPD_MEAN",
                "INSTAL_DPD_SUM",
                "INSTAL_DBD_MAX",
                "INSTAL_DBD_MEAN",
                "INSTAL_DBD_SUM",
                "INSTAL_PAYMENT_PERC_MAX",
                "INSTAL_PAYMENT_PERC_MEAN",
                "INSTAL_PAYMENT_PERC_SUM",
                "INSTAL_PAYMENT_PERC_VAR",
                "INSTAL_PAYMENT_DIFF_MAX",
                "INSTAL_PAYMENT_DIFF_MEAN",
                "INSTAL_PAYMENT_DIFF_SUM",
                "INSTAL_PAYMENT_DIFF_VAR",
                "INSTAL_AMT_INSTALMENT_MAX",
                "INSTAL_AMT_INSTALMENT_MEAN",
                "INSTAL_AMT_INSTALMENT_SUM",
                "INSTAL_AMT_PAYMENT_MIN",
                "INSTAL_AMT_PAYMENT_MAX",
                "INSTAL_AMT_PAYMENT_MEAN",
                "INSTAL_AMT_PAYMENT_SUM",
                "INSTAL_DAYS_ENTRY_PAYMENT_MAX",
                "INSTAL_DAYS_ENTRY_PAYMENT_MEAN",
                "INSTAL_DAYS_ENTRY_PAYMENT_SUM",
                "INSTAL_COUNT",
                "CC_MONTHS_BALANCE_VAR",
                "CC_AMT_BALANCE_MIN",
                "CC_AMT_BALANCE_MAX",
                "CC_AMT_BALANCE_MEAN",
                "CC_AMT_BALANCE_SUM",
                "CC_AMT_BALANCE_VAR",
                "CC_AMT_CREDIT_LIMIT_ACTUAL_MIN",
                "CC_AMT_CREDIT_LIMIT_ACTUAL_MAX",
                "CC_AMT_CREDIT_LIMIT_ACTUAL_MEAN",
                "CC_AMT_CREDIT_LIMIT_ACTUAL_SUM",
                "CC_AMT_CREDIT_LIMIT_ACTUAL_VAR",
                "CC_AMT_DRAWINGS_ATM_CURRENT_MAX",
                "CC_AMT_DRAWINGS_ATM_CURRENT_MEAN",
                "CC_AMT_DRAWINGS_ATM_CURRENT_SUM",
                "CC_AMT_DRAWINGS_ATM_CURRENT_VAR",
                "CC_AMT_DRAWINGS_CURRENT_MAX",
                "CC_AMT_DRAWINGS_CURRENT_MEAN",
                "CC_AMT_DRAWINGS_CURRENT_SUM",
                "CC_AMT_DRAWINGS_CURRENT_VAR",
                "CC_AMT_DRAWINGS_POS_CURRENT_MIN",
                "CC_AMT_DRAWINGS_POS_CURRENT_MAX",
                "CC_AMT_DRAWINGS_POS_CURRENT_MEAN",
                "CC_AMT_DRAWINGS_POS_CURRENT_SUM",
                "CC_AMT_DRAWINGS_POS_CURRENT_VAR",
                "CC_AMT_INST_MIN_REGULARITY_MAX",
                "CC_AMT_INST_MIN_REGULARITY_MEAN",
                "CC_AMT_INST_MIN_REGULARITY_SUM",
                "CC_AMT_INST_MIN_REGULARITY_VAR",
                "CC_AMT_PAYMENT_CURRENT_MIN",
                "CC_AMT_PAYMENT_CURRENT_MAX",
                "CC_AMT_PAYMENT_CURRENT_MEAN",
                "CC_AMT_PAYMENT_CURRENT_SUM",
                "CC_AMT_PAYMENT_CURRENT_VAR",
                "CC_AMT_PAYMENT_TOTAL_CURRENT_MAX",
                "CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN",
                "CC_AMT_PAYMENT_TOTAL_CURRENT_SUM",
                "CC_AMT_RECEIVABLE_PRINCIPAL_MIN",
                "CC_AMT_RECEIVABLE_PRINCIPAL_MAX",
                "CC_AMT_RECEIVABLE_PRINCIPAL_MEAN",
                "CC_AMT_RECEIVABLE_PRINCIPAL_SUM",
                "CC_AMT_RECEIVABLE_PRINCIPAL_VAR",
                "CC_AMT_RECIVABLE_MIN",
                "CC_AMT_RECIVABLE_MAX",
                "CC_AMT_RECIVABLE_MEAN",
                "CC_AMT_RECIVABLE_VAR",
                "CC_AMT_TOTAL_RECEIVABLE_MEAN",
                "CC_CNT_DRAWINGS_ATM_CURRENT_MAX",
                "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN",
                "CC_CNT_DRAWINGS_ATM_CURRENT_SUM",
                "CC_CNT_DRAWINGS_ATM_CURRENT_VAR",
                "CC_CNT_DRAWINGS_CURRENT_MAX",
                "CC_CNT_DRAWINGS_CURRENT_MEAN",
                "CC_CNT_DRAWINGS_CURRENT_SUM",
                "CC_CNT_DRAWINGS_CURRENT_VAR",
                "CC_CNT_DRAWINGS_POS_CURRENT_MIN",
                "CC_CNT_DRAWINGS_POS_CURRENT_MAX",
                "CC_CNT_DRAWINGS_POS_CURRENT_MEAN",
                "CC_CNT_DRAWINGS_POS_CURRENT_VAR",
                "CC_CNT_INSTALMENT_MATURE_CUM_MEAN",
                "CC_CNT_INSTALMENT_MATURE_CUM_SUM",
                "CC_CNT_INSTALMENT_MATURE_CUM_VAR",
                "CC_SK_DPD_DEF_MEAN",
                "CC_NAME_CONTRACT_STATUS_Active_MEAN",
                "CC_NAME_CONTRACT_STATUS_Active_SUM",
                "CC_NAME_CONTRACT_STATUS_Active_VAR"
                ],
            "index": [index],
            "data": [donnees]
          }
        }

        body = str.encode(json.dumps(data))

        url = 'https://projet-oc-conbe.francecentral.inference.ml.azure.com/score'
        # Replace this with the primary/secondary key or AMLToken for the endpoint
        api_key = 'FqwceCO7HjnNj22lc26MRHj2BLRgqXFI'
        if not api_key:
            raise Exception("A key should be provided to invoke the endpoint")

        # The azureml-model-deployment header will force the request to go to a specific deployment.
        # Remove this header to have the request observe the endpoint traffic rules
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'proba-predict-perso-1' }

        req = urllib.request.Request(url, body, headers)

        try:
            response = urllib.request.urlopen(req)

            result = response.read()
            print(result)
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))

            # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
            print(error.info())
            print(error.read().decode("utf8", 'ignore'))
        score_raw = result.decode()
        res_str_list = ast.literal_eval(score_raw)
        score = res_str_list[0][1]
        if score <= 0.075: # treshold 
            reponse = "Félicitations! Votre demande de crédit a été acceptée"
        else :
            reponse = "Malheureusement, votre demande de crédit ne peux pas aboutir pour le moment."
        score_display = round(score, 3)
        seuil = 0.075

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label = 'Votre score', value = score_display)

        with col2:
            st.metric(label = 'Seuil maximal', value = seuil, help="Vous devez obteni un score inférieur pour que votre demande de crédit soit acceptée")
            

        
        
        st.text(reponse)
        st.caption("Le résultat ci-dessus a été duement étudié par nos services. Veuillez trouver ci-dessous des éléments étayants notre réponse.")

    # Appel de la fonction de prédiction quand le bouton défini ci-dessous est cliqué
    if st.button("Votre résultat"):
        predict(data_x, client)
          
    # Informations essentielles du client sélectionné (1 ligne, plusieurs colonnes) mais aussi des tous les clients auxquels on souhaite le comparer     
    df_client = data_csv.loc[client]

    # On sélectionne ensuite les quelques colonnes de ce dataset qu'on veut montrer 
    df_client = pd.DataFrame(df_client)
    cols_to_display = ['cat_sexe', 'age', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'NEW_CREDIT_TO_INCOME_RATIO', 'NEW_CREDIT_TO_ANNUITY_RATIO','BURO_DAYS_CREDIT_MEAN', 'INSTAL_DPD_MAX', 'cat_age', 'cat_revenu']
    df_client_filtered = df_client.loc[cols_to_display]
    df_clients_filtered = data_csv[cols_to_display]


    # Filtrer les données en fonction des options sélectionnées
    # On va distinguer tous les cas de figure
    if gender_choice == "Tous" and age_choice == "Tous" and income_choice == "Tous":
        df_clients_filtered = df_clients_filtered
    else :
        if gender_choice == "Tous" and age_choice != "Tous" and income_choice != "Tous":
            df_clients_filtered = df_clients_filtered.loc[(df_clients_filtered["cat_age"] == age_choice) & (df_clients_filtered["cat_revenu"] == income_choice)]
        else :
            if gender_choice == "Tous" and age_choice == "Tous" and income_choice != "Tous":
                df_clients_filtered = df_clients_filtered.loc[(df_clients_filtered["cat_revenu"] == income_choice)]
            else : 
                if gender_choice == "Tous" and age_choice != "Tous" and income_choice == "Tous":
                    df_clients_filtered = df_clients_filtered.loc[(df_clients_filtered["cat_age"] == age_choice)]
                else :
                    if gender_choice != "Tous" and age_choice == "Tous" and income_choice == "Tous":
                        df_clients_filtered = df_clients_filtered.loc[(df_clients_filtered["cat_sexe"] == gender_choice)]
                    else :
                        if gender_choice != "Tous" and age_choice != "Tous" and income_choice == "Tous":
                            df_clients_filtered = df_clients_filtered.loc[(df_clients_filtered["cat_sexe"] == gender_choice) & (df_clients_filtered["cat_age"] == age_choice)]
                        else :
                            if gender_choice != "Tous" and age_choice != "Tous" and income_choice != "Tous":
                                df_clients_filtered = df_clients_filtered.loc[(df_clients_filtered["cat_sexe"] == gender_choice) & (df_clients_filtered["cat_age"] == age_choice) & (df_clients_filtered["cat_revenu"] == income_choice)]
                            else :
                                df_clients_filtered = df_clients_filtered.loc[(df_clients_filtered["cat_sexe"] == gender_choice) & (df_clients_filtered["cat_revenu"] == income_choice)]

                            
    # Tableaux de données
    df_wide = df_client_filtered.transpose()
    df_wide['BURO_DAYS_CREDIT_MEAN'] = abs(df_wide['BURO_DAYS_CREDIT_MEAN'])
    df_wide_display = df_wide
    df_wide_display = df_wide_display.rename(columns = {
        "cat_sexe" : "Sexe",
        "age" : "Age",
        "AMT_INCOME_TOTAL" : "Revenu annuel",
        "AMT_CREDIT" : "Montant total du crédit",
        "NEW_CREDIT_TO_INCOME_RATIO" : "Ratio Crédit / Revenu",
        "AMT_ANNUITY" : "Annuité",
        "BURO_DAYS_CREDIT_MEAN" : "Jours depuis dernière demande",
        "NEW_CREDIT_TO_ANNUITY_RATIO" : "Ratio Crédit / Annuité",
        'INSTAL_DPD_MAX' : "Nombre max de jours de retard de paiement", 
        'cat_age' : "Catégorie âge", 
        'cat_revenu' : "Catégorie de revenu"
    }) 

    df_clients_filtered['BURO_DAYS_CREDIT_MEAN'] = abs(df_clients_filtered['BURO_DAYS_CREDIT_MEAN'])
    df_clients_display = df_clients_filtered
    df_clients_display = df_clients_display.rename(columns = {
        "cat_sexe" : "Sexe",
        "age" : "Age",
        "AMT_INCOME_TOTAL" : "Revenu annuel",
        "AMT_CREDIT" : "Montant total du crédit",
        "NEW_CREDIT_TO_INCOME_RATIO" : "Ratio Crédit / Revenu",
        "AMT_ANNUITY" : "Annuité",
        "BURO_DAYS_CREDIT_MEAN" : "Jours depuis dernière demande",
        "NEW_CREDIT_TO_ANNUITY_RATIO" : "Ratio Crédit / Annuité",
        'INSTAL_DPD_MAX' : "Nombre max de jours de retard de paiement", 
        'cat_age' : "Catégorie âge", 
        'cat_revenu' : "Catégorie de revenu"
    })
    with st.container():
        st.subheader("Tableaux d'informations importantes")
        st.caption('Le client sélectionné')
        st.dataframe(df_wide_display, use_container_width = True)
        st.caption('Tous les clients')
        st.dataframe(df_clients_display, use_container_width = True)

    # Scatterplot credit_annuity ratio
    st.subheader('Graphique du ratio Crédit sur Annuité et Crédit sur Revenu')
    st.caption("Sur le graphique de gauche, un client dans le bas du cone rembourse lentement son crédit")
    sc_plot1 = alt.Chart(df_wide_display).mark_circle().encode(
        x = 'Montant total du crédit',
        y = "Annuité",
        color = alt.value('darkred'),
        size = alt.value(400)
    )

    sc_plot2 = alt.Chart(df_clients_display.reset_index()).mark_circle().encode(
        x = 'Montant total du crédit',
        y = "Annuité",
        size = alt.value(70),
        tooltip=['index', 'Montant total du crédit', "Annuité"]
    )
    full_sc_chart = sc_plot2 + sc_plot1 

    
    # Scatterplot credit_income ratio
    st.caption("Sur le graphique de droite, un client en haut à gauche du nuage de points a un crédit faible par rapport à ses revenus")
    sc_plot3 = alt.Chart(df_wide_display).mark_circle().encode(
        x = 'Montant total du crédit',
        y = "Revenu annuel",
        color = alt.value('darkred'),
        size = alt.value(400)
    )

    sc_plot4 = alt.Chart(df_clients_display.reset_index()).mark_circle().encode(
        x = 'Montant total du crédit',
        y = "Revenu annuel",
        size = alt.value(70),
        tooltip=['index', 'Montant total du crédit', "Revenu annuel"]
    )
    full_sc_chart2 = sc_plot4 + sc_plot3 
    
    st.altair_chart(full_sc_chart | full_sc_chart2)



if __name__ == '__main__':
    main()
