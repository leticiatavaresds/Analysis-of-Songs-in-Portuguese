#!/usr/bin/env python3.9

"""
Author: Letícia Tavares
Date: 2024-08-05

"""

# Standard library imports
import json
import os
import re
from datetime import timedelta
from time import strftime, localtime

# Third-party library imports
from loguru import logger
from lyricsgenius import Genius
from multiset import Multiset
from textdistance import damerau_levenshtein
from unidecode import unidecode
import requests

# Local application/library specific imports
from vars import credential_kaggle, credential_lastfm, credential_spotify, credential_genius , folder_input


################# API CONNECTIONS #################

def getSpotifyDict():
    # Open Spotify Credentials 
    with open(f"{folder_input}/{credential_spotify}") as test:
        dict_credenctials_sp = json.load(test)

    dict_credenctials_sp = {int(k): value for k, value in dict_credenctials_sp.items()}

    os.environ['SPOTIPY_REDIRECT_URI'] = "http://localhost:8888/tree"

    return dict_credenctials_sp

def getLastFm():
    f = open(f'{folder_input}/{credential_lastfm}')
    data = json.load(f)
    user_agent = data['user_agent']
    api_key = data['api_key']

    return user_agent, api_key


def connectKaggleApi():
    # Open JSON file
    f = open(f'{folder_input}/{credential_kaggle}')

    data = json.load(f)

    # Declare credentials for connecting to the kaggle API
    os.environ['KAGGLE_USERNAME'] = data['username']
    os.environ['KAGGLE_KEY'] = data['key']

    from kaggle.api.kaggle_api_extended import KaggleApi

    # Start connection to the Kaggle API
    api = KaggleApi()
    api.authenticate()

    return api


def getGenius():
    f = open(f'{folder_input}/{credential_genius}')
    data = json.load(f)
    genius_key = data['key']
    genius = Genius(genius_key)

    return genius

################# TEXT PROCCESS #################

def tokenizeText(txt):
    
    # Convert a phrase into a count of bigram tokens of its words
    arr = []
    for wrd in txt.lower().split('  '):
        arr += ([wrd] if len(wrd) == 1 else [wrd[i:i+2]
                for i in range(len(wrd)-1)])
        
    return Multiset(arr)

def cleanString(text):

    # Remove the accents
    text = unidecode(text)

    text = text.replace(" & ", " e ")

    # Remove special characters
    text = re.sub('[^a-zA-Z0-9]', ' ', text)

    # Remove space at the beginning of the string and at the end
    text = text.strip()

    # Remove multiple spaces
    text = re.sub(' +', ' ', text)

    # Convert all uppercase characters in a string into lowercase characters 
    text = text.lower()
    
    return text

def secondsToStr(elapsed=None):

    # Convert seconds to string in the format "days hours:minutes:seconds"
    if elapsed is None:
        return strftime("%d %H:%M:%s", localtime())
    else:
        return str(timedelta(seconds=elapsed))

def getStringBefore(name, substring):
    return name.split(substring)[0]


################# Cclculate Similarity #################


def sorensonDice(text1, text2):
    
    # Sorenson-Dice similarity of Multisets
    text1, text2 = tokenizeText(text1), tokenizeText(text2)
    dice = 2 * len(text1 & text2) / (len(text1) + len(text2))

    return dice


def getMatchSimilarity(str1, str2):

    # Clean the Strings
    str1, str2 = cleanString(str1), cleanString(str2)

    # Calculate the Soreson Dice 
    dice_sim = sorensonDice(str1, str2)
    
    # Calculate the Damerau Lavenshtein distance
    lv_sim = damerau_levenshtein.normalized_similarity(str1, str2)

    # Set the weights
    dice_weight = 0.8
    lv_weight = 1 - dice_weight

    # Calculate the Score
    score =  (lv_sim * lv_weight) + (dice_sim * dice_weight)

    return {
        'dice': dice_sim,
        'lv': lv_sim,
        'score': score
    }
################# FOR SPOTIFY SCRIPTS #################

def solveError(error):

    global dict_credenctials_sp, credential

    # 429 error indicates that app has reached the Web API rate limit
    # So change to another credential for get a new access token
    if error == "429":
        credential += 1

    index = credential % len(dict_credenctials_sp)   

    client_id = dict_credenctials_sp[index]["client_id"]
    client_secret = dict_credenctials_sp[index]["client_secret"]

    # Get new access token
    access_token = spotifyAccessToken(client_id, client_secret)

    # Return access token
    return access_token


def processResponse(response, url, params):

    global access_token, dict_credenctials_sp

    status_code = str(response.status_code)

    # If error is equal to 401 or 429, try another credentials
    if status_code == "401" or status_code == "429":

        for i in range (len(dict_credenctials_sp)):

            logger.info(f"Trying credential {i}.")

            # Get new access token
            access_token = solveError(status_code)

            header = {
                'Authorization': f'Bearer {access_token}'}
            
            # Try to get a new response with new access token
            response = requests.get(url,  params = params, headers = header, timeout=15)
            status_code = str(response.status_code)

            # If status is diferent from erros 401 or 4029, stop trying other credentials
            if status_code != "401" and status_code != "429":
                break

            logger.error(f"{response.status_code} - {response.reason}")

            # If all the credential gets en error, inform that is necessary wait some hour to run the code again and exit the application
            if i == (len(dict_credenctials_sp) - 1):
                logger.error(f"Currently all credentials are showing authorization error 429. Wait 14 hours and then run this code again.")
                exit()

    # If response has no errors, return the response json
    if status_code == "200":
        response = response.json()
    
    # Else return a dict containing the error
    else:
        response = {"error": status_code}

    return response


def spotifyAccessToken(client_id, client_secret):

    AUTH_URL = 'https://accounts.spotify.com/api/token'
    
    # Get access token from Spotify API
    auth_response = requests.post(AUTH_URL, {
                                            'grant_type': 'client_credentials',
                                            'client_id': client_id,
                                            'client_secret': client_secret,
                                        })

    # Save the access token
    access_token = auth_response.json()['access_token']

    # Return access token
    return access_token