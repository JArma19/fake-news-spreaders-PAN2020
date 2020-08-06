# Armanini Justin
# D'Intorni Michael
# Fersini Elisabetta
# Fake news spreaders profiling


import _pickle as pickle
import pandas as pd
import numpy as np
import sys
import stanza
import emoji
import glob
import re
import xml.etree.ElementTree as ET
import os
import pyPredict
import tensorflow as tf
import tensorflow_hub as hub
from collections import Counter
import getopt
import gc


'''
Funzioni helper
'''


def extract_features(path, model, nlp, users_feats_test, output_xml, nrc_df):
    # lettura di tutti i file
    authors_files = [f for f in glob.glob(path)]
    authors_files.sort()
    for author_path in authors_files:
        # estrae id utente
        author_id = os.path.split(author_path)[-1][:-4]

        # albero DOM dell'xml
        tree = ET.parse(author_path)

        # estrae attributo del tag radice
        lang = list(tree.getroot().attrib.values())[0]

        root = tree.getroot()
        full_tweets = ""  # tutti i tweet in una sola stringa
        # root ha un solo figlio, ottengo stringa unica
        for tweet in root[0]:
            full_tweets += tweet.text + " "

        # personality
        personality = pyPredict.runPredictor(root[0])

        # Processamento da parte della Pipeline
        doc = nlp(full_tweets)

        # Lista delle parole appiattita
        words = flatten_list(doc.sentences)

        # conteggio parole per emozione (lista)
        count_emotions = count_words_per_emotions(lang, words, nrc_df)

        # conteggio stile (lista)
        count_style = count_style_metric(doc, full_tweets, words)

        # embeddings
        messages = [tweet.text for tweet in root[0]]
        message_embeddings = embed(messages, model)
        embed_list = [np.array(message_embedding) for message_embedding in np.array(message_embeddings).tolist()]
        embeds_mean_per_user = np.mean(embed_list, axis=0)

        # Lista con valori per ogni utente
        style = list(count_style.values())
        tmp_list = [*count_emotions, *style, personality, *(embeds_mean_per_user.tolist())]

        # Accodo/aggiungo
        users_feats_test.append(tmp_list)  # lista di liste
        output_xml.append([author_id, lang])
        print([author_id, lang])


def count_words_per_emotions(lang, words, nrc_df):
    count = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    for word in words:
        if word.lemma.lower() in nrc_df[lang].values:
            # prendo la riga corrispondente a quella parola e sommo vettore emozioni
            # tolgo le prime due colonne che sono la parola in en e es
            count += np.array(nrc_df.loc[nrc_df[lang] == word.lemma.lower()].values.tolist()[0][2:])

    return count.tolist()


def count_style_metric(doc, string_text, words):
    freq = dict()  # dizionario frequenze di tutti gli aspetti stilistici
    words_ = [word.text for word in words]
    n = len(words_)

    count = Counter(words_)
    freq["unique"] = len([word for word, num in count.items() if num == 1]) / n
    freq["emoji"] = emoji.emoji_count(string_text) / n
    freq["quotes"] = (len(re.findall('\"[^\"]+\"', string_text)) +
                      len(re.findall('\'[^\']+\'', string_text))) / n  # quotes -> " " oppure ' '
    # la stessa lettera 3 o più volte consecutivamente
    freq["stretch"] = len(re.findall('([a-z])\\1\\1{1,}', string_text)) / n
    freq["proper_nouns"] = len([ent for sent in doc.sentences for ent in sent.ents]) / n  # NER

    count_char = 0
    count_complex = 0
    count_verb = 0
    count_aux = 0
    count_adj = 0
    count_noun = 0
    count_conj = 0
    count_adv = 0
    count_def_art = 0
    count_indef_art = 0
    count_pron = 0
    count_num = 0
    count_sp_chars = 0
    count_punct = 0
    count_comma = 0
    count_colon = 0
    count_semicomma = 0
    count_exclam = 0
    count_quest = 0
    count_up_case = 0
    count_start_up_case = 0
    count_hashtag = 0
    count_at = 0
    count_1_sing = 0
    count_1_plur = 0
    count_2 = 0
    count_3_sing_m = 0
    count_3_sing_f = 0
    count_3_plur = 0

    for word in words:
        count_char += len(word.text)  # numero di caratteri

        if len(word.text) > 5:  # complex
            count_complex += 1

        if word.pos == "VERB":  # verbs
            count_verb += 1

        if word.pos == "AUX":  # auxiliar
            count_aux += 1

        if word.pos == "ADJ":  # adjective
            count_adj += 1

        if word.pos == "NOUN":  # nooun
            count_noun += 1

        if word.pos == "CCONJ" or word.pos == "SCONJ":  # congiunzioni
            count_conj += 1

        if word.pos == "ADV":  # avverbi
            count_adv += 1

        if word.pos == "DET":  # articoli
            try:
                properties = extract_properties(word.feats)
                if "PronType" in properties.keys() and "Definite" in properties.keys():
                    if properties["PronType"] == "Art":
                        if properties["Definite"] == "Def":
                            count_def_art += 1

                        if properties["Definite"] == "Ind":
                            count_indef_art += 1
            except AttributeError:
                continue

        if word.pos == "PRON":  # pronomi
            count_pron += 1
            try:
                properties = extract_properties(word.feats)
                if "PronType" in properties.keys() and "Number" in properties.keys() and "Gender" in properties.keys():
                    if properties["PronType"] == "Prs":
                        if properties["Person"] == "1":
                            if properties["Number"] == "Sing":
                                count_1_sing += 1
                            if properties["Number"] == "Plur":
                                count_1_plur += 1

                        if properties["Person"] == "2":
                            count_2 += 1

                        if properties["Person"] == "3":
                            if properties["Number"] == "Sing":
                                if properties["Gender"] == "Masc":
                                    count_3_sing_m += 1
                                if properties["Gender"] == "Fem":
                                    count_3_sing_f += 1
                            if properties["Number"] == "Plur":
                                count_3_plur += 1
            except AttributeError:
                continue

        if word.pos == "NUM":
            count_num += 1

        # special characters = cioè che non è né carattere né cifra
        for char in word.text:
            if not char.isdigit() and not char.isalpha():
                count_sp_chars += 1

        if word.pos == "PUNCT":
            count_punct += 1

        if word.text == ",":
            count_comma += 1

        if word.text == ":":
            count_colon += 1

        if word.text == ";":
            count_semicomma += 1

        if word.text == "!":
            count_exclam += 1

        if word.text == "?":
            count_quest += 1

        if word.text.isupper():  # tutto maiuscolo
            count_up_case += 1

        if word.text[0].isupper() and word.text[1:].islower():  # Iniziale maiuscola
            count_start_up_case += 1

        count_hashtag += word.text.count("#HASHTAG#")

        count_at += word.text.count("@")

    freq["characters"] = count_char / n
    freq["complex"] = count_complex / n
    freq["verbs"] = count_verb / n
    freq["aux"] = count_aux / n
    freq["adj"] = count_adj / n
    freq["nouns"] = count_noun / n
    freq["conj"] = count_conj / n
    freq["adv"] = count_adv / n
    freq["def_art"] = count_def_art / n
    freq["indef_art"] = count_indef_art / n
    freq["pron"] = count_pron / n
    freq["num"] = count_num / n
    freq["sp_chars"] = count_sp_chars / n
    freq["punct"] = count_punct / n
    freq["commas"] = count_comma / n
    freq["colon"] = count_colon / n
    freq["semicommas"] = count_semicomma / n
    freq["exclam"] = count_exclam / n
    freq["quest"] = count_quest / n
    freq["up_case"] = count_up_case / n
    freq["start_up_case"] = count_start_up_case / n
    freq["hashtag"] = count_hashtag / n
    freq["at"] = count_at / n
    freq["1_sing"] = count_1_sing / n
    freq["1_plur"] = count_1_plur / n
    freq["2_pron"] = count_2 / n
    freq["3_sing_m"] = count_3_sing_m / n
    freq["3_sing_f"] = count_3_sing_f / n
    freq["3_plur"] = count_3_plur / n

    return freq



def flatten_list(sentences):
    return [word for sent in sentences for word in sent.words]


def embed(input, model):
    return model(input)


def extract_properties(features):
    list_feats = features.split("|")
    dict_feat = dict((feature.split("=")[0], feature.split("=")[1]) for feature in list_feats)
    return dict_feat


'''
Preparazione file e librerie necessarie
'''

def main():

    # Personality
    exec(open("pyGen.py").read())

    # Embeddings
    os.environ["TFHUB_CACHE_DIR"] = './tfhub'
    model_embedding = hub.load(os.environ["TFHUB_CACHE_DIR"])

    # Emotions
    path_excel = "NRC-Emotion-Lexicon.xlsx"
    # usa colonne inglese, spagnolo, e 8 sentimenti
    nrc_df = pd.read_excel(path_excel, usecols="A,CI,DD:DK")
    nrc_df.rename(columns={"Spanish (es)": "es",
                           "English (en)": "en"}, inplace=True)

    # tutte le parole in minuscolo
    nrc_df["es"] = pd.Series(map(lambda x: str(x).lower(), nrc_df["es"]))
    nrc_df["en"] = pd.Series(map(lambda x: str(x).lower(), nrc_df["en"]))

    users_feats_test = list()  # list for all users and relative features
    output_xml = list()  # list of [author id, lang] for xml output
    
    # Estrazione features en users
    dataset_path = 0
    outputDir = 0
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'i:o:')

    except getopt.GetoptError as err:
        # print help information and exit:
        # # print(err)  # will print something like "option -a not recognized"
        sys.exit(2)

    for param, a in optlist:

        if param == "-i":
            dataset_path = a

        elif param == "-o":
            outputDir = a

        else:
            assert False, "unhandled option"

    print("Input Dataset", dataset_path)
    print("OutputDir", outputDir)

####################################
    #dataset_path = "./tweets"
    #outputDir = "./outputdir"
####################################

    # Estrazione features en users
    #nlp = stanza.Pipeline('en', dir="F:/stanza_resources", processors={"tokenize": "spacy", 'ner': 'conll03'}, use_gpu=True)  # uso conll03
    nlp = stanza.Pipeline('en', processors={"pos": "default", 'ner': 'conll03',"tokenize": "spacy",  }, use_gpu=True)  # uso conll03
    extract_features(dataset_path + "/en/*.xml", model_embedding, nlp, users_feats_test, output_xml, nrc_df)

    # Estrazione features es users
    #nlp = stanza.Pipeline("es", dir="F:/stanza_resources", processors='pos,ner,tokenize,lemma,mwt', use_gpu=True)
    nlp = stanza.Pipeline("es", processors='pos,ner,tokenize,lemma,mwt', use_gpu=True)
    extract_features(dataset_path + "/es/*.xml", model_embedding, nlp, users_feats_test, output_xml, nrc_df)

    # Predict
    with open('svm_trained.pickle', 'rb') as handle:
        my_svm = pickle.load(handle)

    predictions = my_svm.predict(users_feats_test)

    # Output XML
    # creo cartella, in più crea una cartella "en" e "es"
    os.makedirs(outputDir + "/en")
    os.makedirs(outputDir + "/es")

    for i in range(0, len(users_feats_test)):
        # for i in len() output_xml[i] 0 e 1 e predict[i] e a dx dell'uguale metto i parametri
        root = ET.Element("author", id=output_xml[i][0], lang=output_xml[i][1], type=predictions[i])
        tree = ET.ElementTree(root)
        # compongo stringa con author id + .xml e scrivo in una cartella en o es dipende
        tree.write(outputDir + "/" + output_xml[i][1] + "/" + output_xml[i][0] + ".xml")
        print(ET.tostring(root).decode())


if __name__ == '__main__':
    main()
