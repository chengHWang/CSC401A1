#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import numpy as np
import argparse
import json
import re
import pandas as pd
import os

path = "/u/cs401/A1/feats"



# Provided wordlists.
# For F2
FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
# For F3
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
# For F4
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
# For F14
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}

"""
Plan:
we have 173+1 features need to handle in Part2.

extract1:
1. Features 1-17 are easiest part, mostly counting problems. => extractFGroup1()
2. Features 18-29 are statistic, need to use the norm in u/cs401/wordlist => extractFGroup2()

extract2:
3. Features 30-173 are provided, just download and paste to our NumPy array, in u/cs401/A1/feats

main:
4. Feature 174(not really actually) for the class: 0: Left, 1: Center, 2: Right, 3: Alt 
"""


def extract1(comment):
    """ This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    """

    features = np.zeros(173)
    lower_comment, valid_words, features = extractFGroup1(comment, features)    # Features 1-17 will be filled
    features = extractFGroup2(valid_words, features)    # Features 18-29 will be filled

    return features # And will be handled by extract2() immediately


def extractFGroup1(comment, features):
    assert features[0] == 0 # just in case
    features[0] = len(re.findall(regex_list[0], comment))

    def replacement(match):
        return match.group().lower()

    lower_comment = re.sub(r"[a-zA-Z]+/", replacement, comment)

    for i in range(2, 15):
        founded_match = re.findall(regex_list[i-1], lower_comment)
        features[i-1] = len(founded_match)

    sentences = comment.split("\n")[:-1]    # idk why but there will be a "" at the end after split
    sentences_num = len(sentences)
    features[16] = sentences_num

    token_regex = r"\S+/\S+"
    tokens = re.findall(token_regex, lower_comment)
    token_num = len(tokens)
    if sentences_num > 0:
        features[14] = token_num/sentences_num

    punctuation_regex = r"([\#\$\!\?\.\:\;\(\)\"\',\[\]/]{1,}/\S+)"
    valid_token = []
    for t in tokens:
        if re.match(punctuation_regex, t) is None:
            valid_token.append(t)

    valid_token_num = len(valid_token)
    valid_token_total_length = 0
    valid_words = []
    for v_t in valid_token:
        split_v_t = v_t.split("/")
        valid_token_total_length += len(split_v_t[0])
        valid_words.append(split_v_t[0])
    if valid_token_num > 0:
        features[15] = valid_token_total_length/valid_token_num

    return lower_comment, valid_words, features


# This global list storing all the regex for extractFGroup1, entry i for Feature i+1
regex_list = [
    r"([A-Z]{3,})/",                                                # No.1, index 0
    r"(?<=\b)(" + "|".join(FIRST_PERSON_PRONOUNS) + ")(?=\/)",
    r"(?<=\b)(" + "|".join(SECOND_PERSON_PRONOUNS) + ")(?=\/)",
    r"(?<=\b)(" + "|".join(THIRD_PERSON_PRONOUNS) + ")(?=\/)",
    r"(/CC)(?=\b)",                                                 # No.5, index 4
    r"(/VBD)(?=\b)",
    r"(?<=\b)((will\/MD [a-zA-Z]+\/VB)|(go\/VB. to\/TO [a-zA-Z]+\/VB))(?=\b)",
    r",/,",
    r'[-/()!"+,\'&]{2,}',                                           # No.9, index 8
    # I'm not sure about this, sometimes spacy will separate punctuations but sometimes not.
    r"(/NN|/NNS)(?=\b)",
    r"(/NNP|/NNPS)(?=\b)",
    r"(/RB|/RBR|/RBS)(?=\b)",
    r"(/WDT\b|/WP\$\W|/WRB\b|/WP\b)",
    r"(\b|^)(" + "|".join(SLANG) + ")(/[.]{0,4})"                   # No.14, index 13
]

def flatten(arr):
    new_arr = []
    for a in arr:
        if isinstance(a, pd.Series):
            new_arr.extend(a.values)
        else:
            new_arr.append(a)
    return new_arr


wordlists_dir = "/u/cs401/Wordlists"
def extractFGroup2(v_words, features):
    # Remote
    bgl_norm = pd.read_csv(os.path.join(wordlists_dir, "BristolNorms+GilhoolyLogie.csv"), usecols=["WORD", "AoA (100-700)", "IMG", "FAM"])
    rating_norm = pd.read_csv(os.path.join(wordlists_dir, "Ratings_Warriner_et_al.csv"), usecols=["Word", "V.Mean.Sum", "D.Mean.Sum", "A.Mean.Sum"])

    # Local
    # bgl_norm = pd.read_csv("BristolNorms+GilhoolyLogie.csv", usecols=["WORD", "AoA (100-700)", "IMG", "FAM"])
    # rating_norm = pd.read_csv("Ratings_Warriner_et_al.csv", usecols=["Word", "V.Mean.Sum", "D.Mean.Sum", "A.Mean.Sum"])

    bgl_words = bgl_norm.set_index(['WORD'])
    bgl_value_list = []
    rating_words = rating_norm.set_index(['Word'])
    rating_value_list = []

    # get rid of tokens not in the csv
    for v in v_words:
        try:
            bgl_value_list.append(bgl_words.loc[v])
        except:
            pass
    for v in v_words:
        try:
            rating_value_list.append(rating_words.loc[v])
        except:
            pass

    # AOA
    AoA = []
    for x in bgl_value_list:
        w = x.get("AoA (100-700)", np.nan)
        AoA.append(w)
    AoA = flatten(AoA)
    if AoA:
        # print("AoA is not empty")
        features[17] = np.nanmean(AoA)
        features[20] = np.nanstd(AoA)

    # IMG
    IMG = []
    for x in bgl_value_list:
        w = x.get("IMG", np.nan)
        IMG.append(w)
    # print(IMG)
    IMG = flatten(IMG)
    if IMG:
        features[18] = np.nanmean(IMG)
        features[21] = np.nanstd(IMG)

    # FAM
    FAM = []
    for x in bgl_value_list:
        w = x.get("FAM", np.nan)
        FAM.append(w)
    # print(FAM)
    FAM = flatten(FAM)
    if FAM:
        features[19] = np.nanmean(FAM)
        features[22] = np.nanstd(FAM)

    # VMS
    VMS = []
    for x in rating_value_list:
        w = x.get("V.Mean.Sum", np.nan)
        VMS.append(w)
    VMS = flatten(VMS)
    if VMS:
        features[23] = np.nanmean(VMS)
        features[26] = np.nanstd(VMS)

    # AMS
    AMS = []
    for x in rating_value_list:
        w = x.get("A.Mean.Sum", np.nan)
        AMS.append(w)
    AMS = flatten(AMS)
    if AMS:
        features[24] = np.nanmean(AMS)
        features[27] = np.nanstd(AMS)

    # DMS
    DMS = []
    for x in rating_value_list:
        w = x.get("D.Mean.Sum", np.nan)
        DMS.append(w)
    DMS = flatten(DMS)
    if DMS:
        features[25] = np.nanmean(DMS)
        features[28] = np.nanstd(DMS)

    # Features 18-29, index 17-28 complete
    return features

categories = ['Left', 'Center', 'Right', 'Alt']
def LIWC_dic_generator():
    print("start building LIWC library")
    direc = '/u/cs401/A1/feats/'

    ids = "_IDs.txt"
    featFile = "_feats.dat.npy"
    dictionaries= []
    for i in range(len(categories)): # i: 0-3
        cat = categories[i]
        print("Running " + cat)

        # path to id file
        f = open(direc + cat + ids)
        # a list of all lines in the id file
        lines = f.readlines()

        lst = [x.split() for x in lines]
        temp = [(lst[i], i) for i in range(len(lst))]
        dic = {temp[i][1] : temp[i][0] for i in range(len(temp))}
        feats = np.load(direc + cat + featFile, 'r')
        final = {dic[k][0]: feats[k] for k in range(len(temp))}
        dictionaries.append(final)

        f.close()
    return dictionaries
LIWC_dic = LIWC_dic_generator()
# I make it global so it only need to run 1 time, and increase the program's efficiency a lot.

def extract2(feat, comment_class, comment_id):
    """ This function adds features 30-173 for a single comment.

    Parameters:
        feat: np.array of length 173, index 0-28 is filled.
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment

    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this
        function adds feature 30-173). This should be a modified version of
        the parameter feats.
    """
    assert feat[29] == 0    # just in case
    cat_num = categories.index(comment_class)
    feat[29:173] = LIWC_dic[cat_num][comment_id]
    return feat


def main(args):
    #Declare necessary global variables here.
    #Load data
    data = json.load(open(args.input))
    feats = np.zeros((len(data), 173+1))

    for i, comment in enumerate(data):
        # check if the program is still running
        if (i + 1) % 1000 == 0:
            print(f"step: '{i + 1}'")

        features = extract1(comment['body'])

        # This features only have 173 slots, and only Slot 0-28 filled yet.
        features = extract2(features, comment['cat'], comment['id'])

        # Convert features to feats and filled the last slot of cat
        feats[i, :-1]=features

        # Fill in the last entry to represent the categories
        feats[i, -1] = categories.index(comment['cat'])

    np.savez_compressed(args.output, feats)
    # end
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", help="Path to csc401 A1 directory. By default it is set to the cdf directory for the assignment.", default="/u/cs401/A1/")
    args = parser.parse_args()        

    main(args)

