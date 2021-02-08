#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz
 

import sys
import argparse
import os
import json
import re
import spacy
import html


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)


def preproc1(comment , steps=range(1, 6)):
    """ This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    """
    modComm = comment
    if 1 in steps:  
        #modify this to handle other whitespace chars.
        #replace newlines with spaces
        modComm = re.sub(r"\s+", " ", modComm)

    if 2 in steps:  # unescape html
        modComm = html.unescape(modComm)

    if 3 in steps:  # remove URLs
        # if find anything have but not end with http or www, remove it
        modComm = re.sub(r"(http|www)\S+", "", modComm)
        
    if 4 in steps: #remove duplicate spaces.
        modComm = re.sub(r" {2,}", " ", modComm)
        # one more important thing,
        # if we found more spaces in the end of comment, we should remove it instead of make it 1 space.
        modComm = re.sub(r" $", "", modComm)
    if 5 in steps:
        output = ""
        utt = nlp(modComm)
        for sent in utt.sents:
            # print(sent.text)
            for token in sent:
                origin = token.text
                lemma = token.lemma_
                tag = token.tag_

                lemma = re.sub(" ", "-", lemma) # to handle the "going to" problem mentioned in piazza
                if lemma[0] == "-":
                    # no -PRON-
                    output = output + origin
                else:
                    x = re.search("^[A-Z]+$", origin)
                    if x:
                        # means the text is all uppercase, lemma should be uppercase as well
                        output = output + lemma.upper()
                    else:
                        # else we want it in lower cases
                        output = output + lemma.lower()
                output = output + "/" + tag + " "
            output += "\n"

        modComm = output
        # Make sure to:
        #    * Insert "\n" between sentences.   Done
        #    * Split tokens with spaces.        Done
        #    * Write "/POS" after each token.   Done
        #    * the hyphen problem               Done
        #    * the upper case problem of lemma  Done
            #   All Good! :)
            
    
    return modComm


def main(args):
    allOutput = []
    for subdir, dirs, files in os.walk(indir):      # where is this indir come from??? that one in __main__?
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))

            lines = int(args.max)                   # This represent how many comments this program will go through
            staring_line = 1003812966 % len(data)   # This represent where we start, depending on student id

            for lineIndex in range(staring_line, staring_line+lines):
                if lineIndex >= len(data):
                    lineIndex -= len(data) # circular the index

                # read from the data and grab the data we need
                j = json.loads(data[lineIndex])
                commentId = j['id']
                commentBody = j['body']
                commentCat = str(file)

                # ok that's all we want, now we need to build a new dic for the data
                output = {'id': commentId, 'body': preproc1(commentBody), 'cat': commentCat}

                allOutput.append(output)
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()
    # That's all for part1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", help="The directory for A1. Should contain subdir data. Defaults to the directory for A1 on cdf.", default='/u/cs401/A1')
    
    args = parser.parse_args()

    if args.max > 200272:
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    main(args)
