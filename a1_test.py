import re
import html
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

def preproc_x(comment):
    modComm = comment
    modComm = html.unescape(modComm)
    return modComm

def spacy_test():
    utt = nlp(u"I know words. I have the best words")
    for sent in utt.sents:
        print(sent.text)
        for token in sent:
            print(token.text, token.lemma_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

if __name__ == "__main__":
    # comment = "tower&#039s"
    # modC = preproc_x(comment)
    # print(modC)
    spacy_test()