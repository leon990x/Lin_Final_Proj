from sklearn.datasets import load_files
import nltk
#nltk.download('stopwords')
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
#from gensim.models import Word2Vec
#from gensim.models import KeyedVectors
import pandas as pd
data = "textsv2.csv"
d = pd.read_csv(data)
df = pd.DataFrame(d)
s = pd.Series(df["article_text"])
s = list(s)

all_texts = []
for t in s:
    all_texts.append(t)

import numpy as np

import nltk
#nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from sklearn.metrics.pairwise import cosine_similarity
import re
import networkx as nx
from rouge_score_edited import rouge_scorer

#h_sum is our reference summaries for each set of documents. These are extracted from the canvas dataset.

# h_sum = "Prospects were dim for resolution of the political crisis in Cambodia in October 1998.\r\nPrime Minister Hun Sen insisted that talks take place in Cambodia while opposition leaders Ranariddh and Sam Rainsy, fearing arrest at home, wanted them abroad.\r\nKing Sihanouk declined to chair talks in either place.\r\nA U.S. House resolution criticized Hun Sen\'s regime while the opposition tried to cut off his access to loans.\r\nBut in November the King announced a coalition government with Hun Sen heading the executive and Ranariddh leading the parliament.\r\nLeft out, Sam Rainsy sought the King\'s assurance of Hun Sen\'s promise of safety and freedom for all politicians.\r\nCambodian prime minister Hun Sen rejects demands of 2 opposition parties for talks in Beijing after failing to win a 2/3 majority in recent elections.\r\nSihanouk refuses to host talks in Beijing.\r\nOpposition parties ask the Asian Development Bank to stop loans to Hun Sen\'s government.\r\nCCP defends Hun Sen to the US Senate.\r\nFUNCINPEC refuses to share the presidency.\r\nHun Sen and Ranariddh eventually form a coalition at summit convened by Sihanouk.\r\nHun Sen remains prime minister, Ranariddh is president of the national assembly, and a new senate will be formed.\r\nOpposition leader Rainsy left out.\r\nHe seeks strong assurance of safety should he return to Cambodia.\r\nCambodia King Norodom Sihanouk praised formation of a coalition of the Countries top two political parties, leaving strongman Hun Sen as Prime Minister and opposition leader Prince Norodom Ranariddh president of the National Assembly.\r\nThe announcement comes after months of bitter argument following the failure of any party to attain the required quota to form a government.\r\nOpposition leader Sam Rainey was seeking assurances that he and his party members would not be arrested if they return to Cambodia.\r\nRainey had been accused by Hun Sen of being behind an assassination attempt against him during massive street demonstrations in September.\r\nCambodian elections, fraudulent according to opposition parties, gave the CPP of Hun Sen a scant majority but not enough to form its own government.\r\nOpposition leaders fearing arrest, or worse, fled and asked for talks outside the country.\r\nHan Sen refused.\r\nThe UN found evidence of rights violations by Hun Sen prompting the US House to call for an investigation.\r\nThe three-month governmental deadlock ended with Han Sen and his chief rival, Prince Norodom Ranariddh sharing power.\r\nHan Sen guaranteed safe return to Cambodia for all opponents but his strongest critic, Sam Rainsy, remained wary.\r\nChief of State King Norodom Sihanouk praised the agreement."

h_sum = "Hurricane Mitch approached Honduras on Oct. 27, 1998 with winds up to 180mph a Category 5 storm.\r\nIt hit the Honduran coast on Oct. 28 bringing downpours that forced large-scale evacuations.\r\nOn Nov. 1 Nicaragua announced collapse of a drenched volcano crater killing about 2,000.\r\nBy then Mitch\'s winds were down to 30mph, but as disaster reports poured in the death toll finally exceeded 10,000 and half a million left homeless.\r\nThe European Union, international relief agencies, Mexico, the U.S., Japan, Taiwan, the U.K. and U.N. sent financial aid, relief workers and supplies.\r\nPope John Paul II appealed to \"all public and private institutions\" to help.\r\nHonduras braced as category 5 Hurricane Mitch approached.\r\nSlow-moving Mitch battered the Honduran coast for 3 days.\r\nHonduran death estimates grew from 32 to 231 in the first days, to 6,076, with 4,621 missing.\r\nAbout 2,000 were killed in Nicaragua, 239 in El Salvador, 194 in Guatemala, 6 in southern Mexico and 7 in Costa Rica.\r\nThe EU approved 6.4 million ecu in aid to Mitch\'s victims.\r\nThe Pope appealed for aid.\r\nThe US boosted aid to $70 million.\r\nA id workers struggled to reach survivors in danger.\r\nHurricane winds, rain and floods caused massive damage to homes, businesses, roads and bridges.\r\nLatest reports estimate over 10,000 killed in Central America.\r\nHurricane Mitch, category 5 hurricane, brought widespread death and destruction to Central American.\r\nEspecially hard hit was Honduras where an estimated 6,076 people lost their lives.\r\nThe hurricane, which lingered off the coast of Honduras for 3 days before moving off, flooded large areas, destroying crops and property.\r\nThe U.S. and European Union were joined by Pope John Paul II in a call for money and workers to help the stricken area.\r\nPresident Clinton sent Tipper Gore, wife of Vice President Gore to the area to deliver much needed supplies to the area, demonstrating U.S. commitment to the recovery of the region.\r\nA category 5 storm, Hurricane Mitch roared across the northwest Caribbean with 180 mph winds across a 350-mile front that devastated the mainland and islands of Central America.\r\nAlthough the force of the storm diminished, at least 8,000 people died from wind, waves and flood damage.\r\nThe greatest losses were in Honduras where some 6,076 people perished.\r\nAround 2,000 people were killed in Nicaragua, 239 in El Salvador, 194 in Guatemala, seven in Costa Rica and six in Mexico.\r\nAt least 569,000 people were homeless across Central America.\r\nAid was sent from many sources (European Union, the UN, US and Mexico).\r\nRelief efforts are hampered by extensive damage."

# h_sum = "On Oct. 16, 1998 British police arrested former Chilean dictator Pinochet on a Spanish warrant charging murder of Spaniards in Chile, 1973-1983.\r\nFidel Castro denounced the arrest.\r\nThe Chilean government protested strongly.\r\nWhile the British government defended the arrest, it and the Spanish government took no stand on extradition of Pinochet to Spain, leaving it to the courts.\r\nChilean legislators lobbied in Madrid against extradition, while others endorsed it.\r\nThen new charges were filed for crimes against Swiss and French citizens.\r\nPinochet\'s wife and family pleaded that he was too sick to face extradition.\r\nAs of Oct. 28 the matter was not resolved.Pinochet arrested in London on Oct. 16 at a Spanish judge\'s request for atrocities against Spaniards in Chile during his rule.\r\nCastro, Chilean legislators and Pinochet\'s lawyers protested and claimed he had diplomatic immunity.\r\nHis wife asked for his release because he was recovering from recent back surgery.\r\nPinochet visited Thatcher before his surgery.\r\nThe British and Spanish governments defended the arrest, saying it was strictly a legal matter.\r\nThe EC president hoped Pinochet would stand trial.\r\nNone of his Swiss accounts have been frozen yet.\r\nThe Swiss government also asked for his arrest for the 1977 disappearance of a Swiss-Chilean student.\r\nFormer Chilean dictator Augusto Pinochet has been arrested in London at the request of the Spanish government.\r\nPinochet, in London for back surgery, was arrested in his hospital room.\r\nSpain is seeking extradition of Pinochet from London to Spain to face charges of murder in the deaths of Spanish citizens in Chile under Pinochet\'s rule in the 1970s and 80s.\r\nThe arrest raised confusion in the international community as the legality of the move is debated.\r\nPinochet supporters say that Pinochet\'s arrest is illegal, claiming he has diplomatic immunity.\r\nThe final outcome of the extradition request lies with the Spanish courts.\r\nBritain caused international controversy and Chilean turmoil by arresting former Chilean dictator Pinochet in London for Spain\'s investigation of Spanish citizen deaths under Pinochet\'s 17-year rule of torture and political murder.\r\nClaims are Pinochet had diplomatic immunity, extradition is international meddling or illegal because Pinochet is not a Spanish citizen, also his crimes should be punished.\r\nSpain and Britain, big Chilean investors, fear damage to economic relations and let courts decide extradition.\r\nThe Swiss haven\'t investigated Pinochet accounts despite a Spanish request.\r\nPinochet is shielded from details, said too sick to be extradited.\r\n"

def formatter(summary):
    l_sum = sent_tokenize(summary)
    fmt_summary = ""
    for s in l_sum:
        fmt_summary += (s + "\n")
    return fmt_summary

#Textrank implementation: returns ROUGE scores for each corpus when compared to its corresponding h_sum (human summary)
def textrank_summarizer(data, score_select):
    #Textrank Implementation
    df = pd.read_csv(data, encoding='utf-8')

    #Tokenize sentences
    from nltk.tokenize import sent_tokenize
    sentences = []
    for s in df['article_text']:
        sentences.append(sent_tokenize(s))
    sentences = [y for x in sentences for y in x] # flatten list

    #Extracting word vectors (from wikipedia + Gigaword 5 GloVe database)
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    len(word_embeddings)

    #Pre-Processing ()
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]
    def remove_stopwords(sent):
        sent_new = " ".join([i for i in sent if i not in stop_words])
        return sent_new

    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    # Extract top 10 sentences as the summary
    full_sum = ""
    for i in range(len(ranked_sentences)//score_select):
        full_sum += ranked_sentences[i][1]

    textfile = open('Generated_summaries.txt', 'a')
    textfile.write("N = "+str(score_select)+"\n")
    textfile.write(full_sum)
    textfile.write("\n")
    textfile.close()

    full_sum = formatter(full_sum)
    #Py-Rouge evaluation


    tr_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeLsum'], use_stemmer=True)
    tr_scores = tr_scorer.score(full_sum, h_sum)

    return [tr_scores['rougeLsum']['F_Measure'], tr_scores['rougeLsum']['Precision'], tr_scores['rougeLsum']['Recall']]


#Control panel: used to adjust N value for each data run
tr_fmeasures = []
tr_precisions = []
tr_recalls = []


for i in range(1, 2):
    score_tr = textrank_summarizer(data, i)
    tr_fmeasures.append(score_tr[0])
    tr_precisions.append(score_tr[1])
    tr_recalls.append(score_tr[2])

df_tr = pd.DataFrame(list(zip(tr_fmeasures, tr_precisions, tr_recalls)), columns=["[TR] F Measure", "[TR] Precision", "[TR] Recall"])
print(df_tr)
#df_tr.to_csv('tr_14_hurricane.csv', encoding='utf-8')
