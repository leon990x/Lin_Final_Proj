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


#BASELINE set 2
# h_sum = "Honduras braced for potential catastrophe Tuesday as Hurricane Mitch roared through the northwest Caribbean , churning up high waves and intense rain that sent coastal residents scurrying for safer ground .\r\nPresident Carlos Flores Facusse declared a state of maximum alert and the Honduran military sent planes to pluck residents from their homes\r\non islands near the coast .\r\nAt 0900 GMT Tuesday , Mitch was 95 miles ( 152 kilometers ) north of Honduras , near the Swan Islands .\r\nWith winds near 180 mph ( 289 kph ) , and even higher gusts , it was a Category 5 monster _ the highest , most dangerous rating for a storm .\r\nThe 350-mile ( 560-kilometer ) wide hurricane was moving west at 8 mph ( 12 kph ) .\r\n` ` Frightened people are moving into the mountains to search for shelter,\' \' he said .\r\nIn El Progreso , 100 miles ( 160 kilometers ) north of the Honduran capital of Tegucigalpa , the army evacuated more than 5,000 people who live in low-lying banana plantations along the Ulua River , said Nolly Soliman , a resident .\r\nBefore bearing down on Honduras , Mitch swept past Jamaica and the Cayman Islands .\r\nRain squalls flooded streets in the Jamaican capital , Kingston , and government offices and schools closed in the Caymans , a British colony of 28,000 people .\r\nThe strongest hurricane to hit Honduras in recent memory was Fifi in 1974 , which ravaged Honduras \' Caribbean coast , killing at least 2,000 people .Hurricane Mitch paused in its whirl through the western Caribbean on Wednesday to punish Honduras with 120-mph ( 205-kph ) winds , topping trees , sweeping away bridges , flooding neighborhoods and killing at least 32 people .\r\nMitch was drifting west at only 2 mph ( 3 kph ) over the Bay Islands , Honduras \' most popular tourist area .\r\nIt also was only 30 miles ( 50 kms ) off the coast , and hurricane-force winds stretched outward 105 miles ( 165 kms ) ; tropical storm-force winds 175 miles ( 280 kms ) .\r\nThat meant the Honduran coast had been under hurricane conditions for more than a day .\r\n` ` The hurricane has destroyed almost everything,\' \' said Mike Brown , a resident of Guanaja Island which was within miles ( kms ) of the eye of the hurricane .\r\n` ` Few houses have remained standing .\r\n\' \' At its , 4th graf pvsHurricane Mitch cut through the Honduran coast like a ripsaw Thursday , its devastating winds whirling for a third day through resort islands and mainland communities .\r\nAt least 32 people were killed and widespread flooding prompted more than 150,000 to seek higher ground .\r\nMitch , once among the century \'s most powerful hurricanes , weakened today as it blasted this Central American nation , bringing downpours that flooded at least 50 rivers .\r\nIt also kicked up huge waves that pounded seaside communities .\r\nThe storm \'s power was easing and by 1200 GMT , it had sustained winds of 80 mph ( 130 kph ) , down from 100 mph ( 160 kph ) around midnight and well below its 180 mph ( 290 kph ) peak of early Tuesday .\r\nHouston accountant Kathy Montgomery said that she and her friend Nina Devries had tried to leave Cancun but found all the flights full .\r\n` ` It \'s been horrible,\' \' said Montgomery , as she and her friend drank cocktails at an outdoor restaurant .\r\n` ` We could n\'t go out on a boat , we could n\'t go snorkeling .\r\n` ` Even Carlos \' N Charlie \'s and Senor Frog \'s are closed,\' \' she said dejectedly , referring to two restaurants .\r\n` ` Some vacation .At least 231 people have been confirmed dead in Honduras from former-hurricane Mitch , bringing the storm \'s death toll in the region to 357 , the National Emergency Commission said Saturday .\r\nMitch _ once , 2nd graf pvsIn Honduras , at least 231 deaths have been blamed on Mitch , the National Emergency Commission said Saturday .\r\nEl Salvador _ where 140 people died in flash floods _ declared a state of emergency Saturday , as did Guatemala , where 21 people died when floods swept away their homes .\r\nMexico reported one death from Mitch last Monday .\r\nIn the Caribbean , the U.S. Coast Guard widened a search for a tourist schooner with 31 people aboard that has n\'t been heard from since Tuesday .\r\nBy late Sunday , Mitch \'s winds , once near 180 mph ( 290 kph ) , had dropped to near 30 mph ( 50 kph ) , and the storm _ now classified as a tropical depression _ was near Tapachula , on Mexico \'s southern Pacific coast near the Guatemalan border .\r\nMitch was moving west at 8 mph ( 13 kph ) and was dissipating but threatened to strengthen again if it moved back out to sea .Nicaraguan Vice President Enrique Bolanos said Sunday night that between 1,000 and 1,500 people were buried in a 32-square mile ( 82.88 square-kilometer ) area below the slopes of the Casita volcano in northern Nicaragua .\r\nThat is in addition to least another 600 people elsewhere in the country , Bolanos said .BRUSSELS , Belgium ( AP ) - The European Union on Tuesday approved 6.4 million European currency units ( dlrs 7.7 million ) in aid for thousands of victims of the devastation caused by Hurricane Mitch in Central America .\r\nEU spokesman Pietro Petrucci said the funds will be used to provide basic care such as medicine , food , water sanitation and blankets to thousands of people whose homes were destroyed by torrential rains and mudslides .\r\nThe aid will be distributed in Nicaragua , El Salvador , Honduras and Guatemala which have most suffered from Mitch \'s deadly passage , the EU executive Commission said in a statement .\r\nOfficials in Central America estimated Tuesday that about 7,000 people have died in the region .\r\nThe greatest losses were reported in Honduras , where an estimated 5,000 people died and 600,000 people _ 10 percent of the population _ were forced to flee their homes after last week \'s storm .\r\nEl Salvador \'s National Emergency Committee listed 174 dead , 96 missing and 27,000 homeless .\r\nBut its own regional affiliate in San Miguel province reported 125 dead there alone .\r\nGuatemala reported 100 storm-related deaths .\r\nThe latest EU aid follows an initial 400,000 ecu ( dlrs 480,000 ) .\r\nthe EU approved for the region on Friday .\r\nThe full 6.8 million ecu ( dlrs 8.18 million ) will be channeled through humanitarian groups working in the region .Pope John Paul II appealed for aid Wednesday for the Central American countries stricken by hurricane Mitch and said he feels close to the thousands who are suffering .\r\nSpeaking during his general audience , the pope urged ` ` all public and private institutions and all men of good will\' \' to do all they can ` ` in this grave moment of destruction and death .\r\n\' \' Hurricane Mitch killed an estimated 9,000 people throughout Central America in a disaster of such proportions that relief agencies have been overwhelmed .\r\nAmong those attending the audience were six Russian cosmonauts taking a special course in Italy .\r\nAs a gift , they gave John Paul a spacesuit .Better information from Honduras \' ravaged countryside enabled officials to lower the confirmed death toll from Hurricane Mitch from 7,000 to about 6,100 on Thursday , but leaders insisted the need for help was growing .\r\nPresident Carlos Flores declared Hurricane Mitch had set back Honduras \' development by 50 years .\r\nHe urged the more than 1.5 million Hondurans affected by the storm to help with the recovery effort .\r\n` ` The county is semi-destroyed and awaits the maximum effort and most fervent and constant work of every one of its children,\' \' he said .\r\nIn the capital , Tegucigalpa , Mexican rescue teams began searching for avalanche victims .\r\nHonduran doctors dispensed vaccinations to prevent disease outbreaks in shelters crammed with refugees .\r\n` ` This is the first place we \'ve been\' \' with the dogs , Honduran army Maj. Freddy Diaz Celaya said .\r\n` ` From here we \'ll continue searching downriver .\r\n\' \' Concerned that crowded shelter conditions could produce outbreaks of hepatitis , respiratory infections and other ailments , the Health Ministry announced an inoculation campaign , especially for children .\r\nDoctors volunteering at a shelter housing 4,000 people at Tegucigalpa \'s Polytechnic Development Institute said they \'d heard of the campaign but had yet to receive word or medicines from the Health Ministry .\r\n` ` We have to vaccinate the children,\' \' said Dr. Mario Soto , who has treated at least 300 children at the shelter for diarrhea , conjunctivitis and bacterial infections .Aid workers struggled Friday to reach survivors of Hurricane Mitch , who are in danger of dying from starvation and disease in the wake of the storm that officials estimate killed more than 10,000 people .\r\nForeign aid and pledges of assistance poured into Central America , but damage to roads and bridges reduced the amount of supplies reaching hundreds of isolated communities to a trickle : only as much as could be dropped from a helicopter , when the aircraft can get through .\r\nIn the Aguan River Valley in northern Honduras , floodwaters have receded , leaving a carpet of mud over hundreds of acres ( hectares ) .\r\nIn many nearby villages , residents have gone days without potable water or food .\r\nA 7-month-old baby died in the village of Olvido after three days without food .Two British ships that were in the area on an exercise were on their way to Honduras to join relief efforts , the Defense Ministry said Friday .\r\n` ` It \'s a coincidence that the ships are there but they \'ve got men and equipment that can be put to work in an organized way,\' \' said International Development Secretary Clare Short .\r\nNicaragua said Friday it will accept Cuba \'s offer to send doctors as long as the communist nation flies them in on its own helicopters and with their own supplies .\r\nNicaraguan leaders previously had refused Cuba \'s offer of medical help , saying it did not have the means to transport or support the doctors .\r\nNicaragua \'s leftist Sandinistas , who maintained close relations with Fidel Castro during their 1979-90 rule , had criticized the refusal by President Arnoldo Aleman \'s administration ."

# h_sum = "Prospects were dim for resolution of the political crisis in Cambodia in October 1998.\r\nPrime Minister Hun Sen insisted that talks take place in Cambodia while opposition leaders Ranariddh and Sam Rainsy, fearing arrest at home, wanted them abroad.\r\nKing Sihanouk declined to chair talks in either place.\r\nA U.S. House resolution criticized Hun Sen\'s regime while the opposition tried to cut off his access to loans.\r\nBut in November the King announced a coalition government with Hun Sen heading the executive and Ranariddh leading the parliament.\r\nLeft out, Sam Rainsy sought the King\'s assurance of Hun Sen\'s promise of safety and freedom for all politicians.\r\nCambodian prime minister Hun Sen rejects demands of 2 opposition parties for talks in Beijing after failing to win a 2/3 majority in recent elections.\r\nSihanouk refuses to host talks in Beijing.\r\nOpposition parties ask the Asian Development Bank to stop loans to Hun Sen\'s government.\r\nCCP defends Hun Sen to the US Senate.\r\nFUNCINPEC refuses to share the presidency.\r\nHun Sen and Ranariddh eventually form a coalition at summit convened by Sihanouk.\r\nHun Sen remains prime minister, Ranariddh is president of the national assembly, and a new senate will be formed.\r\nOpposition leader Rainsy left out.\r\nHe seeks strong assurance of safety should he return to Cambodia.\r\nCambodia King Norodom Sihanouk praised formation of a coalition of the Countries top two political parties, leaving strongman Hun Sen as Prime Minister and opposition leader Prince Norodom Ranariddh president of the National Assembly.\r\nThe announcement comes after months of bitter argument following the failure of any party to attain the required quota to form a government.\r\nOpposition leader Sam Rainey was seeking assurances that he and his party members would not be arrested if they return to Cambodia.\r\nRainey had been accused by Hun Sen of being behind an assassination attempt against him during massive street demonstrations in September.\r\nCambodian elections, fraudulent according to opposition parties, gave the CPP of Hun Sen a scant majority but not enough to form its own government.\r\nOpposition leaders fearing arrest, or worse, fled and asked for talks outside the country.\r\nHan Sen refused.\r\nThe UN found evidence of rights violations by Hun Sen prompting the US House to call for an investigation.\r\nThe three-month governmental deadlock ended with Han Sen and his chief rival, Prince Norodom Ranariddh sharing power.\r\nHan Sen guaranteed safe return to Cambodia for all opponents but his strongest critic, Sam Rainsy, remained wary.\r\nChief of State King Norodom Sihanouk praised the agreement."

h_sum = "Hurricane Mitch approached Honduras on Oct. 27, 1998 with winds up to 180mph a Category 5 storm.\r\nIt hit the Honduran coast on Oct. 28 bringing downpours that forced large-scale evacuations.\r\nOn Nov. 1 Nicaragua announced collapse of a drenched volcano crater killing about 2,000.\r\nBy then Mitch\'s winds were down to 30mph, but as disaster reports poured in the death toll finally exceeded 10,000 and half a million left homeless.\r\nThe European Union, international relief agencies, Mexico, the U.S., Japan, Taiwan, the U.K. and U.N. sent financial aid, relief workers and supplies.\r\nPope John Paul II appealed to \"all public and private institutions\" to help.\r\nHonduras braced as category 5 Hurricane Mitch approached.\r\nSlow-moving Mitch battered the Honduran coast for 3 days.\r\nHonduran death estimates grew from 32 to 231 in the first days, to 6,076, with 4,621 missing.\r\nAbout 2,000 were killed in Nicaragua, 239 in El Salvador, 194 in Guatemala, 6 in southern Mexico and 7 in Costa Rica.\r\nThe EU approved 6.4 million ecu in aid to Mitch\'s victims.\r\nThe Pope appealed for aid.\r\nThe US boosted aid to $70 million.\r\nA id workers struggled to reach survivors in danger.\r\nHurricane winds, rain and floods caused massive damage to homes, businesses, roads and bridges.\r\nLatest reports estimate over 10,000 killed in Central America.\r\nHurricane Mitch, category 5 hurricane, brought widespread death and destruction to Central American.\r\nEspecially hard hit was Honduras where an estimated 6,076 people lost their lives.\r\nThe hurricane, which lingered off the coast of Honduras for 3 days before moving off, flooded large areas, destroying crops and property.\r\nThe U.S. and European Union were joined by Pope John Paul II in a call for money and workers to help the stricken area.\r\nPresident Clinton sent Tipper Gore, wife of Vice President Gore to the area to deliver much needed supplies to the area, demonstrating U.S. commitment to the recovery of the region.\r\nA category 5 storm, Hurricane Mitch roared across the northwest Caribbean with 180 mph winds across a 350-mile front that devastated the mainland and islands of Central America.\r\nAlthough the force of the storm diminished, at least 8,000 people died from wind, waves and flood damage.\r\nThe greatest losses were in Honduras where some 6,076 people perished.\r\nAround 2,000 people were killed in Nicaragua, 239 in El Salvador, 194 in Guatemala, seven in Costa Rica and six in Mexico.\r\nAt least 569,000 people were homeless across Central America.\r\nAid was sent from many sources (European Union, the UN, US and Mexico).\r\nRelief efforts are hampered by extensive damage."

# h_sum = "On Oct. 16, 1998 British police arrested former Chilean dictator Pinochet on a Spanish warrant charging murder of Spaniards in Chile, 1973-1983.\r\nFidel Castro denounced the arrest.\r\nThe Chilean government protested strongly.\r\nWhile the British government defended the arrest, it and the Spanish government took no stand on extradition of Pinochet to Spain, leaving it to the courts.\r\nChilean legislators lobbied in Madrid against extradition, while others endorsed it.\r\nThen new charges were filed for crimes against Swiss and French citizens.\r\nPinochet\'s wife and family pleaded that he was too sick to face extradition.\r\nAs of Oct. 28 the matter was not resolved.Pinochet arrested in London on Oct. 16 at a Spanish judge\'s request for atrocities against Spaniards in Chile during his rule.\r\nCastro, Chilean legislators and Pinochet\'s lawyers protested and claimed he had diplomatic immunity.\r\nHis wife asked for his release because he was recovering from recent back surgery.\r\nPinochet visited Thatcher before his surgery.\r\nThe British and Spanish governments defended the arrest, saying it was strictly a legal matter.\r\nThe EC president hoped Pinochet would stand trial.\r\nNone of his Swiss accounts have been frozen yet.\r\nThe Swiss government also asked for his arrest for the 1977 disappearance of a Swiss-Chilean student.\r\nFormer Chilean dictator Augusto Pinochet has been arrested in London at the request of the Spanish government.\r\nPinochet, in London for back surgery, was arrested in his hospital room.\r\nSpain is seeking extradition of Pinochet from London to Spain to face charges of murder in the deaths of Spanish citizens in Chile under Pinochet\'s rule in the 1970s and 80s.\r\nThe arrest raised confusion in the international community as the legality of the move is debated.\r\nPinochet supporters say that Pinochet\'s arrest is illegal, claiming he has diplomatic immunity.\r\nThe final outcome of the extradition request lies with the Spanish courts.\r\nBritain caused international controversy and Chilean turmoil by arresting former Chilean dictator Pinochet in London for Spain\'s investigation of Spanish citizen deaths under Pinochet\'s 17-year rule of torture and political murder.\r\nClaims are Pinochet had diplomatic immunity, extradition is international meddling or illegal because Pinochet is not a Spanish citizen, also his crimes should be punished.\r\nSpain and Britain, big Chilean investors, fear damage to economic relations and let courts decide extradition.\r\nThe Swiss haven\'t investigated Pinochet accounts despite a Spanish request.\r\nPinochet is shielded from details, said too sick to be extradited.\r\n"

def formatter(summary):
    l_sum = sent_tokenize(summary)
    fmt_summary = ""
    for s in l_sum:
        fmt_summary += (s + "\n")
    return fmt_summary

#contains combo_sum and textrank. Score_select is fraction of text selected as summary for textrank
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
    from rouge_score import rouge_scorer

    tr_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeLsum'], use_stemmer=True)
    tr_scores = tr_scorer.score(full_sum, h_sum)

    return [tr_scores['rougeLsum']['F_Measure'], tr_scores['rougeLsum']['Precision'], tr_scores['rougeLsum']['Recall']]


#print(textrank_summarizer(data, 2))




#NLTK WF
def nltk_summarizer(all_texts, score_select):
    def frequency_table(data):
        '''Tokenizing text by word and building a frequency table.'''
        words = word_tokenize(data)
        stop_words = set(stopwords.words("english"))

        freqtable = {}
        for word in words:
            word = word.lower()
            if word not in stop_words:
                if word in freqtable:
                    freqtable[word] += 1
                else:
                    freqtable[word] = 1
       # print(freqtable.items())
        return freqtable


    def scores(data, freqtable):
        '''Creates a dictionary of scores for each sentence.'''
        sentences = sent_tokenize(data)
        sent_score = {}

        for sent in sentences:
            for word, freq in freqtable.items():
                if word in sent.lower():
                    if sent in sent_score:
                        sent_score[sent] += freq
                    else:
                        sent_score[sent] = freq
        return sent_score

    # print out setences higher than average
    def summary(data, sent_score):
        '''Calculate average sentence score and print sentences at least 1.5 times greater than the average'''
        sum_scores = 0
        for sentence in sent_score:
            sum_scores += sent_score[sentence]
        average_score = sum_scores / len(sent_score)

        sentences = sent_tokenize(data)
        summary = ''
        for sentence in sentences:
            if sentence in sent_score and sent_score[sentence] > average:
                summary += " " + sentence
        return summary

    # print out top 5 highest scoring sentences
    def summary_N(data, sent_score):
        summary = ''
        sorted_scores = sorted(sent_score, reverse = True)
        for sent in list(sorted_scores)[0:(len(sorted_scores)//score_select)]:
            summary += sent
        return summary

    def single_summarize(data):
        '''Combines functions to summarize a single document.'''
        freqtable = frequency_table(data)
        sent_scores = scores(data, freqtable)
        single_summary = summary_N(data, sent_scores)
        return single_summary

    def multi_sum(data):
        '''Generates summaries for all documents. Takes in a list of texts'''
        summaries = []
        for d in data:
            s = single_summarize(d)
            summaries.append(s)
        return summaries

    def combine(summaries):
        '''Merge summaries into a single summary. Takes in a list of summaries'''
        all_sums = ''
        for s in summaries:
            all_sums += s
        return all_sums

    def sum_all(data):
        '''Generates comprehensive summary. Takes in a list of texts'''
        all_summaries = multi_sum(data)
        sum_sheet = combine(all_summaries)
        final_summary = single_summarize(sum_sheet) #can change to include further redundancy reduction
        return final_summary

    combine_sum = sum_all(all_texts)
    combine_sum = formatter(combine_sum)

    #Py-Rouge evaluation
    from rouge_score import rouge_scorer

    wf_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeLsum'], use_stemmer=True)
    wf_scores = wf_scorer.score(combine_sum, h_sum)
    #print("WF:", wf_scores, "\n\nTextrank", tr_scores, "\n\nTextScore+", tr_wf_scores)
    #print("score_select", score_select)
    #print(sent_tokenize(combine_sum))

    return [wf_scores['rougeLsum']['F_Measure'], wf_scores['rougeLsum']['Precision'], wf_scores['rougeLsum']['Recall']]

#Control panel
tr_fmeasures = []
tr_precisions = []
tr_recalls = []

wf_fmeasures = []
wf_precisions = []
wf_recalls = []

#5-25,
for i in range(9, 12):
    score_tr = textrank_summarizer(data, i)
    #score_wf = nltk_summarizer(all_texts, i)

    tr_fmeasures.append(score_tr[0])
    tr_precisions.append(score_tr[1])
    tr_recalls.append(score_tr[2])

    #wf_fmeasures.append(score_wf[0])
    #wf_precisions.append(score_wf[1])
    #wf_recalls.append(score_wf[2])

df_tr = pd.DataFrame(list(zip(tr_fmeasures, tr_precisions, tr_recalls)), columns=["[TR] F Measure", "[TR] Precision", "[TR] Recall"])
#df_wf = pd.DataFrame(list(zip(wf_fmeasures, wf_precisions, wf_recalls)), columns=["[WF] F Measure", "[WF] Precision", "[WF] Recall"])
print(df_tr)
#df_tr.to_csv('tr_14_hurricane.csv', encoding='utf-8')
#print(df_wf)

#print(nltk_summarizer(all_texts, 2))
##wf_y = [assorted_summarizer(all_texts, 1)]
