##### Homework 3 #######
#Lina Kaval

##### inputs
# name of input file
in_file = 'SUBMISSION FILES/run3-input.txt'
out_file = 'LinaKaval-run3-output-2.txt'
# number of NFRs expected
nfr_num = 4


############################################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

#read in file
with open(in_file, 'r') as f:
    data = f.readlines()

#separate input into lists
NFR_list = data[0:nfr_num*2:2]
FR_list = data[nfr_num*2::2]

#clean lists
for i in range(0, len(NFR_list)):
    NFR_list[i] = re.sub(r'NFR\d+(.+):', '', NFR_list[i], count = 1).strip()#.split()
    if 'no' in NFR_list[i].lower():
        NFR_list[i] = re.sub(r' No \S+ \S+ ', ' ', NFR_list[i])

    if 'user' in NFR_list[i].lower():
        NFR_list[i] = NFR_list[i] + 'search click use password usability customer browse login download enter email access ratings input preferred'        
print(NFR_list)
for j in range(0, len(FR_list)):
    FR_list[j] = re.sub(r'FR\d+:', '', FR_list[j], count = 1).strip()#.split()

    if 'preferred' in FR_list[j].lower():
        FR_list[j] = re.sub(r'adjuster ', ' ', FR_list[j])
        #FR_list[j] = FR_list[j] + 'user'
    if 'return' in FR_list[j].lower():
        FR_list[j] = re.sub(r'return ', ' ', FR_list[j])
    
with open(out_file, 'w') as outf:
    #tf idf vectorizer and compute cosine
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    for j in range(0, len(FR_list)):
        nfr_sim = []
        for i in range(0, len(NFR_list)):
            vectors = vectorizer.fit_transform([NFR_list[i], FR_list[j]])
            cosine_sim = cosine_similarity(vectors, vectors)[0][1]
            #print(cosine_sim)
            if(cosine_sim > 0.072):
                cos_bit = 1
            else:
                cos_bit = 0
            nfr_sim.append(cos_bit)
            
            #nfr_sim.append(cos_bit)
            #print(vectors)
            #feature_names = vectorizer.get_feature_names_out()
            #dense = vectors.todense()
            #denselist = dense.tolist()
            #print(denselist)
            #df = pd.DataFrame(denselist, columns = feature_names)
            
            #print(df)
        #### change to 4 if nfr is 4
        output = "FR{0},{1},{2},{3},{4}".format(j+1, nfr_sim[0], nfr_sim[1], nfr_sim[2], nfr_sim[3])
        outf.write(output + '\n')   
