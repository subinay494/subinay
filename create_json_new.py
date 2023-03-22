from bs4 import BeautifulSoup
from collections import defaultdict
import os
import json
import re
file_path="/home/subinay/Documents/data/sentence_tag/Annotated_Anjana_200/"
file_path1="/home/subinay/Documents/data/sentence_tag/Annotated_Bipasha_200/"
json_data=open("/home/subinay/Documents/data/sentence_tag/annotated_anjana_200.json")
json_data1=open("/home/subinay/Documents/data/sentence_tag/annotated_bipasha_200.json")
jdata = json.load(json_data)     
jdata1=json.load(json_data1)
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|P.Ws|etc|viz|No|Pt|ss|Cr|P|C|W|no|Ss|Nos)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|P.Ws.)"
digits = "([0-9])"
def split_into_sentences(text):  
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    #sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences
# def create_tuple(tag_text,text,k,line):
#     list_anno1=[]
#     list_new=[]
#     for key,value in tag_text.items():
#         if (key in text and key!=''):
#             t=split_into_sentences(key)
#             for i in range(len(t)):
#                 j=k
#                 if t[i]!='':
#                     m=[t[i],j,value]
#                     list_anno1.append(m)
#                     j=j+1
#     #print(list_anno1)
#     for i in range(len(list_anno1)):
#         if(list_anno1[i][0]==line):
#             list_new
        
           
#     return  s
def search(str,line,path,tag):
    list_new=[]
    with open(path+str,"r") as f2:
        content = f2.read();
    result = " ".join(line.strip() for line in content.splitlines())
    #soup = BeautifulSoup(result,'html.parser')
    set_of_sentences=split_into_sentences(result)
    for i in range(len(set_of_sentences)):
        content=set_of_sentences[i]
        result = " ".join(line.strip() for line in content.splitlines())
        soup1 = BeautifulSoup(result,'html.parser')
        tags = [tag.name for tag in soup1.find_all()]
        tags=set(tags)
        tag=set(tag)
        if(line in set_of_sentences[i] and (tags==tag)):
            #print(i)
            list_new.append(i)

        # if tags:
        #     tags=set(tags)
        #     tag_text ={}
        #     tag_text =defaultdict(list)
        #     for tag in tags:
        #         for a in soup.find_all(tag):
        #             tag_text[a.text].append(i)
        #     index1=create_tuple(tag_text,set_of_sentences[i],i,line)
    #print(list_new)
    return list_new

dict3={}
def document(str,jdata,jdata1):  # to split the document in to sentenecs
    with open("/home/subinay/Documents/data/Tag_prediction_model/clean_random_200_doc/"+str,"r")as f1:
        content = f1.read();
    result = " ".join(line.strip() for line in content.splitlines())
    sentence=split_into_sentences(result)
    dict={}
    v1=jdata[str]
    dict1=[]
    for k, v in v1.items():
        t=split_into_sentences(k)
        if(len(t)>1):
            count=0
            for j in range(len(sentence)):
                if t[0] in sentence[j] and t[0]!='':
                    count+=1
                    m1=[j]
            if(count==1):
                j=m1[0]
                for i in range(len(t)):
                    if t[i]!='':
                        tag=[','.join(v)]
                        m=[t[i],j,tag]
                        j=j+1
                        dict1.append(m)
            else:
                index=search(str,t[0],file_path,v)
                j=index[0]
                for i in range(len(t)):
                    if t[i]!='':
                        tag=[','.join(v)]
                        m=[t[i],j,tag]
                        j=j+1
                        dict1.append(m)
        else:
            count=0
            for j in range(len(sentence)):
                if t[0] in sentence[j] and t[0]!='':
                    count+=1
                    m1=[j]
            if(count==1):
                j=m1[0]
                for i in range(len(t)):
                    tag=[','.join(v)]
                    m=[t[i],j,tag]
                    j=j+1
                    dict1.append(m)
            else:
                index=search(str,t[0],file_path,v)
                j=index[0]
                for i in range(len(t)):
                    tag=[','.join(v)]
                    m=[t[i],j,tag]
                    j=j+1
                    dict1.append(m)
    dict[str]=dict1
    v1=jdata1[str]
    dict2={}
    dict4=[]
    for k, v in v1.items():
        t=split_into_sentences(k)
        if(len(t)>1):
            count=0
            for j in range(len(sentence)):
                if t[0] in sentence[j] and t[0]!='':
                    count+=1
                    m1=[j]
            if(count==1):
                j=m1[0]
                for i in range(len(t)):
                    if t[i]!='':
                        tag=[','.join(v)]
                        m=[t[i],j,tag]
                        j=j+1
                        dict4.append(m)
            else:
                index=search(str,t[0],file_path1,v)
                j=index[0]
                for i in range(len(t)):
                    if t[i]!='':
                        tag=[','.join(v)]
                        m=[t[i],j,tag]
                        j=j+1
                        dict4.append(m)
        else:
            count=0
            for j in range(len(sentence)):
                if t[0] in sentence[j] and t[0]!='':
                    count+=1
                    m1=[j]
            if(count==1):
                j=m1[0]
                for i in range(len(t)):
                    tag=[','.join(v)]
                    m=[t[i],j,tag]
                    j=j+1
                    dict4.append(m)
            else:
                index=search(str,t[0],file_path1,v)
                j=index[0]
                for i in range(len(t)):
                    tag=[','.join(v)]
                    m=[t[i],j,tag]
                    j=j+1
                    dict4.append(m)
    dict2[str]=dict4
    v1=dict[str]
    v=dict2[str]
    # print(v1)
    # print("###")
    # print(v)
    dict5={}
    dict5['anno1']=v1
    dict5['anno2']=v
    dict3[str]=dict5
dir="/home/subinay/Documents/data/Tag_prediction_model/clean_random_200_doc/"
files=os.listdir(dir)
#for f in files:
document("2008.INSC.5.txt",jdata,jdata1)
with open("search_file.json", "w") as outfile:
      json.dump(dict3, outfile)
























        
























#     dict2[str]=dict4
#     v1=dict[str]
#     v=dict2[str]
#     print(v1)
#     print("###")
#     print(v)
   
#     dict5={}
#     dict5['anno1']=v1
#     dict5['anno2']=v
#     dict3[str]=dict5

# dir="/home/subinay/Documents/data/Tag_prediction_model/clean_random_200_doc/"
# files=os.listdir(dir)
# #for f in files:
# document("2013.INSC.349.txt",jdata,jdata1)
# with open("search_file.json", "w") as outfile:
#       json.dump(dict3, outfile)











      # count=0
        # for j in range(len(sentence)):
        #     if t[0] in sentence[j] and t[0] !='':
        #             count+=1
        #             tag=[','.join(v)]
        #             m=[t[i],j,tag]
        #     if(count==1):
        #         dict4.append(m)
        #     else:
        #         index=search(str,t[i],file_path1,v)
        #         tag=[','.join(v)]
        #         m=[t[i],index[0],tag]
        #         dict4.append(m)