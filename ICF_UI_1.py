"""
!/usr/bin/python
-*- coding: utf-8 -*-
This file is subject to the terms and conditions defined in
file 'LICENSE.txt' which is part of this source code package.
__author__ = 'Nafisa Ali'

#################Module Information##############################################
#   Module Name         :   Document Analysis.py
#   Input Parameters    :   None
#   Output              :   None
#   Execution Steps     :   Streamlit application
#   Predecessor module  :   This module is a generic module
#   Successor module    :   NA
#   Last changed on     :   25th March 2023
#   Last changed by     :   Nafisa Ali
#   Reason for change   :   Code development
##################################################################################
"""
import traceback
import warnings
from pprint import pprint
warnings.filterwarnings('ignore')
from IPython.display import display, clear_output
from IPython.core.display import HTML
from IPython.display import HTML
from elasticsearch import Elasticsearch, RequestsHttpConnection
from haystack.utils import print_documents
from haystack.pipelines import DocumentSearchPipeline
from haystack.nodes import Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever
import pandas as pd
import numpy as np
# %config InlineBackend.figure_format = 'retina'
import numpy as np
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import OpenAIAnswerGenerator
from haystack.pipelines import GenerativeQAPipeline
import pandas as pd
import openai
# API_KEY = '#'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' #'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'
API_KEY = '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' #'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'
openai.api_key = API_KEY
from scipy.spatial import distance
import time
from IPython.display import display, clear_output
from IPython.core.display import HTML
from IPython.display import HTML
from fuzzywuzzy import fuzz
import ast
from fuzzywuzzy import fuzz
import streamlit as st
import json
import webbrowser


entity_dict = {'DOC_1': {'Drug': 'BMS-986165', 'Disease': 'Plaque Psoriasis', 'country': 'Canada'}, 
               'DOC_2': {'Drug': 'BMS-986165', 'Disease': 'Plaque Psoriasis', 'country': 'United States of America'}, 
               'DOC_3': {'Drug': 'Nivolumab', 'Disease': 'Hepatocellular Carcinoma', 'country': 'United States'}, 
               'DOC_4': {'Drug': 'Nivolumab', 'Disease': 'Hepatocellular Carcinoma', 'country': 'United States'}, 
               'DOC_5': {'Drug': 'Nivolumab', 'Disease': 'Hepatocellular Carcinoma', 'country': 'Japan'}, 
               'DOC_6': {'Drug': 'Nivolumab', 'Disease': 'Hepatocellular Carcinoma', 'country': 'United States'}}

# document_link_dict = {'DOC_1': "https://zsassociates-my.sharepoint.com/:b:/g/personal/nafisa_ali_zs_com/EaYJ1Pjm_NhFjpi_r2XA284Bi0tBN4zxjKs3d8YmtOI1SQ?e=hzgZo1",
#                       'DOC_2': "https://zsassociates-my.sharepoint.com/:b:/g/personal/nafisa_ali_zs_com/EbTm1GCJ37BDu4Tg5hG2pZsBNBVc03MLbo50JxyIAWfCUQ?e=KQqBPe",
#                       'DOC_3': "https://zsassociates-my.sharepoint.com/:b:/g/personal/nafisa_ali_zs_com/EQbBPvxlPA1BhPfD3VqYCKAB_Bl-eKzVZLX4yg61Uddyjg?e=clttcA",
#                       'DOC_4': "https://zsassociates-my.sharepoint.com/:b:/g/personal/nafisa_ali_zs_com/ER4EiR2cLMFBpwks6uwffdABvUQ9pxFDNVCS8GWw5IDIMg?e=xmL7hK",
#                       'DOC_5': "https://zsassociates-my.sharepoint.com/:b:/g/personal/nafisa_ali_zs_com/ETGbecNPm3JGhbl9-5cPJTEBUsJcXMLyY2FsdwODGf3zow?e=C2nbd2",
#                       'DOC_6': "https://zsassociates-my.sharepoint.com/:b:/g/personal/nafisa_ali_zs_com/EVKHVKjeaEJEo4W2PQKW2xAB9vL8Jf1a8CPC225DDIs9XA?e=GAYlbk" 
#                      }

document_link_dict = {'DOC_1': "https://zsassociates-my.sharepoint.com/:b:/g/personal/nafisa_ali_zs_com/EQ_F9gGHR_xMj2UYbYTvqC8BdxNvvRtje9QvdtdV72wWvw?e=q4oSR5",
                      'DOC_2': "https://zsassociates-my.sharepoint.com/:b:/g/personal/nafisa_ali_zs_com/EXmPlLV61G5AjztINgkO36MBeGOsDoB31PHTWNXET4JoIw?e=rSDXes",
                      'DOC_3': "https://zsassociates-my.sharepoint.com/:b:/g/personal/nafisa_ali_zs_com/Eeu-4YdUmDdBkbHKNFfDVSIBkNs8v7v5nW8Z61BoIPKp4Q?e=MHj51f",
                      'DOC_4': "https://zsassociates-my.sharepoint.com/:b:/g/personal/nafisa_ali_zs_com/EQZmhjUpzmBNoH69zNU5bFEBPAWVHU7IV-Ng5B8ed8pYNg?e=Bw5pNQ",
                      'DOC_5': "https://zsassociates-my.sharepoint.com/:b:/g/personal/nafisa_ali_zs_com/ETUrAXcOHY5PoQgpPTh--CcBCH8wYknUhcDihnrmLL9BUA?e=8s2J1w",
                      'DOC_6': "https://zsassociates-my.sharepoint.com/:b:/g/personal/nafisa_ali_zs_com/EVQ0MgdMGzJOm-pi-Ab2qEMBzM7trKw59OvK0V36bRvPbw?e=j3A4Ok" 
                     }


page_bg_img = '''
<style>
.stApp {
background-image: url("https://st.depositphotos.com/4376739/60921/i/450/depositphotos_609213220-stock-photo-medical-abstract-background-health-care.jpg");
background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# protocol_document_store = FAISSDocumentStore(sql_url="sqlite:///my_doc_store_60000.db", embedding_dim=1536, faiss_index_factory_str="Flat")
# icf_document_store = FAISSDocumentStore(sql_url="sqlite:///my_doc_store_60001.db", embedding_dim=1536, faiss_index_factory_str="Flat")
# combined_document_store = FAISSDocumentStore(sql_url="sqlite:///my_doc_store_60002.db", embedding_dim=1536, faiss_index_factory_str="Flat")

# protocol_document_store.save(index_path = "protocol_document_store_index", config_path = "protocol_document_store_config")
# icf_document_store.save(index_path = "icf_document_store_index", config_path = "icf_document_store_config")
# combined_document_store.save(index_path = "combined_document_store_index", config_path = "combined_document_store_config")


# protocol_document_store = FAISSDocumentStore.load(index_path = "protocol_document_store_index_1", config_path = "protocol_document_store_config_1")
# icf_document_store = FAISSDocumentStore.load(index_path = "icf_document_store_index_1", config_path = "icf_document_store_config_1")
# combined_document_store = FAISSDocumentStore.load(index_path = "combined_document_store_index_1", config_path = "combined_document_store_config_1")

protocol_document_store = FAISSDocumentStore.load(index_path = "pdocstore6_index", config_path = "pdocstore6_config")
icf_document_store = FAISSDocumentStore.load(index_path = "idocstore6_index", config_path = "idocstore6_config")
combined_document_store = FAISSDocumentStore.load(index_path = "cdocstore6_index", config_path = "cdocstore6_config")


protocol_retriever = EmbeddingRetriever(
 document_store = protocol_document_store,
 embedding_model="text-embedding-ada-002",
 batch_size = 32,
 api_key= API_KEY,
 max_seq_len = 1024
)

icf_retriever = EmbeddingRetriever(
 document_store = icf_document_store,
 embedding_model="text-embedding-ada-002",
 batch_size = 32,
 api_key= API_KEY,
 max_seq_len = 1024
)

combined_retriever = EmbeddingRetriever(
 document_store = combined_document_store,
 embedding_model="text-embedding-ada-002",
 batch_size = 32,
 api_key= API_KEY,
 max_seq_len = 1024
)

generator = OpenAIAnswerGenerator(api_key= API_KEY, model="text-davinci-003", temperature=.5, max_tokens=200)

protocol_gpt_search_engine = GenerativeQAPipeline(generator=generator, retriever=protocol_retriever)
icf_gpt_search_engine = GenerativeQAPipeline(generator=generator, retriever=icf_retriever)
combined_gpt_search_engine = GenerativeQAPipeline(generator=generator, retriever=combined_retriever)


class ProcessQuery:
    def __init__(self):
        pass
    
#     def query_classifier(self, query):
#         curr_assertion = """A numerical type question is a type of question that asks for a numeric answer. 
#         At times, it may ask to show some pattern of asking for documents such as 'List of...', 'Show me the list of...', 'Identify the...', 'Enlist the...' etc. and it may sometimes start with 'How many', or 'Count of', 'Which documents', 'Which ICFs' etc.
#         A question that is not a Numerical type question is a Qualitative type question. 

#         Based on the information, classify the above query as a 'Numerical type question' or 'Qualitative type question'.
#         If the answer is 'Numerical type question', further classify it into which category of operations will the query belongs. 
#         List of operations: ['Show', 'List', 'Count', 'Enlist', 'Identify']
#         If the answer is 'Qualitative type question', do nothing

#         Just mention the answers being comma-separated only. No extra words such as 'answer'."""

#         gpt3_prompt = 'Query: ' + query + '\n' + curr_assertion
#         openai.api_key = API_KEY

#         response1 = openai.Completion.create(
#           model="text-davinci-003",
#           prompt=gpt3_prompt,
#           temperature=1,
#           max_tokens=100,
#           top_p=1,
#           frequency_penalty=0,
#           presence_penalty=0
#         )
#         gpt3_reasoning = response1['choices'][0]['text']
#         return gpt3_reasoning

    def query_classifier(self, query):

        curr_assertion = """ There can be four categories of a question. 1. Numerical type question 2. Qualitative type question 3. Compliance type question 4. Consent for genomic research question\n
        A numerical type question is a type of question that asks for a numeric answer. 
        At times, it may ask to show some pattern of asking for documents such as 'List of...', 'Show me the list of...', 'Identify the...', 'Enlist the...' etc. and it may sometimes start with 'How many', or 'Count of', 'Which documents', 'Which ICFs' etc.
        A question that is not a Numerical type question is a Qualitative type question. 
        Compliance issue means not following a standard protocol or rule. It is also means violation, breach, contravention and infringement.
        Consent for genomic research type of question means that the question is asking about consent or aggreement related to genomic research or genetic information or DNA testing or Biomarker testing. For example, "Give me a list of samples that are clear to use for genomic research", "List of documents which have consented for genomic research" are some Consent for genomic research questions.
        Based on the information, classify the above query as a 'Numerical type question' or 'Qualitative type question' or 'Compliance type question' or 'Consent for genomic research question'
        If the answer is 'Numerical type question', further clarify if it is talking about a compliance issue or consent for genomic research question or not. If it is talking about compliance, mention 'Compliance'. If it is talking about Consent for genomic research, mention 'Consent for genomic research'. 
        Just mention the answers being comma-separated only. No extra words such as 'answer' etc."""

        gpt3_prompt = 'Query: ' + query + '\n' + curr_assertion
        openai.api_key = API_KEY
        response1 = openai.Completion.create(
          model="text-davinci-003",
          prompt=gpt3_prompt,
          temperature=1,
          max_tokens=100,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
        gpt3_reasoning = response1['choices'][0]['text']
        return gpt3_reasoning
    
    def rephrase_query(self, query): # ICF-document, patient-subject (different function - entity resolution)
        curr_assertion = 'Perform prompt engineering on the query to convert the question into a statement-type question without a question mark at the end. Example: Replace "How many" with "Number of" or "Count of"'

        gpt3_prompt = 'Query: ' + query + '\n' + curr_assertion
        openai.api_key = API_KEY

        response1 = openai.Completion.create(
          model="text-davinci-003",
          prompt=gpt3_prompt,
          temperature=1,
          max_tokens=100,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
        gpt3_reasoning = response1['choices'][0]['text']
        return gpt3_reasoning
    
class DataIngestion:
    def __init__(self):
        pass
    
    def create_embeddings(self, text):
        response_2 = openai.Embedding.create(
          input=text,
          
          model="text-embedding-ada-002"
        )
        return response_2['data'][0]['embedding']

    def prepareData(self, preprocessed_df):
        all_doc_dict = [] 
        for index, rows in preprocessed_df.iterrows():         
            temp_dict = {} 
            meta_dict = {}
            temp_dict['content'] = rows['Section_Name'] + ':- ' + rows['Content_modified']
            meta_dict['document_ID'] = rows['DocID']
            meta_dict['section_name'] = rows['Section_Name']
            meta_dict['category'] = rows['Category']
            meta_dict['entity_dictionary'] = rows['Entity Dictionary']
            temp_dict['meta'] = meta_dict
            temp_dict['name'] = rows['DocID']+'_Row_'+str(index)
            all_doc_dict.append(temp_dict)
                
        return all_doc_dict  
    
    def docWrite_protocol(self, all_doc_dict):
        protocol_document_store.write_documents(all_doc_dict)
        
    def docWrite_icf(self, all_doc_dict):
        icf_document_store.write_documents(all_doc_dict)
    
    def docWrite_combined(self, all_doc_dict):
        combined_document_store.write_documents(all_doc_dict)
        
    def read_data(self, document_number):
        file_path = r"C:\Users\na27078\Downloads\Nafisa ICF data\ICF WITH ENTITIES V1\df{}_w_response_entities_Copy.csv".format(document_number)
        df = pd.read_csv(file_path).fillna('')
        df['DocID'] = ['DOC_{}'.format(document_number)]*len(df)
#         df['Text_Embedding'] = df['Content_modified'].apply(lambda x: self.create_embeddings(x))
        df['Text_Embedding'] = [[]]*len(df)
        for i in range(len(df)):
            if i%10:
                time.sleep(3)
            if df['Content_modified'][i]=='' or df['Content_modified'][i]==' ' or len(df['Content_modified'][i])==0:
                continue
            else:
                df['Text_Embedding'][i] = self.create_embeddings(df['Content_modified'][i])
            
        df.reset_index(drop = True, inplace = True)
        return df
    
    def ingest_protocol_data_haystack(self, df):
        df_study = df[df['Category']=='Study']
        df_study.reset_index(drop = True, inplace = True)
        all_doc_dict = self.prepareData(df_study)
        self.docWrite_protocol(all_doc_dict)
        protocol_document_store.update_embeddings(protocol_retriever)
        
    def ingest_icf_data_haystack(self, df):
        df_certificate = df[df['Category']=='Certificate']
        df_certificate.reset_index(drop = True, inplace = True)
        all_doc_dict = self.prepareData(df_certificate)
        self.docWrite_icf(all_doc_dict)
        icf_document_store.update_embeddings(icf_retriever)        
        
    def ingest_combined_data_haystack(self, df):
        df.reset_index(drop = True, inplace = True)
        all_doc_dict = self.prepareData(df)
        self.docWrite_combined(all_doc_dict)
        combined_document_store.update_embeddings(combined_retriever)
        
        
class PostProcess:
    def __init__(self):
        pass
    def get_entities(x):
        '''Function to extract the drug and disease from content - It iterates through each sentence till it finds both drug and disease'''
        list_of_content = x
        for cont in list_of_content:
            entity_dict =  self.get_biomedical_entities(cont)
            if entity_dict['Drug'] != 'Entity Not Found' and entity_dict['Disease'] != 'Entity Not Found':
                break
        return entity_dict
    
    def add_entity_column(dataframe):
        '''Function to create dictionary of extracted drug, disease, Country from ICF documents'''
        list_of_content = dataframe['Content'].tolist()
        entity_dict =  self.get_entities(list_of_content)        
        id = dataframe[(dataframe['Section_Name']=='Id')]['Content'].values[0]
        country=self.get_gpt_response('Identify the country the address is referring to in below section'+'\n' +id)
        entity_dict['country'] = country
        dataframe['Entity Dictionary'] = str(entity_dict)
        return dataframe
    
    def get_gpt_response(self, gpt3_prompt):
        response1 = openai.Completion.create(
        model="text-davinci-003",
        prompt=gpt3_prompt,
        temperature=0,
        max_tokens=900,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        return response1['choices'][0]['text'].strip('\n')

    def get_biomedical_entities(self, question):
        list_of_entities = ['Drug','Disease']
        entity_extracted={}
        for ent in list_of_entities:
            query = 'Identify the {} entity from below text'.format(ent) + ' give "Entity Not Found" if it is not a {} entity'.format(ent) + '\n' + question
            entity_extracted[ent] = self.get_gpt_response(query)
            time.sleep(1)
        time.sleep(2)
        
        return entity_extracted
    
    def question_entity_extraction(self, question):
        '''Function to extract entities from question'''
        entity_dict = self.get_biomedical_entities(question)
        country = self.get_gpt_response('Identify the Country entity from below text'+ ' give "Entity Not Found" if it is not a {} entity'.format('country') +'\n' +question)
        entity_dict['country'] = country
        return entity_dict
    
    def pospt(self, question_dic, dictionary_doc):
        flag = True
        for ent, value in question_dic.items():
            if value!='Entity Not Found':
                if fuzz.partial_ratio(value.lower(),dictionary_doc[ent].lower()) >98:
                    flag = True
                else:
                    flag = False
                    return False
        return flag
    
    
class QualitativePipeline: #Haystack
    def __init__(self):
        self.post_process_obj = PostProcess()
        self.retriever_parameter = 5#10
    def get_results_protocol(self, query):
        query_entity = self.post_process_obj.question_entity_extraction(query)
        document_dict = {}
        retriever_parameter = self.retriever_parameter
        params = {"Retriever": {"top_k": retriever_parameter}, "Generator": {"top_k": 1}}
        response = protocol_gpt_search_engine.run(query = query, params = params)
        answer = response['answers'][0].answer        
        for documents in response['documents']:
            document_meta =  documents.meta
            document_entity_dictionary = ast.literal_eval(document_meta['entity_dictionary'])
            if self.post_process_obj.pospt(query_entity, document_entity_dictionary):
                document_dict[documents.content] = [document_meta['document_ID'], document_meta['section_name']]
        if len(document_dict)==0:
            for documents in response['documents']:
                document_meta =  documents.meta
                document_dict[documents.content] = [document_meta['document_ID'], document_meta['section_name']]
            
        return answer, document_dict 
    
    def get_results_icf(self, query):
        query_entity = self.post_process_obj.question_entity_extraction(query)
        document_dict = {}
        retriever_parameter = self.retriever_parameter
        params = {"Retriever": {"top_k": retriever_parameter}, "Generator": {"top_k": 1}}
        response = icf_gpt_search_engine.run(query = query, params = params)
        answer = response['answers'][0].answer        
        for documents in response['documents']:
            document_meta =  documents.meta
            document_entity_dictionary = ast.literal_eval(document_meta['entity_dictionary'])
            if self.post_process_obj.pospt(query_entity, document_entity_dictionary):
                document_dict[documents.content] = [document_meta['document_ID'], document_meta['section_name']]
        if len(document_dict)==0:
            for documents in response['documents']:
                document_meta =  documents.meta
                document_dict[documents.content] = [document_meta['document_ID'], document_meta['section_name']]
            
        return answer, document_dict   
    
    def get_results_combined(self, query):
        query_entity = self.post_process_obj.question_entity_extraction(query)
        document_dict = {}
        retriever_parameter = self.retriever_parameter
        params = {"Retriever": {"top_k": retriever_parameter}, "Generator": {"top_k": 1}}
        response = combined_gpt_search_engine.run(query = query, params = params)
        answer = response['answers'][0].answer        
        for documents in response['documents']:
            document_meta =  documents.meta
            document_entity_dictionary = ast.literal_eval(document_meta['entity_dictionary'])
            if self.post_process_obj.pospt(query_entity, document_entity_dictionary):
                document_dict[documents.content] = [document_meta['document_ID'], document_meta['section_name']]
        if len(document_dict)==0:
            for documents in response['documents']:
                document_meta =  documents.meta
                document_dict[documents.content] = [document_meta['document_ID'], document_meta['section_name']]
            
        return answer, document_dict   
    
    
    
class QuantitativePipeline: #Cosine similarity -> No Haystack 
    def __init__(self):
        self.threshold = 0.85
        self.post_process_obj = PostProcess()
    
    def create_embeddings(self, text):
        response_2 = openai.Embedding.create(
          input=text,
          model="text-embedding-ada-002"
        )
        return response_2['data'][0]['embedding']
    
    def get_similarity(self, query, text):
        cosine_sim = 1 - distance.cosine(query, text)
        return cosine_sim   
    
    def get_results(self, query, df, entity_dict):
        query_embedding = self.create_embeddings(query)
        final_df = pd.DataFrame()
        section, content_list, docid, score = [], [], [], []
        for i in range(len(df)):
            section.append(df['Section_Name'][i])
            content_list.append(df['Content_modified'][i])
            docid.append(df['DocID'][i])
            try:
                cosine_sim = self.get_similarity(query_embedding, ast.literal_eval(df['Text_Embedding'][i]))
            except:
                cosine_sim = 0
            score.append(cosine_sim)

        final_df['Content'] = content_list
        final_df['DocID'] = docid
        final_df['Similarity_Score'] = score    
        final_df_ = final_df.sort_values(by=['Similarity_Score'], ascending=False)
        cut_df = final_df_[final_df_['Similarity_Score']>=self.threshold]
        cut_df.reset_index(drop = True, inplace = True)
        new_df = pd.DataFrame()
        
        query_entity = self.post_process_obj.question_entity_extraction(query)
        
        c,d = [], []
        for i in range(len(cut_df)):
            docid = cut_df['DocID'][i]
            document_entity_dictionary = entity_dict[docid]
            if self.post_process_obj.pospt(query_entity, document_entity_dictionary):
                d.append(docid)
                c.append(cut_df['Content'][i])
                
        if len(d)!=0:
            new_df['DocID'] = d
            new_df['Content'] = c
            
            cut_df_ = new_df.groupby('DocID')['Content'].agg('\n\nPara:'.join).reset_index()
            
        else:
            cut_df_ = cut_df.groupby('DocID')['Content'].agg('\n\nPara:'.join).reset_index()           
#         if category_of_operation.strip() == 'Show' or category_of_operation.strip() == 'List' or category_of_operation.strip() == 'Enlist' or category_of_operation.strip() == 'Identify':            
        return cut_df_, list(cut_df_['DocID']), len(list(cut_df_['DocID']))
#         else:
#             return cut_df_, list(cut_df_['DocID']), len(list(cut_df_['DocID']))    



def complete_ingestion():
    #No. of documents = 6
    ingestion_obj = DataIngestion()
    combined_df = pd.DataFrame()
    entity_dict = {}
    for i in range(1, 7):
        df = ingestion_obj.read_data(i)
        entity_dict['DOC_{}'.format(i)] = ast.literal_eval(df['Entity Dictionary'][0])
        combined_df = combined_df.append(df)
        st.write(str(i) + ' completed!', entity_dict)
    combined_df.reset_index(drop = True, inplace = True)
    ingestion_obj.ingest_protocol_data_haystack(combined_df)
    ingestion_obj.ingest_icf_data_haystack(combined_df)
    ingestion_obj.ingest_combined_data_haystack(combined_df)
    
    st.write('Ingestion Successful..!')
    return combined_df, entity_dict    


def half_ingestion(combined_df):
    ingestion_obj = DataIngestion()
    combined_df.reset_index(drop = True, inplace = True)
    ingestion_obj.ingest_protocol_data_haystack(combined_df)
    ingestion_obj.ingest_icf_data_haystack(combined_df)
    ingestion_obj.ingest_combined_data_haystack(combined_df)
    
    st.write('Ingestion Successful..!')
    
def clean_answer(text):
    response1 = openai.Completion.create(
    model="text-davinci-003",
    prompt= 'Clean the text and make it meaningful in two to there lines. Text - ' + str(text) + ' Mention only the answer. No extra words.',
    temperature=0,
    max_tokens=900,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response1['choices'][0]['text'].strip('\n')


class Compliance:
    def __init__(self, document_number, query):
        self.document_number = document_number
        self.post_process_obj = PostProcess()
        self.query_entity = self.post_process_obj.question_entity_extraction(query)
        
    def read_data_certificate(self):
        file_path = r"C:\Users\na27078\Downloads\Nafisa ICF data\ICF WITH ENTITIES V1\df{}_w_response_entities_Copy.csv".format(self.document_number)
        df = pd.read_csv(file_path).fillna('')
        df['DocID'] = ['DOC_{}'.format(self.document_number)]*len(df)
        df_certificate = df[df['Category']=='Certificate']
        df_certificate.reset_index(drop = True, inplace = True)
        return df_certificate
    
    def create_json_data(self, df):
        output_json = {}
        for i in range(len(df)):
            if ('consent' in df['Section_Name'][i].lower().strip() or 'additional research' in df['Section_Name'][i].lower().strip() or 'Signature'.lower().strip() in df['Section_Name'][i] or 'Regarding Your Samples collected during the study'.lower().strip() in df['Section_Name'][i].lower().strip() or 'TCI'.lower().strip() in df['Section_Name'][i].lower().strip()
            or 'AGREEMENT'.lower().strip() in df['Section_Name'][i].lower().strip() or 'Participate'.lower().strip() in df['Section_Name'][i].lower().strip()) and ('Mobile'.lower().strip() not in df['Section_Name'][i].lower().strip()):
                output_json[df['Section_Name'][i].strip() + " : " + df['Content'][i].strip()] = df['Final_signature'][i]

        return json.dumps(output_json)       
    
    def prompt(self, text):
        curr_assertion = "Theory: Compliance issue happens when only a patient or subject has signed or agreed or consented to the study or research or clinical trial but the study personnel or doctor or person conducting consent discussion or researcher or investigator or signature of person conducting consent discussion has not signed in the document or is missing or is 'No'. The signature or consent for the study/research will be only mentioned as 'Yes' if 'No' is found in any case of patient/subject and 'yes' for investigator cases, it is a compliance or if patient cases are 'Yes' but 'No' for investigator cases, it is a compliance issue. If a blank is present for patient but not for investigator or vice-versa, it is a compliance issue. Do not consider values for Additional research, Consent Withdrawal Form etc. Concentrate only for the entire research. Question: According to the theory, does the Document_text above has compliance issue? Answer in Yes or No and give short one line reason also."
        gpt3_prompt = 'Document_text: ' + text + '\n' + curr_assertion
        response1 = openai.Completion.create(
          model="text-davinci-003",
          prompt=gpt3_prompt,
          temperature=1,
          max_tokens=100,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
        gpt3_reasoning = response1['choices'][0]['text']
        if gpt3_reasoning.lower().strip().startswith('yes'):
            return gpt3_reasoning
        else:
            return "NO"
        
    def process(self):
        df = self.read_data_certificate()
        df.reset_index(drop = True, inplace = True)
        document_entity_dictionary = entity_dict['DOC_'+str(self.document_number)]
        if len(self.query_entity)!=0:
            if self.post_process_obj.pospt(self.query_entity, document_entity_dictionary):
                final_str = self.create_json_data(df)
                try:
                    final_answer = self.prompt(final_str)
                except:
                    return 'Not this document'
                return final_answer
            else:
                final_str = self.create_json_data(df)
                try:
                    final_answer = self.prompt(final_str)
                except:
                    return 'Not this document'
                return final_answer  
        else:
            final_str = self.create_json_data(df)
            try:
                final_answer = self.prompt(final_str)
            except:
                return 'Not this document'
            return final_answer        
        
        
class GenomicResearch:
    def __init__(self, document_number, query):
        self.document_number = document_number
        self.query = 'Does this research involve analyses that reveal genetic/genomic information or additional research may involve your genetic information?'
        self.threshold = 0.85
        self.post_process_obj = PostProcess()
        self.query_entity = self.post_process_obj.question_entity_extraction(query)
    #Yes data
    
    def check_final_signature(self):
    
        file_path = r"C:\Users\na27078\Downloads\Nafisa ICF data\ICF WITH ENTITIES V1\df{}_w_response_entities_Copy.csv".format(self.document_number)
        df = pd.read_csv(file_path).fillna('')
        df['DocID'] = ['DOC_{}'.format(self.document_number)]*len(df)
        df_certificate = df[df['Category']=='Certificate']
        df_certificate.reset_index(drop = True, inplace = True)
        df_yes = df_certificate[df_certificate['Final_signature']=='Yes']
        df_yes.reset_index(drop = True, inplace = True)
        output_json = {}
        for i in range(len(df_yes)):
            output_json[df_yes['Section_Name'][i] + ': ' + df_yes['Content'][i]] = {'Consent': df_yes['Final_signature'][i], 'Document ID': df_yes['DocID'][i]}
        text = json.dumps(output_json)   

        curr_assertion = """Did the patient/subject sign or consent or agreed for the research or study or is there a final signature of the patient present in Document_Text? Answer in either 'Yes' or 'No'. No extra words required in the answer."""

        gpt3_prompt = 'Document_Text: ' + text + '\n' + curr_assertion

        response1 = openai.Completion.create(
          model="text-davinci-003",
          prompt=gpt3_prompt,
          temperature=1,
          max_tokens=100,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
        gpt3_reasoning = response1['choices'][0]['text']

        if 'Yes'.lower().strip() in gpt3_reasoning.lower().strip():
            return 1
        else:
            return 0

    def create_embeddings(self, text):
        response_2 = openai.Embedding.create(
          input=text,
          model="text-embedding-ada-002"
        )
        return response_2['data'][0]['embedding']

    def get_similarity(self, query, text):
        cosine_sim = 1 - distance.cosine(query, text)
        return cosine_sim
    
    def GR_yes_read_data_certificate(self):
        file_path = r"C:\Users\na27078\Downloads\Nafisa ICF data\ICF WITH ENTITIES V1\df{}_w_response_entities_Copy.csv".format(self.document_number)
        df = pd.read_csv(file_path).fillna('')
        df['DocID'] = ['DOC_{}'.format(self.document_number)]*len(df)
        df_certificate = df[df['Category']=='Certificate']
        df_certificate.reset_index(drop = True, inplace = True)
        df_yes = df_certificate[df_certificate['Final_signature']=='Yes']
        df_yes.reset_index(drop = True, inplace = True)
        return df_yes

    def yes_create_json_data(self, df):
        output_json = {}
        for i in range(len(df)):
            output_json[df['Section_Name'][i].strip() + " : " + df['Content'][i].strip()] = {'Consent': df['Final_signature'][i], 'Document ID': df['DocID'][i]}

        return output_json#str(output_json.T.to_dict())  
    
    def jsonCreate(self):
        gr_combined_df = pd.DataFrame()
#         for i in range(self.document_number, self.document_number+1):
#             df = self.GR_yes_read_data_certificate(i)
#             gr_combined_df = gr_combined_df.append(df)
        df = self.GR_yes_read_data_certificate()
        gr_combined_df = gr_combined_df.append(df)
        gr_combined_df.reset_index(drop = True, inplace = True)
        gr_output_json = self.yes_create_json_data(gr_combined_df)
        final_str =json.dumps(gr_output_json)
        return final_str
    
    def yes_prompt(self, text):
        curr_assertion = "In Document_Text, has the patient agreed or consented or said Yes to additional research 'including' genetic research or genetic information in Document_Text? Refer to the consent values (Yes or No). Do not look for 'excluding relevant genetic research/information'. Give accurate answers in the form of 'Yes' or 'No'."
        gpt3_prompt = 'Document_Text: ' + text + '\n' + curr_assertion
        response1 = openai.Completion.create(
          model="text-davinci-003",
          prompt=gpt3_prompt,
          temperature=1,
          max_tokens=100,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
        gpt3_reasoning = response1['choices'][0]['text']
        return gpt3_reasoning
    
    def validate(self, answer):
        if 'Yes'.lower().strip() in answer.lower().strip():
            return 1
        else:
            file_path = r"C:\Users\na27078\Downloads\Nafisa ICF data\ICF WITH ENTITIES V1\df{}_w_response_entities_Copy.csv".format(self.document_number)
            df = pd.read_csv(file_path).fillna('')
            df['DocID'] = ['DOC_{}'.format(self.document_number)]*len(df)
            temp_df = pd.DataFrame()
            section_name, content, final_sig, docid = [], [], [], []
            for i in range(len(df)):
                if 'Additional Research'.lower().strip() in df['Content'][i].lower().strip() or 'Genetic'.lower().strip() in df['Content'][i].lower().strip() or 'DNA'.lower().strip() in df['Content'][i].lower().strip() or 'Genomic'.lower().strip() in df['Content'][i].lower().strip() or 'Biomarker'.lower().strip() in df['Content'][i].lower().strip():
                    section_name.append(df['Section_Name'][i])
                    content.append(df['Content'][i])
                    final_sig.append(df['Final_signature'][i])
                    docid.append(df['DocID'][i])

            temp_df['Section_Name'] = section_name
            temp_df['Content'] = content
            temp_df['Final_signature'] = final_sig
            temp_df['DocID'] = docid
            
            embed = []
            for i in range(len(temp_df)):
                if i%5==0:
                    time.sleep(3)
                x = self.create_embeddings(temp_df['Content'][i])
                embed.append(x)
            temp_df['Embed'] = embed
            
            qembed = self.create_embeddings(self.query)
            cos_sim = []
            for i in range(len(temp_df)):
                cosine_sim = self.get_similarity(qembed, temp_df['Embed'][i])
                cos_sim.append(cosine_sim)
            temp_df['Cosine_similarity'] = cos_sim
            
            temp_df_ = temp_df[temp_df['Cosine_similarity']>=0.85]
            if not temp_df_.empty:
                return 1
            else:
                return 0
            
    def process(self):
        final_str = self.jsonCreate()
        document_entity_dictionary = entity_dict['DOC_'+str(self.document_number)]
        if len(self.query_entity)!=0:
            if self.post_process_obj.pospt(self.query_entity, document_entity_dictionary):
                gpt3_answer = self.yes_prompt(final_str)
                bool_val_1 = self.validate(gpt3_answer)
                bool_val_2 = self.check_final_signature()
                if bool_val_1 and bool_val_2:
                    return "YES"
                else:
                    return "Not this document"
        else:
            gpt3_answer = self.yes_prompt(final_str)
            bool_val_1 = self.validate(gpt3_answer)
            bool_val_2 = self.check_final_signature()
            if bool_val_1 and bool_val_2:
                return "YES"
            else:
                return "Not this document"
            
            
process_query_obj = ProcessQuery()
quant_obj = QuantitativePipeline()
qual_obj = QualitativePipeline()

def process(combined_df, entity_dict, query):
    
        
#     if choice=='Certificate':
#         df = combined_df[combined_df['Category']=='Certificate'.strip()]
#         df.reset_index(drop = True, inplace = True)
#     elif choice=='Protocol':
#         df = combined_df[combined_df['Category']=='Study'.strip()]
#         df.reset_index(drop = True, inplace = True)
#     else:
    choice = 'Combined'
    df = combined_df
    df.reset_index(drop = True, inplace = True)
    classified_query = process_query_obj.query_classifier(query)
    classified_query = classified_query.replace('\n', '').strip()
    if ',' in classified_query or ':' in classified_query:
        question_class = classified_query.split(',')[0].strip()
        if 'Numerical type question'.lower().strip() in question_class.lower().strip() or 'Numerical'.lower().strip() in question_class.lower().strip():
            if 'Compliance'.lower().strip() in classified_query.lower().strip() or 'compliance' in query.lower().strip():
                count_of_documents = 0
                compliant_dict = {}
                noncompliant_dict = {}
                for temp_var in range(1, 7):
                    compliance_obj = Compliance(temp_var, query)
                    final_answer = compliance_obj.process()
                    if final_answer == "NO":
                        noncompliant_dict['DOC_'+str(temp_var)] = 'No compliance issue found'
                    elif final_answer=='Not this document':
                        noncompliant_dict['DOC_'+str(temp_var)] = 'No compliance issue found'
                    else:
                        count_of_documents = count_of_documents + 1
                        compliant_dict['DOC_'+str(temp_var)] = final_answer
                        
                st.markdown('Here is the answer to your question: \n')
                st.write('Number of documents having compliant issues: ' + str(count_of_documents) + ' (' + str(round(count_of_documents/6, 2)*100) + ' % of total documents present)')
                st.write('Number of documents not having compliant issues: ' + str(6-count_of_documents) + ' (' + str(round((6-count_of_documents)/6, 2)*100) +' % of total documents present)')
                for key, value in compliant_dict.items():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(key, ' : ', value)
                    with col2:
                        if st.button('Open document '+ key):
                            webbrowser.open_new_tab(document_link_dict[key])
#                         st.markdown(f''' <a href={document_link_dict[key]}><button style="background-color:LightSalmon;">Refer to the {key} here</button></a>''', unsafe_allow_html=True)
                    
            elif 'genomic research'.lower().strip() in classified_query.lower().strip():
                count_of_documents = 0
                gr_list = []
                non_gr_list = []
                for temp_var in range(1, 7):
                    gr_obj = GenomicResearch(temp_var, query)
                    final_answer = gr_obj.process()
                    if "YES"==final_answer:
                        count_of_documents = count_of_documents+1
                        gr_list.append('DOC_'+str(temp_var))
                    else:
                        non_gr_list.append('DOC_'+str(temp_var))
                    
                st.markdown('Here is the answer to your question: \n')
                st.write('Number of documents qualified for your query : ' + str(count_of_documents) + ' (' + str(round((count_of_documents/6), 2)*100) + ' % of total documents present)')
                st.write('which leaves : ' + str(6-count_of_documents) + ' (' + str(round(((6-count_of_documents)/6), 2)*100) + ' % of total documents not satisfying your query)')
                for temp in gr_list:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(temp)
                    with col2:   
                        if st.button('Open document '+ temp):
                            webbrowser.open_new_tab(document_link_dict[temp])
#                         st.markdown(f''' <a href={temp}><button style="background-color:LightSalmon;">Refer to the {temp} here</button></a>''', unsafe_allow_html=True)
                    
                    
            else:
                cut_df_, docids_list, docids_count = quant_obj.get_results(query, df, entity_dict)
                if len(docids_list)==0:
#                     st.markdown(' Sorry! No results found for your question.. \n ')
                    st.markdown('')
                else:
                    cut_df_.reset_index(drop = True, inplace = True)
                    st.markdown('Here is the answer to your question: \n ')
#                     if query.lower().strip()=='What type of data is collected in Psoriasis trials?'.lower().strip():
#                         st.write('Show me documents related to Psoriasis trials?')
#                     else:
#                         st.write(query)
#                     st.markdown('List of documents satisfying the question :  ')
#                     st.write(docids_list)
                    with st.expander("List of documents satisfying this query"):
                        
                        for i in docids_list:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(i)
                            with col2:
                                if st.button('Open document '+ i):
                                    webbrowser.open_new_tab(document_link_dict[i])
            
                    st.markdown('Count of documents :')
                    st.write(docids_count)
                    st.write('\n')
                    st.markdown('Here are the relevant content from documents you can refer..\n')
                    for i in range(len(cut_df_)):
#                         col1, col2 = st.columns(2)
#                         with col1:
                        st.markdown('**Document ID:**')# {}'.format(cut_df_['DocID'][i]))
                        st.markdown(cut_df_['DocID'][i])  
                        st.markdown('**Content:**')
                        st.write(cut_df_['Content'][i])
                        st.write('\n')
#                         with col2:
#                             if st.button('Open document '+ cut_df_['DocID'][i]):
#                                 webbrowser.open_new_tab(document_link_dict[cut_df_['DocID'][i]])
#                             st.markdown(f''' <a href={document_link_dict[cut_df_['DocID'][i]]}><button style="background-color:LightSalmon;">Refer to the {cut_df_['DocID'][i]} here</button></a>''', unsafe_allow_html=True)
                        
        else:
            
            if 'Compliance'.lower().strip() in classified_query.lower().strip() or 'compliance' in query.lower().strip():
                count_of_documents = 0
                compliant_dict = {}
                noncompliant_dict = {}
                for temp_var in range(1, 7):
                    compliance_obj = Compliance(temp_var, query)
                    final_answer = compliance_obj.process()
                    if final_answer == "NO":
                        noncompliant_dict['DOC_'+str(temp_var)] = 'No compliance issue found'
                    elif final_answer=='Not this document':
                        noncompliant_dict['DOC_'+str(temp_var)] = 'No compliance issue found'
                    else:
                        count_of_documents = count_of_documents + 1
                        compliant_dict['DOC_'+str(temp_var)] = final_answer
                        
                st.markdown('Here is the answer to your question: \n')
                st.markdown('Number of documents having compliant issues: ' + str(count_of_documents)+ ' (' + str(round(count_of_documents/6, 2))+ ' % of total documents present)')
                st.markdown('Number of documents not having compliant issues: ' + str(6-count_of_documents) + ' (' + str(round((6-count_of_documents)/6, 2)*100)+ ' % of total documents present)')
                for key, value in compliant_dict.items():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(key, ' : ', value)
                    with col2:
                        if st.button('Open document '+ key):
                            webbrowser.open_new_tab(document_link_dict[key])
#                         st.markdown(f''' <a href={document_link_dict[key]}><button style="background-color:LightSalmon;">Refer to the {key} here</button></a>''', unsafe_allow_html=True)
            elif 'genomic research'.lower().strip() in classified_query.lower().strip():
                count_of_documents = 0
                gr_list = []
                non_gr_list = []
                for temp_var in range(1, 7):
                    gr_obj = GenomicResearch(temp_var, query)
                    final_answer = gr_obj.process()
                    if "YES"==final_answer:
                        count_of_documents = count_of_documents+1
                        gr_list.append('DOC_'+str(temp_var))
                    else:
                        non_gr_list.append('DOC_'+str(temp_var))
                    
                st.markdown('Here is the answer to your question: \n')
                st.markdown('Number of documents qualified for your query : ' + str(count_of_documents) + ' (' + str(round((count_of_documents/6), 2)*100)+ ' % of total documents present)')
                st.write('which leaves : ' + str(6-count_of_documents) + ' (' + str(round(((6-count_of_documents)/6), 2)*100) + ' % of total documents not satisfying your query)')
                for temp in gr_list:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(temp)
                    with col2:
                        if st.button('Open document '+ temp):
                            webbrowser.open_new_tab(document_link_dict[temp])
#                         st.markdown(f''' <a href={temp}><button style="background-color:LightSalmon;">Refer to the {temp} here</button></a>''', unsafe_allow_html=True)
            
            elif choice=='Protocol':
                answer, document_dict = qual_obj.get_results_protocol(query)
                st.markdown("**Here's your answer.**")
                st.write(answer)
                with st.expander("List of documents satisfying this query"):
                    docs_list = []
                    for k, v in document_dict.items():
                        docs_list.append(str(v[0]))
                    dl = list(set(docs_list))
                    for i in dl:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(i)
                        with col2:
                            if st.button('Open document '+ i):
                                webbrowser.open_new_tab(document_link_dict[i])
#                     with col1:
#                 st.markdown('Here is the answer to your question: \n')
#                 if query.lower().strip()=='What type of data is collected in Psoriasis trials?'.lower().strip():
#                     st.write('Show me documents related to Psoriasis trials?')
#                 else:
#                     st.write(query)
#                 answer = clean_answer(answer)
#                 st.write(answer)
                st.markdown('**Here are the relevant content from protocol you can refer..\n**')
                for key, value in document_dict.items():
#                     col1, col2 = st.columns(2)
#                     with col1:
                    st.markdown('**Document ID:**')#{}'.format(str(value[0])))
                    st.write(str(value[0]))
                    st.markdown('**Section Name:**')# {}'.format(str(value[1])))
                    st.write(str(value[1]))
                    st.markdown('**Content:**')# {}'.format(key))
                    st.write(key)
                    st.write('\n')
#                     with col2:
#                         if st.button('Open document '+ str(value[0])):
#                             webbrowser.open_new_tab(document_link_dict[str(value[0])])
#                         st.markdown(f''' <a href={document_link_dict[str(value[0])]}><button style="background-color:LightSalmon;">Refer to the {str(value[0])} here</button></a>''', unsafe_allow_html=True)
                    
            elif choice=='Certificate':
                answer, document_dict = qual_obj.get_results_icf(query)
                st.markdown("**Here's your answer.**")
                st.write(answer)
                with st.expander("List of documents satisfying this query"):
                    docs_list = []
                    for k, v in document_dict.items():
                        docs_list.append(str(v[0]))
                    dl = list(set(docs_list))
                    for i in dl:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(i)
                        with col2:
                            if st.button('Open document '+ i):
                                webbrowser.open_new_tab(document_link_dict[i])
#                 st.markdown('Here is the answer to your question: \n')
#                 if query.lower().strip()=='What type of data is collected in Psoriasis trials?'.lower().strip():
#                     st.write('Show me documents related to Psoriasis trials?')
#                 else:
#                     st.write(query)
#                 answer = clean_answer(answer)
#                 st.write(answer.replace('I do not agree to the above statement', 'Cross').replace('I agree to the above statement', 'Tick').replace('Certificate', 'Document'))
                st.markdown('Here are the relevant content from documents you can refer..\n')
                for key, value in document_dict.items():
#                     col1, col2 = st.columns(2)
#                     with col1:
                    st.markdown('**Document ID:**')#{}'.format(str(value[0])))
                    st.write(str(value[0]))
                    st.markdown('**Section Name:**')# {}'.format(str(value[1])))
                    st.write(str(value[1]))
                    st.markdown('**Content:**')# {}'.format(key))
                    st.write(key)
                    st.write('\n')
#                     with col2:
#                         if st.button('Open document '+ str(value[0])):
#                             webbrowser.open_new_tab(document_link_dict[str(value[0])])
#                         st.markdown(f''' <a href={document_link_dict[str(value[0])]}><button style="background-color:LightSalmon;">Refer to the {str(value[0])} here</button></a>''', unsafe_allow_html=True)
                    
            else:
                answer, document_dict = qual_obj.get_results_combined(query)
                st.markdown("**Here's your answer.**")
                st.write(answer)
                with st.expander("List of documents satisfying this query"):
                    docs_list = []
                    for k, v in document_dict.items():
                        docs_list.append(str(v[0]))
                    dl = list(set(docs_list))
                    for i in dl:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(i)
                        with col2:
                            if st.button('Open document '+ i):
                                webbrowser.open_new_tab(document_link_dict[i])
#                 st.markdown('Here is the answer to your question: \n ')
#                 if query.lower().strip()=='What type of data is collected in Psoriasis trials?'.lower().strip():
#                     st.write('Show me documents related to Psoriasis trials?')
#                 else:
#                     st.write(query)
#                 answer = clean_answer(answer)
#                 st.write(answer.replace('I do not agree to the above statement', 'Cross').replace('I agree to the above statement', 'Tick').replace('Certificate', 'Document'))
                st.markdown('Here are the relevant content from documents you can refer..\n')
                for key, value in document_dict.items():
#                     col1, col2 = st.columns(2)
#                     with col1:
                    st.markdown('**Document ID:**')#{}'.format(str(value[0])))
                    st.write(str(value[0]))
                    st.markdown('**Section Name:**')# {}'.format(str(value[1])))
                    st.write(str(value[1]))
                    st.markdown('**Content:**')# {}'.format(key))
                    st.write(key)
                    st.write('\n')
#                     with col2:
#                         if st.button('Open document '+ str(value[0])):
#                             webbrowser.open_new_tab(document_link_dict[str(value[0])])
#                         st.markdown(f''' <a href={document_link_dict[str(value[0])]}><button style="background-color:LightSalmon;">Refer to the {str(value[0])} here</button></a>''', unsafe_allow_html=True)
                    
    else:
        if 'Compliance'.lower().strip() in classified_query.lower().strip() or 'compliance' in query.lower().strip():
                count_of_documents = 0
                compliant_dict = {}
                noncompliant_dict = {}
                for temp_var in range(1, 7):
                    compliance_obj = Compliance(temp_var, query)
                    final_answer = compliance_obj.process()
                    if final_answer == "NO":
                        noncompliant_dict['DOC_'+str(temp_var)] = 'No compliance issue found'
                    elif final_answer=='Not this document':
                        noncompliant_dict['DOC_'+str(temp_var)] = 'No compliance issue found'
                    else:
                        count_of_documents = count_of_documents + 1
                        compliant_dict['DOC_'+str(temp_var)] = final_answer
                        
                st.markdown('Here is the answer to your question: \n')
                st.write('Number of documents having compliant issues: '+ str(count_of_documents) + ' (' + str(round(count_of_documents/6, 2)*100)+  ' % of total documents present)')
                st.write('Number of documents not having compliant issues: ' + str(6-count_of_documents) + ' (' + str(round((6-count_of_documents)/6, 2)*100) + ' % of total documents present)')
                for key, value in compliant_dict.items():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(key, ' : ', value)
                    with col2:
                        if st.button('Open document '+ key):
                            webbrowser.open_new_tab(document_link_dict[key])
                        
#                         st.markdown(f''' <a href={document_link_dict[key]}><button style="background-color:LightSalmon;">Refer to the {key} here</button></a>''', unsafe_allow_html=True)
                        
        elif 'genomic research'.lower().strip() in classified_query.lower().strip():
            count_of_documents = 0
            gr_list = []
            non_gr_list = []
            for temp_var in range(1, 7):
                gr_obj = GenomicResearch(temp_var, query)
                final_answer = gr_obj.process()
                if "YES"==final_answer:
                    count_of_documents = count_of_documents+1
                    gr_list.append('DOC_'+str(temp_var))
                else:
                    non_gr_list.append('DOC_'+str(temp_var))

            st.markdown('Here is the answer to your question: \n')
            st.write('Number of documents qualified for your query : ' + str(count_of_documents) + ' (' + str(round((count_of_documents/6), 2)*100) + ' % of total documents present)')
            st.write('which leaves : ' + str(6-count_of_documents) + ' (' + str(round(((6-count_of_documents)/6), 2)*100) + ' % of total documents not satisfying your query)')
            for temp in gr_list:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(temp)
                with col2:
                    if st.button('Open document '+ temp):
                        webbrowser.open_new_tab(document_link_dict[temp])
#                     st.markdown(f''' <a href={temp}><button style="background-color:LightSalmon;">Refer to the {temp} here</button></a>''', unsafe_allow_html=True)
                
 
        elif choice=='Protocol':
            answer, document_dict = qual_obj.get_results_protocol(query)
            st.markdown("**Here's your answer.**")
            st.write(answer)
            with st.expander("List of documents satisfying this query"):
                docs_list = []
                for k, v in document_dict.items():
                    docs_list.append(str(v[0]))
                dl = list(set(docs_list))
                for i in dl:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(i)
                    with col2:
                        if st.button('Open document '+ i):
                            webbrowser.open_new_tab(document_link_dict[i])
#             st.markdown('Here is the answer to your question: \n ')
#             if query.lower().strip()=='What type of data is collected in Psoriasis trials?'.lower().strip():
#                 st.write('Show me documents related to Psoriasis trials?')
#             else:
#                 st.write(query)
#             answer = clean_answer(answer)
#             st.write(answer.replace('I do not agree to the above statement', 'Cross').replace('I agree to the above statement', 'Tick').replace('Certificate', 'Document'))
            st.markdown('Here are the relevant content from protocols you can refer..\n')
            for key, value in document_dict.items():
#                 col1, col2 = st.columns(2)
#                 with col1:
                st.markdown('**Document ID:**')#{}'.format(str(value[0])))
                st.write(str(value[0]))
                st.markdown('**Section Name:**')# {}'.format(str(value[1])))
                st.write(str(value[1]))
                st.markdown('**Content:**')# {}'.format(key))
                st.write(key)
                st.write('\n')
#                 with col2:
#                     if st.button('Open document '+ str(value[0])):
#                         webbrowser.open_new_tab(document_link_dict[str(value[0])])
#                     st.markdown(f''' <a href={document_link_dict[str(value[0])]}><button style="background-color:LightSalmon;">Refer to the {str(value[0])} here</button></a>''', unsafe_allow_html=True)
                
        elif choice=='Certificate':
            answer, document_dict = qual_obj.get_results_icf(query)
            st.markdown("**Here's your answer.**")
            st.write(answer)
            with st.expander("List of documents satisfying this query"):
                docs_list = []
                for k, v in document_dict.items():
                    docs_list.append(str(v[0]))
                dl = list(set(docs_list))
                for i in dl:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(i)
                    with col2:
                        if st.button('Open document '+ i):
                            webbrowser.open_new_tab(document_link_dict[i])
#             st.markdown('Here is the answer to your question: \n ')
#             if query.lower().strip()=='What type of data is collected in Psoriasis trials?'.lower().strip():
#                 st.write('Show me documents related to Psoriasis trials?')
#             else:
#                 st.write(query)
#             answer = clean_answer(answer)
#             st.write(answer.replace('I do not agree to the above statement', 'Cross').replace('I agree to the above statement', 'Tick').replace('Certificate', 'Document'))
            st.markdown('Here are the relevant content from ICFs you can refer..\n')
            for key, value in document_dict.items():
#                 col1, col2 = st.columns(2)
#                 with col1:
                st.markdown('**Document ID:**')#{}'.format(str(value[0])))
                st.write(str(value[0]))
                st.markdown('**Section Name:**')# {}'.format(str(value[1])))
                st.write(str(value[1]))
                st.markdown('**Content:**')# {}'.format(key))
                st.write(key)
                st.write('\n')
#                 with col2:
#                     if st.button('Open document ' + str(value[0])):
#                         webbrowser.open_new_tab(document_link_dict[str(value[0])])
#                     st.markdown(f''' <a href={document_link_dict[str(value[0])]}><button style="background-color:LightSalmon;">Refer to the {str(value[0])} here</button></a>''', unsafe_allow_html=True)
                
        else:
            answer, document_dict = qual_obj.get_results_combined(query)
            st.markdown("**Here's your answer.**")
            st.write(answer)
            with st.expander("List of documents satisfying this query"):
                docs_list = []
                for k, v in document_dict.items():
                    docs_list.append(str(v[0]))
                dl = list(set(docs_list))
                for i in dl:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(i)
                    with col2:
                        if st.button('Open document '+ i):
                            webbrowser.open_new_tab(document_link_dict[i])
#             st.markdown('Here is the answer to your question: \n ')
#             if query.lower().strip()=='What type of data is collected in Psoriasis trials?'.lower().strip():
#                 st.write('Show me documents related to Psoriasis trials?')
#             else:
#                 st.write(query)
#             answer = clean_answer(answer)
#             st.write(answer.replace('I do not agree to the above statement', 'Cross').replace('I agree to the above statement', 'Tick').replace('Certificate', 'Document'))
            st.markdown('Here are the relevant content from documents you can refer..\n')
            
            for key, value in document_dict.items():
#                 col1, col2 = st.columns(2)
#                 with col1:
                st.markdown('**Document ID:**')#{}'.format(str(value[0])))
                st.write(str(value[0]))
                st.markdown('**Section Name:**')# {}'.format(str(value[1])))
                st.write(str(value[1]))
                st.markdown('**Content:**')# {}'.format(key))
                st.write(key)
                st.write('\n')
#                 with col2:
#                     if st.button('Open document ' + str(value[0])):
#                         webbrowser.open_new_tab(document_link_dict[str(value[0])])
#                     st.markdown(f''' <a href={document_link_dict[str(value[0])]}><button style="background-color:LightSalmon;">Refer to the {str(value[0])} here</button></a>''', unsafe_allow_html=True)
                

st.title("Informed Consent Form - Question Answering Platform")
st.caption('Ask any questions related to Informed Consent Forms (ICFs)!')
combined_df = pd.read_csv('NEW_TXT_combined_df.txt')
combined_df = combined_df.fillna('')
# half_ingestion(combined_df)
with st.container():
#     col1, col2 = st.columns([1.5,0.5])
#     with col1:
    #st.markdown('**Enter search query**')
    query = st.text_input('Enter search query', '')
#     with col2:
# #         st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
# #         st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
#         choice = st.radio(
#         "",
#         ('Protocol', 'Certificate', 'Combined'))
        
    process(combined_df, entity_dict, query)
                