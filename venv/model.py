###--------BASIC-------------###

#Importing VectorStoreIndex, which helps in Indexing the embedded values.
#Simple Directory Reader helps in reading files in a directory
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
#pprint_response helps in printing the nearest results that has came as a response. (This is optional, if you want the result with max accuracy)
from llama_index.core.response.pprint_utils import pprint_response


###------ADVANCED REQUIREMENT ---------------###

#VectorIndexRetriever is used to retrieve indexes from VectorStoreIndex
from llama_index.core.retrievers import (VectorIndexRetriever)

#RetreiverQueryEngine is used to create an advanced query engine. You can use the query engine if you want to modify the output
from llama_index.core.query_engine import RetrieverQueryEngine
#SimilarityPostProcessor, is used to control the threshold of the similarity score.
from llama_index.core.indices.postprocessor import SimilarityPostprocessor


from dotenv import load_dotenv
import os

load_dotenv()

#Assigning OpenAI key
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


def load_model(path, prompt):
    # Load documents and build index
    documents = SimpleDirectoryReader(
        path
    ).load_data()

    #VectorStoreIndex performs embeddings of the text
    index = VectorStoreIndex.from_documents(documents, show_progress = True)

    #Defining the query engine. This Query Engine performs query over the embedded data. (This is basic query engine)
    #query_engine = index.as_query_engine()  #This is actually a query engine that helps in retrieval

    #If you want to control the output of the query engine you can use advanced query engine

    #First try to modify the retriever with the number of results you can get
    retriever = VectorIndexRetriever(index=index,similarity_top = 4)    #similarity_top gives the top 4 results of the retriever

    #We can use postprocessor, to handle the similarity score
    postprocessor = SimilarityPostprocessor(similarity_cutoff = 0.80)   #similarity_cutoff only takes output more than it

    #Create the advanced query engine. Here the query engine takes retriever as the input.
    query_engine = RetrieverQueryEngine(retriever=retriever,node_postprocessors = postprocessor)

    response = query_engine.query(prompt)

    pprint_response(response,show_source = True)

    print("Response: ", response)
    return index


if __name__ == "__main__":
    prompt = "Give me a comparative analysis of the population of Kolkata and Mumbai"
    load_model(r"/home/supch/RAG_LlamaIndex_Openai/pdfs",prompt)