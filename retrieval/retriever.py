#function to for retriving the data  and k is used to retriver the 4 top most value 


def fetcher(vector_store, query):
    try:
        return vector_store.similarity_search(
            
            search_query = query,
            k = 4 
            
            )
    except Exception as e:
        print(f"Error fetching data: {e}")
        return "Unable to fetch data"
