import faiss
import pickle
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from vector_engine.utils import vector_search
import gdown

@st.cache
def read_data():
    """Read from Google"""
    id_data = "1-I6sGmfrlGxBvTbhp91GljBp23RNEPrm"
    url = 'https://drive.google.com/uc?id=' + id_data
    output = 'legal.csv'
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)


@st.cache(allow_output_mutation=True)
def load_bert_model(name="distilbert-base-nli-stsb-mean-tokens"):
    """Instantiate a sentence-level DistilBERT model."""
    return SentenceTransformer(name)


@st.cache(allow_output_mutation=True)
def load_faiss_index(path_to_faiss="models/faiss_index.pickle"):
    """Load and deserialize the Faiss index."""
    # Download from Google
    id_data = "1AFNS2rdO4_x_XzKa4nAcMtwreODoeF_P"
    url = 'https://drive.google.com/uc?id=' + id_data
    gdown.download(url, path_to_faiss, quiet=False)
    with open(path_to_faiss, "rb") as h:
        data = pickle.load(h)
    return faiss.deserialize_index(data)


def main():
    # Load data and models
    data = read_data()
    model = load_bert_model()
    faiss_index = load_faiss_index()

    st.title("Search across military documents")

    # User search
    user_input = st.text_area("Search box", "")

    # Filters
    st.sidebar.markdown("**Filters**")
    # filter_year = st.sidebar.slider("Publication year", 2010, 2021, (2010, 2021), 1)
    # filter_citations = st.sidebar.slider("Citations", 0, 250, 0)
    num_results = st.sidebar.slider("Number of search results", 10, 50, 10)

    # Fetch results
    if user_input:
        # Get paper IDs
        D, I = vector_search([user_input], model, faiss_index, num_results)
        # Slice data on year
        # frame = data[
        #     (data.year >= filter_year[0])
        #     & (data.year <= filter_year[1])
        #     & (data.citations >= filter_citations)
        # ]
        frame = data
        # Get individual results
        for id_ in I.flatten().tolist():
            if id_ in set(frame.id):
                f = frame[(frame.id == id_)]
            else:
                continue

            # st.write(
            #     f"""**{f.iloc[0].original_title}**  
            # **Citations**: {f.iloc[0].citations}  
            # **Publication year**: {f.iloc[0].year}  
            # **Abstract**
            # {f.iloc[0].abstract}
            # """
            # )
            st.write(
                f"""**{f.iloc[0].title}**"""
            )
            link = f'**OPEN**: [{f.iloc[0].link}]({f.iloc[0].link})'
            st.markdown(link, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
