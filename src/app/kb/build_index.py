from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import pathlib

DOCS = pathlib.Path(__file__).parent / "docs"

def load_docs():
    loader = DirectoryLoader(str(DOCS), glob="**/*.*", loader_cls=PyPDFLoader)
    pdfs = loader.load()
    html_loader = DirectoryLoader(str(DOCS), glob="**/*.html", loader_cls=UnstructuredHTMLLoader)
    htmls = html_loader.load()
    return pdfs + htmls

def main():
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    Chroma.from_documents(chunks, OpenAIEmbeddings(), collection_name="fleet_kb")
    print("Built vector index.")

if __name__ == "__main__":
    main()
