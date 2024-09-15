# Import necessary libraries
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
import torch
import os

class EasyRAG:
    """
    A simple RAG (Retrieval-Augmented Generation) system that does not use FAISS, but instead relies on in-memory
    retrieval and generation.
    """

    def __init__(
        self, 
        embedding_model_name: str = "all-MiniLM-L6-v2",
        gen_model_name: str = "t5-base",
        device: Optional[str] = None
    ):
        """
        Initializes the EasyRAG class with specified models for embeddings and generation.

        Args:
            embedding_model_name (str): Name of the model to use for creating embeddings.
            gen_model_name (str): Name of the model to use for generating text.
            device (str): Device to run the model on, e.g., "cuda". If None, it will auto-detect.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer(embedding_model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name).to(self.device)
        self.generation_pipeline = pipeline(
            "text2text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            device=0 if self.device == "cuda" else -1
        )

    def _extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """
        Extracts text from a PDF file and splits it into chunks for retrieval.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            List[str]: List of text chunks extracted from the PDF.
        """
        doc = fitz.open(pdf_path)
        text_chunks = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if not text.strip():
                pass
                #print(f"No text found on page {page_num}.")
            else:
                chunks = self._split_text_into_chunks(text)
                text_chunks.extend(chunks)
        if not text_chunks:
            raise Exception("No text extracted from the PDF.")
        return text_chunks

    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
        """
        Splits a large text into smaller chunks for retrieval.

        Args:
            text (str): The text to split.
            chunk_size (int): The maximum size of each chunk in characters.
            chunk_overlap (int): The number of characters to overlap between chunks.

        Returns:
            List[str]: List of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
        )
        langchain_docs = [LangchainDocument(page_content=text)]
        return [chunk.page_content for chunk in text_splitter.split_documents(langchain_docs)]

    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Creates embeddings for a list of text chunks.

        Args:
            texts (List[str]): List of text chunks.

        Returns:
            np.ndarray: Embeddings of the text chunks.
        """
        if not texts:
            return np.array([])
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu().numpy()

    def _retrieve_documents(self, query: str, embeddings: np.ndarray, docs_processed: List[str], top_k: int = 5) -> List[str]:
        """
        Retrieves the top-k most relevant documents based on the query.

        Args:
            query (str): The query to search for.
            embeddings (np.ndarray): The embeddings of the documents.
            docs_processed (List[str]): The list of document chunks.
            top_k (int): Number of top documents to retrieve.

        Returns:
            List[str]: List of retrieved document chunks.
        """
        if embeddings.size == 0:
            raise Exception("No embeddings available to perform retrieval.")

        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
        similarities = cosine_similarity(query_embedding, embeddings)
        top_k = min(top_k, len(docs_processed))
        top_k_indices = similarities.argsort()[0][-top_k:][::-1]
        return [docs_processed[i] for i in top_k_indices]

    def ask_question_with_pdf(self, question: str, pdf_path: str, top_k: int = 5) -> str:
        """
        Generates a response for the given question using the RAG model, with information retrieved from a PDF.

        Args:
            question (str): The question or prompt provided by the user.
            pdf_path (str): Path to the PDF file containing relevant information.
            top_k (int): Number of top documents to retrieve and use for generating the answer.

        Returns:
            str: Generated response to the question, augmented with information retrieved from the PDF.
        """
        if not os.path.exists(pdf_path):
            return "PDF file not found."

        # Extract and chunk text from the PDF
        pdf_chunks = self._extract_text_from_pdf(pdf_path)
        if not pdf_chunks:
            return "No text could be extracted from the PDF to answer the question."

        # Create embeddings for the chunks
        embeddings = self._create_embeddings(pdf_chunks)
        if embeddings.size == 0:
            return "Failed to create embeddings for the PDF content."

        # Retrieve relevant chunks based on the question
        retrieved_docs = self._retrieve_documents(question, embeddings, pdf_chunks, top_k=top_k)
        if not retrieved_docs:
            return "No relevant information found in the PDF to answer the question."

        # Prepare the context for the generation
        context = "\n".join(retrieved_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        # Generate the answer
        try:
            answer = self.generation_pipeline(prompt)
            return answer[0]["generated_text"]
        except Exception as e:
            raise Exception(f"An error occurred during text generation: {e}")
            return "Unable to generate an answer at this time."