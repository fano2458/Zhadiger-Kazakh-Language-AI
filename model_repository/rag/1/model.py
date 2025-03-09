import torch
import numpy as np
import pandas as pd
import re
import triton_python_backend_utils as pb_utils

from transformers import AutoTokenizer, AutoModel

torch.set_float32_matmul_precision('high')

import sys
sys.path.append('/assets')

# Simple document class to replace Haystack's Document
class Document:
    def __init__(self, content, id, meta=None):
        self.content = content
        self.id = id
        self.meta = meta or {}

class TritonPythonModel:
    def initialize(self, args):
        model_name = 'intfloat/multilingual-e5-large-instruct'
        self.file_path = "files.csv"
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling layer to convert token embeddings to sentence embeddings
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0] 
        
        # Calculate attention mask for proper averaging (ignoring padding tokens)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings weighted by attention mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Return average (sum / token count)
        return sum_embeddings / sum_mask
    
    def encode(self, texts, batch_size=32):
        """
        Custom encode method to get embeddings from AutoModel
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Add special instruction prefix for e5 models
            prefixed_batch = ["query: " + text for text in batch_texts]
            
            # Tokenize
            encoded_input = self.tokenizer(
                prefixed_batch, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)
            
            # Compute token embeddings (with gradient computation disabled for efficiency)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            # Perform mean pooling
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize the embeddings (important for cosine similarity)
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            # Move to CPU and convert to numpy
            all_embeddings.append(sentence_embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)

    def custom_sent_tokenize(self, text):
        """
        Custom sentence tokenizer using regex to replace NLTK's sent_tokenize
        """
        # Basic sentence splitting on .!? followed by a space and uppercase letter
        # This is a simplified approach and might not work perfectly for all languages
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Further split on common sentence terminators
        result = []
        for sentence in sentences:
            # Split on periods, exclamation marks, and question marks
            parts = re.split(r'(?<=[.!?])\s+', sentence)
            result.extend(parts)
            
        # Remove empty strings and strip whitespace
        return [s.strip() for s in result if s.strip()]

    def process_chunk(self, file_path, text):
        # Use the custom sentence tokenizer
        sentences = self.custom_sent_tokenize(text)
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Iterate over the sentences and combine them into chunks
        for sentence in sentences:
            sentence_length = len(self.tokenizer(sentence)["input_ids"])
            
            if current_length + sentence_length <= 490:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Save the current chunk as a single text
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        # Append any remaining sentences as a final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Ensure we don't have extremely small chunks by merging them with previous chunks
        if len(chunks) > 1 and len(self.tokenizer(chunks[-1])["input_ids"]) < 100:
            chunks[-2] = chunks[-2] + " " + chunks[-1]
            chunks.pop(-1)
        
        return [(file_path, chunk) for chunk in chunks]
    
    def builder(self, document_embeddings, chunked_df):
        documents = []
        for i in range(document_embeddings.shape[0]):
            file_path = chunked_df.iloc[i]['file_path']
            chunk = chunked_df.iloc[i]['chunk']
            documents.append(Document(content=chunk, id=str(i), meta={"file_path": file_path}))

        return documents
        
    def retrieve_documents(self, document_embeddings, documents, user_request, top_k=5):
        """
        Retrieve most similar documents using PyTorch cosine similarity
        """
        # Convert query to embedding
        query_embedding = self.encode(user_request)
        query_embedding_tensor = torch.tensor(query_embedding).float()
        
        # Convert document embeddings to tensor
        doc_embeddings_tensor = torch.tensor(document_embeddings).float()
        
        # Normalize embeddings for cosine similarity (they should be already normalized, but just in case)
        query_embedding_norm = torch.nn.functional.normalize(query_embedding_tensor, p=2, dim=1).squeeze(0)
        doc_embeddings_norm = torch.nn.functional.normalize(doc_embeddings_tensor, p=2, dim=1)
        
        # Calculate cosine similarity
        similarities = torch.matmul(doc_embeddings_norm, query_embedding_norm)
        
        # Get top k indices
        top_indices = torch.argsort(similarities, descending=True)[:top_k].cpu().numpy()
        
        # Get the scores
        top_scores = similarities[top_indices].cpu().numpy()
        
        # Get the documents
        results = []
        for idx, score in zip(top_indices, top_scores):
            doc = documents[idx]
            doc.score = float(score)
            results.append(doc)
            
        return results

    def execute(self, requests):
        responses = []

        for request in requests:
            user_request = pb_utils.get_input_tensor_by_name(request, "user_request").as_numpy()
            texts = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            file_paths = pb_utils.get_input_tensor_by_name(request, "file_paths").as_numpy()
            user_request = [el.decode() for el in user_request]
            texts = [el.decode() for el in texts]
            file_paths = [el.decode() for el in file_paths]

            resulting_chunks = []

            for i, text in enumerate(texts):
                resulting_chunks.append(self.process_chunk(file_paths[i], text))

            chunked_df = pd.DataFrame([item for sublist in resulting_chunks for item in sublist], columns=['file_path', 'chunk'])
            document_embeddings = self.encode(chunked_df["chunk"].tolist())

            documents = self.builder(document_embeddings, chunked_df)
            retrieval_results = self.retrieve_documents(document_embeddings, documents, user_request[0])

            with open("/assets/rag_logs.txt", "a") as f:
                f.write(f"User request: {user_request}\n")
                f.write(f"Retrieved documents: {[doc.content[:100] + '...' for doc in retrieval_results]}\n")

            # Get the content of retrieved documents as response
            retrieved_texts = [doc.content for doc in retrieval_results]
            output_tensor = pb_utils.Tensor("output", np.array(retrieved_texts, dtype=np.object_))
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses