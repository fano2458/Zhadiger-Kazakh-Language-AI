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
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device('cpu')
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
            
            # Tokenize with explicit max_length and truncation
            encoded_input = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)
            
            with open("/assets/rag_logs.txt", "a") as f:
                f.write(f"Processing batch of {len(batch_texts)} texts\n")
                f.write(f"Encoded input shape: {encoded_input['input_ids'].shape}\n")

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

    def process_chunk(self, file_path, text):
        """
        Split text into chunks of approximately 200 words each
        """
        # Split the text into words
        words = text.split()
        
        # Define the target chunk size (in words)
        words_per_chunk = 200
        
        chunks = []
        
        # Process the text in chunks of approximately 200 words
        for i in range(0, len(words), words_per_chunk):
            chunk_words = words[i:i+words_per_chunk]
            chunk_text = " ".join(chunk_words)
            
            # Make sure the chunk isn't too large for the model
            if len(self.tokenizer.encode(chunk_text)) > 512:
                tokens = self.tokenizer.encode(chunk_text, add_special_tokens=False)
                tokens = tokens[:500]  # Leave room for special tokens
                chunk_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                
            chunks.append(chunk_text)
        
        # Log the chunks for debugging
        with open("/assets/rag_logs.txt", "a") as f:
            f.write(f"Created {len(chunks)} chunks for file {file_path}\n")
            if chunks:
                f.write(f"Average chunk length (words): {sum(len(c.split()) for c in chunks) / len(chunks):.1f}\n")
        
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
        
        # Normalize embeddings for cosine similarity
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
            try:
                # Clear log file at the beginning of each request for cleaner logs
                with open("/assets/rag_logs.txt", "w") as f:
                    f.write("Starting new RAG request processing\n")
                
                user_request = pb_utils.get_input_tensor_by_name(request, "user_request").as_numpy()
                texts = pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
                file_paths = []
                user_request = [el.decode() for el in user_request]
                texts = [el.decode() for el in texts]
                # Generate file paths for each text as p1, p2, etc.
                if len(file_paths) == 0 or len(file_paths) != len(texts):
                    file_paths = [f'p{i+1}' for i in range(len(texts))]
                else:
                    file_paths = [path.decode() if isinstance(path, bytes) else path for path in file_paths]

                with open("/assets/rag_logs.txt", "a") as f:
                    f.write(f"User request: {user_request}\n")
                    f.write(f"Received {len(texts)} documents\n")
                    f.write(f"File paths: {file_paths}\n")

                # Process all documents and collect chunks
                all_chunks = []
                for i, text in enumerate(texts):
                    chunks = self.process_chunk(file_paths[i], text)
                    all_chunks.extend(chunks)
                    with open("/assets/rag_logs.txt", "a") as f:
                        f.write(f"Document {i} was split into {len(chunks)} chunks\n")

                chunked_df = pd.DataFrame(all_chunks, columns=['file_path', 'chunk'])
                
                with open("/assets/rag_logs.txt", "a") as f:
                    f.write(f"Total chunks created: {len(chunked_df)}\n")
                
                # Generate embeddings for all chunks
                document_embeddings = self.encode(chunked_df["chunk"].tolist())

                # Build document objects
                documents = self.builder(document_embeddings, chunked_df)
                
                # Retrieve most relevant documents
                retrieval_results = self.retrieve_documents(document_embeddings, documents, user_request[0])

                with open("/assets/rag_logs.txt", "a") as f:
                    f.write(f"Retrieved {len(retrieval_results)} documents\n")
                    for i, doc in enumerate(retrieval_results):
                        f.write(f"Doc {i} score: {doc.score}, from: {doc.meta['file_path']}\n")
                        f.write(f"Content preview: {doc.content[:100]}...\n\n")

                # Get the content of retrieved documents as response
                retrieved_texts = [doc.content for doc in retrieval_results]
                output_tensor = pb_utils.Tensor("output", np.array(retrieved_texts, dtype=np.object_))
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(response)
                
            except Exception as e:
                # Log any errors that occur during processing
                with open("/assets/rag_logs.txt", "a") as f:
                    f.write(f"Error during execution: {str(e)}\n")
                    import traceback
                    f.write(traceback.format_exc())
                
                # Return an error message in the response
                empty_response = np.array(["Error during document retrieval"], dtype=np.object_)
                output_tensor = pb_utils.Tensor("output", empty_response)
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(response)

        return responses