import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def last_token_pool(last_hidden_states: torch.Tensor,
                   attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last token for Qwen model"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with instruction following Qwen format"""
    return f'Instruct: {task_description}\nQuery: {query}'

class CosineSimilarityZeroShot:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B", device: str = "cuda", use_qwen: bool = True):
        self.model_name = model_name
        self.device = device
        self.use_qwen = use_qwen
        
        if use_qwen:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        else:
            self.model = SentenceTransformer(model_name).to(device)

    def _get_embeddings_qwen(self, texts: List[str], is_query: bool = False, task_description: Optional[str] = None) -> np.ndarray:
        """Get embeddings using Qwen model with last token pooling"""
        if is_query and task_description:
            # Format queries with instruction
            texts = [get_detailed_instruct(task_description, text) for text in texts]
        
        embeddings = []
        batch_size = 32  # Process in batches to avoid OOM
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize with padding and truncation
                batch_dict = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=8192,  # Qwen's max length
                    return_tensors="pt"
                )
                
                # Move to device
                batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
                
                # Get model outputs
                outputs = self.model(**batch_dict)
                
                # Get embeddings using last token pooling
                batch_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                
                # Normalize embeddings
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                
                embeddings.extend(batch_embeddings.cpu().numpy())
                
        return np.array(embeddings)

    def _get_embeddings_mpnet(self, texts: List[str], is_query: bool = False, task_description: Optional[str] = None) -> np.ndarray:
        """Get embeddings using sentence-transformers model"""
        if is_query and task_description:
            # Format queries with instruction for MPNet
            texts = [f"{task_description} {text}" for text in texts]
        return self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True).cpu().numpy()

    def _get_embeddings(self, texts: List[str], is_query: bool = False, task_description: Optional[str] = None) -> np.ndarray:
        """Get embeddings using selected model"""
        if self.use_qwen:
            return self._get_embeddings_qwen(texts, is_query, task_description)
        else:
            return self._get_embeddings_mpnet(texts, is_query, task_description)

    def _format_plain(self, terms: List[str], types: List[str]) -> tuple:
        """Format inputs without special prompting"""
        return terms, types, None, None

    def _format_prompted_style1(self, terms: List[str], types: List[str]) -> tuple:
        """Format inputs with first prompting style"""
        if self.use_qwen:
            formatted_terms = [f"Define the term: {term}" for term in terms]
            formatted_types = [f"Category description: {type_}" for type_ in types]
        else:
            formatted_terms = [f"What is the meaning of '{term}'?" for term in terms]
            formatted_types = [f"This category represents {type_}" for type_ in types]
        return formatted_terms, formatted_types, None, None

    def _format_prompted_style2(self, terms: List[str], types: List[str]) -> tuple:
        """Format inputs with second prompting style using instruction format"""
        if self.use_qwen:
            term_task = "Given a term, provide its semantic meaning and key characteristics"
            type_task = "Given a category type, describe its defining features and scope"
        else:
            term_task = "Analyze this term and describe its key characteristics and domain"
            type_task = "Explain this category's scope and what concepts it encompasses"
        
        return terms, types, term_task, type_task

    def _select_types_adaptive(self, scores: np.ndarray) -> List[int]:
        """Select single type with highest score"""
        selected_index = np.argmax(scores)
        return [selected_index]

    def predict(self, terms_data: List[Dict[str, Any]], types: List[str], 
                output_file: str, prompt_style: str = "plain") -> None:
        # Extract terms
        terms = [item["term"] for item in terms_data]
        
        # Format inputs based on prompt style
        if prompt_style == "plain":
            formatted_terms, formatted_types, term_task, type_task = self._format_plain(terms, types)
        elif prompt_style == "style1":
            formatted_terms, formatted_types, term_task, type_task = self._format_prompted_style1(terms, types)
        else:  # style2
            formatted_terms, formatted_types, term_task, type_task = self._format_prompted_style2(terms, types)
        
        # Get embeddings
        print("Computing term embeddings...")
        term_embeddings = self._get_embeddings(formatted_terms, is_query=True, task_description=term_task)
        
        print("Computing type embeddings...")
        type_embeddings = self._get_embeddings(formatted_types, is_query=True, task_description=type_task)
        
        # Compute cosine similarities
        print("Computing cosine similarities...")
        similarity_matrix = cosine_similarity(term_embeddings, type_embeddings)
        
        # Make predictions
        predictions = []
        for i, term_data in enumerate(terms_data):
            scores = similarity_matrix[i]
            selected_indices = self._select_types_adaptive(scores)
            selected_types = [types[idx] for idx in selected_indices]
            
            predictions.append({
                "id": term_data["id"],
                "types": selected_types
            })
        
        # Save predictions
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        print(f"Predictions saved to {output_file}")

def run_pipeline(data_dir: str, output_dir: str):
    # Get domain name from data_dir
    domain = Path(data_dir).name  # Get B4_blind from B4_blind
    domain_short = domain.split('_')[0]  # Get B4 from B4_blind
    
    # Load data with correct file names
    with open(Path(data_dir) / f"{domain_short}-Blind-Terms.json", 'r', encoding='utf-8') as f:
        terms_data = json.load(f)
    
    with open(Path(data_dir) / f"{domain_short}-Blind-Types.txt", 'r', encoding='utf-8') as f:
        types = [line.strip() for line in f if line.strip()]
    
    # Run MPNet version
    print("\nRunning MPNet cosine similarity...")
    model_mpnet = CosineSimilarityZeroShot(
        model_name="sentence-transformers/all-mpnet-base-v2",
        use_qwen=False
    )
    
    # Plain MPNet
    model_mpnet.predict(
        terms_data=terms_data,
        types=types,
        output_file=str(Path(output_dir) / "predictions_cosine_mpnet_plain.json"),
        prompt_style="plain"
    )
    
    # Prompted MPNet Style 1
    model_mpnet.predict(
        terms_data=terms_data,
        types=types,
        output_file=str(Path(output_dir) / "predictions_cosine_mpnet_prompted_style1.json"),
        prompt_style="style1"
    )
    
    # Prompted MPNet Style 2
    model_mpnet.predict(
        terms_data=terms_data,
        types=types,
        output_file=str(Path(output_dir) / "predictions_cosine_mpnet_prompted_style2.json"),
        prompt_style="style2"
    )
    
    # Run Qwen version
    print("\nRunning Qwen cosine similarity...")
    model_qwen = CosineSimilarityZeroShot(
        model_name="Qwen/Qwen3-Embedding-4B",
        use_qwen=True
    )
    
    # Plain Qwen
    model_qwen.predict(
        terms_data=terms_data,
        types=types,
        output_file=str(Path(output_dir) / "predictions_cosine_qwen_plain.json"),
        prompt_style="plain"
    )
    
    # Prompted Qwen Style 1
    model_qwen.predict(
        terms_data=terms_data,
        types=types,
        output_file=str(Path(output_dir) / "predictions_cosine_qwen_prompted_style1.json"),
        prompt_style="style1"
    )
    
    # Prompted Qwen Style 2
    model_qwen.predict(
        terms_data=terms_data,
        types=types,
        output_file=str(Path(output_dir) / "predictions_cosine_qwen_prompted_style2.json"),
        prompt_style="style2"
    )

if __name__ == "__main__":
    # Example usage
    data_dir = "/path/to/data"
    output_dir = "/path/to/output"
    run_pipeline(data_dir, output_dir) 