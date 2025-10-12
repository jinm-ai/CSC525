"""Customer support chatbot using a fine-tuned language model with RAG capabilities"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import WebBaseLoader
import os

class AmazonKnowledgeBase:
    """Handles retrieval-augmented generation with Amazon help content"""
    
    def __init__(self, urls=None):
        """Initialize the knowledge base with Amazon help URLs"""
        self.urls = urls or [
            'https://www.amazon.com/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7',  # Returns
            'https://www.amazon.com/gp/help/customer/display.html?nodeId=201889750',  # Orders
            'https://www.amazon.com/gp/help/customer/display.html?nodeId=201117750'   # Shipping
        ]
        self.vector_store = None
        
    def build_knowledge_base(self, force_rebuild=False):
        """Build or load the knowledge base"""
        cache_path = "amazon_knowledge_base.faiss"
        
        # Check if cached knowledge base exists
        if os.path.exists(cache_path) and not force_rebuild:
            print("Loading cached knowledge base...")
            try:
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                self.vector_store = FAISS.load_local(cache_path, embeddings)
                print("Knowledge base loaded successfully!")
                return True
            except Exception as e:
                print(f"Failed to load cached knowledge base: {e}")
                # Fall back to rebuilding
        
        print(f"Building knowledge base from {len(self.urls)} URLs...")
        all_docs = []
        
        # Fetch and process each URL
        for url in self.urls:
            try:
                print(f"Fetching content from {url}...")
                loader = WebBaseLoader(url)
                docs = loader.load()
                
                # If WebBaseLoader fails, try BeautifulSoup directly
                if not docs:
                    content = self._fetch_with_bs4(url)
                    if content:
                        docs = [Document(page_content=content, metadata={"source": url})]
                        
                all_docs.extend(docs)
                print(f"Added {len(docs)} document(s) from {url}")
            except Exception as e:
                print(f"Error processing {url}: {e}")
        
        if not all_docs:
            print("Failed to build knowledge base - no content retrieved")
            return False
            
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_chunks = text_splitter.split_documents(all_docs)
        print(f"Split content into {len(all_chunks)} chunks")
        
        # Create vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(all_chunks, embeddings)
        
        # Save for future use
        try:
            self.vector_store.save_local(cache_path)
            print(f"Knowledge base saved to {cache_path}")
        except Exception as e:
            print(f"Could not save knowledge base: {e}")
            
        return True
    
    def _fetch_with_bs4(self, url):
        """Fallback method to fetch content using BeautifulSoup"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try to find main content - Amazon help pages often have a help-content div
            main_content = soup.find('div', class_='help-content')
            if main_content:
                return main_content.get_text(separator=' ', strip=True)
            
            # Fallback to body content
            body = soup.find('body')
            if body:
                return body.get_text(separator=' ', strip=True)
                
            return None
        except Exception:
            return None
            
    def retrieve_relevant_context(self, query, k=3):
        """Retrieve relevant context for a query"""
        if not self.vector_store:
            print("Knowledge base not initialized")
            return []
            
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error retrieving from knowledge base: {e}")
            return []


class CustomerSupportChatbot:
    def __init__(self, model_path='c:/Users/chaid/Jin/trained-customer-support-bot'):
        """Initialize the chatbot with the trained model and tokenizer"""
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad_token to eos_token for GPT-like models if it's not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        print("Model loaded successfully!")
        
        # Initialize RAG knowledge base
        self.knowledge_base = AmazonKnowledgeBase()
        self.use_rag = self.knowledge_base.build_knowledge_base()
        
        # Other initializations
        self.dm_pattern = re.compile(r'\b(dm|direct message)\b', re.IGNORECASE)
        self.history = []  # Store conversation history

    def generate_response(self, customer_message, max_length=250, temperature=0.8, max_retries=2):
        """Generate a response, with optional RAG enhancement."""
        # Append the latest customer message to history
        self.history.append(f"Customer: {customer_message}")
        
        # Try RAG-enhanced generation if available
        if self.use_rag:
            try:
                # Get relevant documents from knowledge base
                context_docs = self.knowledge_base.retrieve_relevant_context(customer_message)
                if context_docs:
                    # Build context from retrieved documents
                    context_text = "\n".join([doc.page_content for doc in context_docs])
                    
                    # Create RAG-enhanced prompt with retrieved context
                    rag_prompt = (
                        f"Based on the following Amazon information:\n"
                        f"---\n{context_text}\n---\n\n"
                        f"Please answer the customer question professionally:\n"
                        f"Customer: {customer_message}\nSupport:"
                    )
                    
                    # Generate with RAG context
                    response = self._generate_with_prompt(rag_prompt, max_length, temperature)
                    
                    # If response seems relevant, return it
                    if not self.dm_pattern.search(response):
                        cleaned_response = self.clean_response(response)
                        final_response = self.correct_grammar(cleaned_response)
                        self.history.append(f"Support: {final_response}")
                        return final_response
            except Exception as e:
                print(f"RAG generation failed: {e}")
        
        # Fall back to standard generation if RAG fails or is unavailable
        return self._generate_standard_response(customer_message, max_length, temperature, max_retries)
    
    def _generate_with_prompt(self, prompt, max_length, temperature):
        """Generate response with a specific prompt"""
        encoded_input = self.tokenizer(prompt, return_tensors='pt')
        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # Extract just the response part after "Support:"
        response = generated_text[len(prompt):].strip()
        return response
        
    def _generate_standard_response(self, customer_message, max_length, temperature, max_retries):
        """Generate a response using the base model without RAG"""
        # Build the prompt from the last N turns
        history_prompt = "\n".join(self.history[-6:])  # 3 customer/support pairs
        input_text = f"{history_prompt}\nSupport:"
        
        for attempt in range(max_retries + 1):
            # On retries, use a higher temperature to encourage diversity
            current_temp = temperature + (attempt * 0.15)
            
            encoded_input = self.tokenizer(input_text, return_tensors='pt')
            input_ids = encoded_input['input_ids'].to(self.device)
            attention_mask = encoded_input['attention_mask'].to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + max_length,
                    temperature=current_temp,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = generated_text[len(input_text):].strip()
            
            # If the response does not contain a DM request, clean and return it
            if not self.dm_pattern.search(response):
                # After generating response, add it to history
                cleaned_response = self.clean_response(response)
                final_response = self.correct_grammar(cleaned_response)
                self.history.append(f"Support: {final_response}")
                return final_response
                
        # If all retries fail, return the last generated response after cleaning
        cleaned_response = self.clean_response(response)
        final_response = self.correct_grammar(cleaned_response)
        self.history.append(f"Support: {final_response}")
        return final_response
    
    def clean_response(self, text):
        """More aggressive post-processing to remove artifacts."""
        # 1. Remove the [USER] token (case-insensitive)
        text = re.sub(r'\[USER\]', '', text, flags=re.IGNORECASE).strip()
        
        # 2. Remove signature-like gibberish
        text = re.sub(r'\b[a-zA-Z0-9]{8,}\b', '', text)
        text = re.sub(r'\b[a-zA-Z]*[0-9]+[a-zA-Z0-9]*\b', '', text)
        text = re.sub(r'\s-\w+\b', '', text) # Removes agent tags like " -EmmaW"
        text = re.sub(r'-[A-Z]{2,}\b', '', text) # Removes tags like "-LHJBY"

        # 3. Remove duplicate sentences
        sentences = re.split(r'([.!?])\s*', text)
        if len(sentences) > 1:
            grouped_sentences = ["".join(s).strip() for s in zip(sentences[0::2], sentences[1::2])]
            grouped_sentences = [s for s in grouped_sentences if len(s.split()) > 3 and not re.search(r"\bwe'd your and\b", s)]
            unique_sentences = list(dict.fromkeys(s for s in grouped_sentences if s))
            text = ' '.join(unique_sentences)
        
        # 4. Final cleanup for extra whitespace and dangling punctuation
        text = ' '.join(text.split()).strip()
        text = re.sub(r'\s+([.!?])', r'\1', text)

        return text if text else "I'm sorry, I'm having trouble with that request. Could you please rephrase?"

    def correct_grammar(self, text):
        """Use LanguageTool API to correct grammar and sentence fragments."""
        url = "https://api.languagetoolplus.com/v2/check"
        data = {
            "text": text,
            "language": "en-US"
        }
        try:
            response = requests.post(url, data=data)
            matches = response.json().get("matches", [])
            for match in reversed(matches):  # Reverse to not mess up offsets
                replacement = match.get("replacements", [])
                if replacement:
                    start = match["offset"]
                    end = start + match["length"]
                    text = text[:start] + replacement[0]["value"] + text[end:]
            return text
        except Exception as e:
            print(f"Grammar correction failed: {e}")
            return text

    def chat_session(self):
        """Run an interactive chat session in the console"""
        print("Customer Support Chatbot - Type 'quit' to exit")
        print("=" * 50)
        if self.use_rag:
            print("RAG knowledge base loaded: The chatbot can answer questions about Amazon policies")
        else:
            print("Running without RAG knowledge base")
        print("=" * 50)
        
        while True:
            user_input = input("\nCustomer: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Thank you for chatting with us!")
                break
                
            response = self.generate_response(user_input)
            print(f"\nSupport: {response}")


def main():
    """Main function to run the chatbot"""
    chatbot = CustomerSupportChatbot()
    chatbot.chat_session()


if __name__ == "__main__":
    main()