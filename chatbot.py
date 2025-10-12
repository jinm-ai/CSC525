"""Customer support chatbot using a fine-tuned language model with enhanced anti-repetition and RAG"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import requests
import os
import time
from bs4 import BeautifulSoup
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class KnowledgeBase:
    def __init__(self, embedding_model="sentence-transformers/all-mpnet-base-v2", index_path="./knowledge_base"):
        """Initialize the knowledge base with an embedding model and storage location"""
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.index_path = index_path
        self.urls_crawled = set()
        self.db = None
        
        # Create the directory if it doesn't exist
        os.makedirs(index_path, exist_ok=True)
        
        # Try to load an existing index
        if os.path.exists(os.path.join(index_path, "index.faiss")):
            self.load_knowledge_base()
        
    def crawl_website(self, urls, max_depth=1, max_pages=100):
        """Crawl websites from the provided URLs"""
        urls_to_visit = [(url, 0) for url in urls]  # (url, depth)
        visited = set()
        
        all_docs = []
        page_count = 0
        
        while urls_to_visit and page_count < max_pages:
            url, depth = urls_to_visit.pop(0)
            
            if url in visited or depth > max_depth:
                continue
                
            visited.add(url)
            print(f"Crawling {url} (depth {depth})")
            
            try:
                loader = WebBaseLoader(url)
                url_docs = loader.load()
                all_docs.extend(url_docs)
                self.urls_crawled.add(url)
                page_count += 1
                
                # Rate limiting to be respectful
                time.sleep(1)
                
                if depth < max_depth:
                    # Find links to follow
                    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                    soup = BeautifulSoup(response.text, "html.parser")
                    links = soup.find_all("a", href=True)
                    
                    base_url = "/".join(url.split("/")[:3])  # e.g., https://www.amazon.com
                    
                    for link in links:
                        href = link["href"]
                        
                        # Convert relative URLs to absolute
                        if href.startswith("/"):
                            href = base_url + href
                        
                        # Only follow links from the same domain
                        if href.startswith(base_url) and href not in visited:
                            urls_to_visit.append((href, depth + 1))
                
            except Exception as e:
                print(f"Error crawling {url}: {e}")
        
        return all_docs
    
    def process_documents(self, documents):
        """Process and split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def build_knowledge_base(self, urls):
        """Build the knowledge base from the provided URLs"""
        print("Starting to build knowledge base...")
        # Crawl websites
        documents = self.crawl_website(urls)
        
        # Process documents
        print(f"Processing {len(documents)} documents...")
        chunks = self.process_documents(documents)
        
        # Build vector store
        if not chunks:
            print("No documents were retrieved. Check the URLs and try again.")
            return False
        
        print(f"Creating vector store with {len(chunks)} chunks...")
        self.db = FAISS.from_documents(chunks, self.embedding_model)
        
        # Save the knowledge base
        self.save_knowledge_base()
        return True
    
    def save_knowledge_base(self):
        """Save the knowledge base to disk"""
        if self.db is not None:
            self.db.save_local(self.index_path)
            print(f"Knowledge base saved to {self.index_path}")
            
            # Save the list of crawled URLs
            with open(os.path.join(self.index_path, "crawled_urls.txt"), "w") as f:
                for url in self.urls_crawled:
                    f.write(f"{url}\n")
    
    def load_knowledge_base(self):
        """Load the knowledge base from disk"""
        try:
            self.db = FAISS.load_local(self.index_path, self.embedding_model)
            print(f"Knowledge base loaded from {self.index_path}")
            
            # Load the list of crawled URLs
            crawled_urls_path = os.path.join(self.index_path, "crawled_urls.txt")
            if os.path.exists(crawled_urls_path):
                with open(crawled_urls_path, "r") as f:
                    self.urls_crawled = set(line.strip() for line in f)
            
            return True
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return False
    
    def get_retriever(self, k=3):
        """Get a retriever for the knowledge base"""
        if self.db is None:
            print("Knowledge base not initialized. Please build or load a knowledge base first.")
            return None
        
        return self.db.as_retriever(search_kwargs={"k": k})
    
    def search_documents(self, query, k=3):
        """Search for relevant documents based on a query"""
        if self.db is None:
            return []
        
        docs = self.db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]


class CustomerSupportChatbot:
    def __init__(self, model_path='c:/Users/chaid/Jin/trained-customer-support-bot', knowledge_base_path="./knowledge_base"):
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
        # Use a regex pattern to detect DM requests more broadly
        self.dm_pattern = re.compile(r'\b(dm|direct message)\b', re.IGNORECASE)
        self.history = []  # Store conversation history
        
        # Initialize knowledge base
        self.knowledge_base = KnowledgeBase(index_path=knowledge_base_path)

    def generate_response(self, customer_message, max_length=250, temperature=0.8, max_retries=2):
        """Generate a response, with a retry mechanism to avoid canned DM requests."""
        # Append the latest customer message to history
        self.history.append(f"Customer: {customer_message}")
        
        # Build the prompt from the last N turns (e.g., 3 exchanges)
        history_prompt = "\n".join(self.history[-6:])  # 3 customer/support pairs
        
        # Get relevant context from knowledge base
        context_docs = self.knowledge_base.search_documents(customer_message, k=3)
        context_section = ""
        if context_docs:
            # Frame the context clearly for the model
            context_section = "---CONTEXT---\n" + "\n---\n".join(context_docs) + "\n---END CONTEXT---"
        
        # Create a more explicit and structured input prompt
        instruction = (
            "You are a helpful Amazon customer support assistant. "
            "Use the provided context to answer the customer's question directly. "
            "If the context does not contain the answer, state that you don't have that information. "
            "Do not ask the user to contact support through another channel or to send a DM."
        )
        
        input_text = f"{instruction}\n\n{context_section}\n\n---CONVERSATION HISTORY---\n{history_prompt}\nSupport:"
        
        response = "" # Initialize response to handle cases where all retries fail
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
                    repetition_penalty=1.5, # Increased penalty
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Isolate the newly generated text more reliably
            response_start = generated_text.find(history_prompt) + len(history_prompt)
            response = generated_text[response_start:].replace("Support:", "").strip()

            # If the response does not contain a DM request, clean and return it
            if not self.dm_pattern.search(response):
                # After generating response, add it to history
                self.history.append(f"Support: {response}")
                response = self.clean_response(response)
                response = self.correct_grammar(response)  # Correct grammar
                return response

        # If all retries fail, return the last generated response after cleaning
        response = self.clean_response(response)
        response = self.correct_grammar(response)  # Correct grammar
        return response
    
    def clean_response(self, text):
        """More aggressive post-processing to remove artifacts."""
        # 1. Remove the [USER] token (case-insensitive)
        text = re.sub(r'\[USER\]', '', text, flags=re.IGNORECASE).strip()
        
        # 2. Remove signature-like gibberish and random character strings
        # This regex is more aggressive and targets strings of capital letters/spaces
        text = re.sub(r'\b([A-Z]{2,}\s+){3,}[A-Z]{2,}\b', '', text) # e.g., JEZ HM KI MH...
        text = re.sub(r'\b[a-zA-Z0-9]{8,}\b', '', text)
        text = re.sub(r'\b[a-zA-Z]*[0-9]+[a-zA-Z0-9]*\b', '', text)
        text = re.sub(r'\s-\w+\b', '', text) # Removes agent tags like " -EmmaW"
        text = re.sub(r'-[A-Z]{2,}\b', '', text) # Removes tags like "-LHJBY"

        # 3. Remove duplicate sentences
        sentences = re.split(r'([.!?])\s*', text)
        if len(sentences) > 1:
            grouped_sentences = ["".join(s).strip() for s in zip(sentences[0::2], sentences[1::2])]
            # Remove sentences that are too short or look incomplete
            grouped_sentences = [s for s in grouped_sentences if len(s.split()) > 3 and not re.search(r"\bwe'd your and\b", s)]
            unique_sentences = list(dict.fromkeys(s for s in grouped_sentences if s))
            text = ' '.join(unique_sentences)
        
        # 4. Final cleanup for extra whitespace and dangling punctuation
        text = ' '.join(text.split()).strip()
        text = re.sub(r'\s+([.!?])', r'\1', text) # remove space before punctuation

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
        
        while True:
            user_input = input("\nCustomer: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Thank you for chatting with us!")
                break
                
            response = self.generate_response(user_input)
            print(f"\nSupport: {response}")


def setup_knowledge_base():
    """Set up the knowledge base with Amazon help pages"""
    amazon_urls = [
        "https://www.amazon.ca/gp/help/customer/display.html?nodeId=GGE5X8EV7VNVTK6R",
        "https://www.amazon.com/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7",
        "https://www.amazon.ca/gp/help/customer/display.html?nodeId=GSD587LKW72HKU2V",
        "https://www.amazon.com/gp/help/customer/display.html",
        "https://www.amazon.com/gp/help/customer/display.html?nodeId=200127470"
    ]
    
    kb = KnowledgeBase()
    kb.build_knowledge_base(amazon_urls)
    return kb


def main():
    """Main function to run the chatbot"""
    # Check if knowledge base exists, build if needed
    kb_path = "./knowledge_base"
    if not os.path.exists(os.path.join(kb_path, "index.faiss")):
        print("Building knowledge base from Amazon help pages...")
        setup_knowledge_base()
    
    chatbot = CustomerSupportChatbot(knowledge_base_path=kb_path)
    chatbot.chat_session()


if __name__ == "__main__":
    main()