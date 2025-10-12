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
        else:
            print("No existing knowledge base found. Building from manual file...")
            self.build_knowledge_base_from_manual("c:/Users/chaid/Jin/CSC525/knowledge_base_manual.txt")

    # def crawl_website(self, urls, max_depth=1, max_pages=100):
    #     """Crawl websites from the provided URLs"""
    #     urls_to_visit = [(url, 0) for url in urls]  # (url, depth)
    #     visited = set()
        
    #     all_docs = []
    #     page_count = 0
        
    #     while urls_to_visit and page_count < max_pages:
    #         url, depth = urls_to_visit.pop(0)
            
    #         if url in visited or depth > max_depth:
    #             continue
                
    #         visited.add(url)
    #         print(f"Crawling {url} (depth {depth})")
            
    #         try:
    #             loader = WebBaseLoader(url)
    #             url_docs = loader.load()
    #             all_docs.extend(url_docs)
    #             self.urls_crawled.add(url)
    #             page_count += 1
                
    #             # Rate limiting to be respectful
    #             time.sleep(1)
                
    #             if depth < max_depth:
    #                 # Find links to follow
    #                 response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    #                 soup = BeautifulSoup(response.text, "html.parser")
    #                 links = soup.find_all("a", href=True)
                    
    #                 base_url = "/".join(url.split("/")[:3])  # e.g., https://www.amazon.com
                    
    #                 for link in links:
    #                     href = link["href"]
                        
    #                     # Convert relative URLs to absolute
    #                     if href.startswith("/"):
    #                         href = base_url + href
                        
    #                     # Only follow links from the same domain
    #                     if href.startswith(base_url) and href not in visited:
    #                         urls_to_visit.append((href, depth + 1))
                
    #         except Exception as e:
    #             print(f"Error crawling {url}: {e}")
        
    #     return all_docs
    
    def ingest_manual_file(self, manual_path):
        """Ingest the manual RAG file as a single document"""
        print(f"Ingesting manual knowledge base from {manual_path}...")
        with open(manual_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Wrap as a LangChain Document
        from langchain.docstore.document import Document
        doc = Document(page_content=content, metadata={"source": manual_path})
        return [doc]


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
        #documents = self.crawl_website(urls)
        documents = self.ingest_manual_file("c:/Users/chaid/Jin/CSC525/knowledge_base_manual.txt")

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
    
    def build_knowledge_base_from_manual(self, manual_path):
        """Build the knowledge base from the manual file"""
        print("Starting to build knowledge base from manual file...")
        documents = self.ingest_manual_file(manual_path)
        
        # Process documents
        print(f"Processing {len(documents)} documents...")
        chunks = self.process_documents(documents)
        
        # Build vector store
        if not chunks:
            print("No chunks were created from the manual file.")
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
            self.db = FAISS.load_local(
                self.index_path, 
                self.embedding_model,
                allow_dangerous_deserialization=True  # Add this parameter
            )
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
            print("Knowledge base is not initialized.")
            return []
        
        docs = self.db.similarity_search(query, k=k)
        if not docs:
            # print(f"No relevant context found for '{query}'. Falling back to top {k} chunks.")
            # Fallback: return top k chunks for generic queries
            docs = self.db.similarity_search("", k=k)
        return [doc.page_content for doc in docs]

    def save_rag_context(self, query, k=5, out_path="rag_context.txt"):
        """Save the top-k retrieved context for a query to a text file."""
        docs = self.db.similarity_search(query, k=k) if self.db else []
        with open(out_path, "w", encoding="utf-8") as f:
            for i, doc in enumerate(docs, 1):
                src = doc.metadata.get("source", "unknown")
                f.write(f"[{i}] Source: {src}\n{doc.page_content}\n\n")
        print(f"Saved RAG context for query '{query}' to {out_path}")


class CustomerSupportChatbot:
    def __init__(self, model_path='c:/Users/chaid/Jin/trained-customer-support-bot', # 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                 knowledge_base_path="./knowledge_base"):  
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

    def extract_answer_from_context(self, question, context_docs):
        """Extract answer directly from context if model fails"""
        from transformers import pipeline
        
        # Use a dedicated QA model for extraction
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        
        combined_context = " ".join(context_docs[:3])  # Use top 3 chunks
        result = qa_pipeline(question=question, context=combined_context)
        
        if result['score'] > 0.3:  # Confidence threshold
            return f"According to Amazon's policy: {result['answer']}"
        return None

    def validate_response_uses_context(self, response, context_docs):
        """Check if response contains keywords from context"""
        if not re.search(r"\[\d+\]", response):
            return False
        
        context_keywords = set()
        for doc in context_docs[:2]:
            # Extract key terms (simple version)
            words = re.findall(r'\b[a-z]{4,}\b', doc.lower())
            context_keywords.update(words[:20])  # Top 20 words per doc
        
        response_words = set(re.findall(r'\b[a-z]{4,}\b', response.lower()))
        overlap = len(context_keywords & response_words)
        
        return overlap >= 3  # Require at least 3 matching keywords

    def generate_response(self, customer_message, max_length=250, temperature=0.8, max_retries=2):
        """Generate a response, with a retry mechanism to avoid canned DM requests."""
        # Append the latest customer message to history
        self.history.append(f"Customer: {customer_message}")
        
        # Build the prompt from the last N turns (e.g., 3 exchanges)
        history_prompt = "\n".join(self.history[-6:])  # 3 customer/support pairs
        
        # Get relevant context from knowledge base
        # For policy-related queries, retrieve more context
        k_context = 5 if "policy" in customer_message.lower() or "return" in customer_message.lower() else 3
        context_docs = self.knowledge_base.search_documents(customer_message, k=k_context)
        context_section = "No relevant context found."
        if context_docs:
            context_section = "\n\n".join(
                f"[{idx + 1}] {doc}" for idx, doc in enumerate(context_docs)
            )
        
        # Debugging: Print the retrieved context
        # print("Retrieved context for debugging:\n", context_section)
        
        # **IMPROVED: More explicit instruction-following prompt**
        # Check if we have meaningful context
        has_meaningful_context = context_docs and any(len(doc.strip()) > 50 for doc in context_docs)
        
        if has_meaningful_context:
            input_text = (
                "You are an Amazon customer support agent. Follow these rules strictly:\n"
                "1. Answer ONLY using information from the Context below\n"
                "2. Copy relevant phrases directly from the Context\n"
                "3. After each fact, add [X] where X is the context number\n"
                "4. If the Context doesn't contain the answer, say: 'I don't have that specific information in our records.'\n"
                "5. Do NOT use external knowledge or training data\n\n"
                f"Context:\n{context_section}\n\n"
                f"Customer: {customer_message}\n"
                "Support (remember to cite [X] after each fact):"
            )
        else:
            # No meaningful context - allow general knowledge
            input_text = (
                "You are an Amazon customer support agent. The customer asked a question that is not covered in our knowledge base.\n"
                "Provide a helpful answer using your general knowledge, but start your response with:\n"
                "'I don't have specific information about this in our current database, but generally speaking...'\n\n"
                f"Customer: {customer_message}\n"
                "Support:"
            )
        
        # **REMOVED: extractive QA - causes issues with TinyLlama**
        # Try template-based response first for common queries
        if has_meaningful_context:
            template_response = self.get_template_response(customer_message, context_docs)
            if template_response:
                self.history.append(f"Support: {template_response}")
                return template_response
        
        response = "" # Initialize response to handle cases where all retries fail
        for attempt in range(max_retries + 1):
            # **ADJUSTED: Lower temperature for more faithful copying**
            current_temp = max(0.3, temperature - (attempt * 0.1))

            encoded_input = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1024)
            input_ids = encoded_input['input_ids'].to(self.device)
            attention_mask = encoded_input['attention_mask'].to(self.device)
            
            # print(f"Generating response from model (attempt {attempt+1}, temp={current_temp:.2f})...")
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + max_length,
                    temperature=current_temp,
                    do_sample=True,
                    top_p=0.9,  # Lower for more focused sampling
                    top_k=40,
                    repetition_penalty=1.8,  # Higher penalty
                    no_repeat_ngram_size=4,  # Prevent 4-word repetitions
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # **IMPROVED: Better extraction**
            if "Support (remember to cite" in generated_text:
                response = generated_text.split("Support (remember to cite")[-1].split(":")[-1].strip()
            elif "Support:" in generated_text:
                response = generated_text.split("Support:")[-1].strip()
            else:
                response = generated_text[len(input_text):].strip()
        
            # Stop at first newline or when context is repeated
            response = response.split('\n')[0].strip()
            
            # **RELAXED: Check for ANY citation marker (only if we expected context)**
            if has_meaningful_context:
                has_citation = bool(re.search(r'\[\d+\]', response))
                uses_context_words = self.validate_response_uses_context(response, context_docs)
                
                if has_citation or uses_context_words:
                    # print(f"✓ Valid response generated (citations={has_citation}, context_match={uses_context_words})")
                    self.history.append(f"Support: {response}")
                    response = self.clean_response(response)
                    return response
                else:
                    pass
                    # print(f"✗ Attempt {attempt + 1}: No citations or context match, retrying...")
            else:
                # No context expected, just return the response
                self.history.append(f"Support: {response}")
                response = self.clean_response(response)
                return response
    
        # **FALLBACK: If model fails, construct response manually**
        # print("⚠ Model failed all attempts. Using template fallback.")
        if has_meaningful_context:
            fallback = self.construct_fallback_response(customer_message, context_docs)
        else:
            fallback = "I don't have specific information about this in our current database. Could you please provide more details or rephrase your question?"
        self.history.append(f"Support: {fallback}")
        return fallback

    def get_template_response(self, question, context_docs):
        """Generate template-based responses for common queries"""
        question_lower = question.lower()
        
        # Shipping time query
        if any(word in question_lower for word in ["shipping time", "delivery time", "how long"]):
            # Extract shipping times from context
            times = []
            for doc in context_docs:
                matches = re.findall(r'(Standard|Express|Priority|Two-Day|One-Day) Shipping[\s\n]+(\d+[–-]\d+ business days)', doc)
                times.extend(matches)
            
            if times:
                response = "Our shipping times are: "
                for method, duration in times[:3]:  # Top 3 methods
                    response += f"{method} Shipping takes {duration} [1]. "
                return response.strip()
        
        return None

    def construct_fallback_response(self, question, context_docs):
        """Construct a response by extracting key facts from context"""
        if not context_docs:
            return "I don't have specific information about this in our current database. However, I'm here to help - could you please provide more details?"
        
        # Extract first 2 sentences from top context
        top_context = context_docs[0]
        sentences = re.split(r'[.!?]\s+', top_context)
        relevant_text = '. '.join(sentences[:2]).strip()
        
        if len(relevant_text) > 20:
            return f"Based on our records: {relevant_text} [1]"
        else:
            return "I don't have detailed information about that at the moment. Could you please rephrase your question?"

    def clean_response(self, text):
        """More targeted post-processing to remove artifacts."""
        # Remove [USER] token
        text = re.sub(r'\[USER\]', '', text, flags=re.IGNORECASE).strip()
        
        # Remove agent signatures (e.g., "-EmmaW", "-LHJBY") 
        text = re.sub(r'\s*-[A-Z]{2,}\w*\b', '', text)
        
        # Remove obvious gibberish (3+ consecutive capital letters)
        text = re.sub(r'\b([A-Z]{3,}\s+)+[A-Z]{3,}\b', '', text)
        
        # Remove incomplete sentences at the end
        sentences = re.split(r'([.!?])', text)
        if len(sentences) > 2:
            # Reconstruct sentences
            clean_sentences = []
            for i in range(0, len(sentences)-1, 2):
                sentence = (sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')).strip()
                if len(sentence.split()) > 3:  # Keep sentences with 4+ words
                    clean_sentences.append(sentence)
            text = ' '.join(clean_sentences)
        
        # Final whitespace cleanup
        text = ' '.join(text.split()).strip()
        
        return text if text else "I'm sorry, I'm having trouble with that request. Could you please rephrase?"

    def correct_grammar(self, text):
        """Use LanguageTool API to correct grammar - only on final short response."""
        # Skip grammar check if text is too long
        if len(text) > 1000:
            print("Skipping grammar correction - text too long")
            return text
        
        url = "https://api.languagetoolplus.com/v2/check"
        data = {
            "text": text,
            "language": "en-US"
        }
        try:
            response = requests.post(url, data=data, timeout=5)
            response.raise_for_status()
            matches = response.json().get("matches", [])
            for match in reversed(matches):
                replacement = match.get("replacements", [])
                if replacement:
                    start = match["offset"]
                    end = start + match["length"]
                    text = text[:start] + replacement[0]["value"] + text[end:]
            return text
        except requests.exceptions.RequestException as e:
            print(f"Grammar correction skipped: {e}")
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
    # amazon_urls = [
    #     "https://www.amazon.ca/gp/help/customer/display.html?nodeId=GGE5X8EV7VNVTK6R",
    #     "https://www.amazon.com/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7",
    #     "https://www.amazon.ca/gp/help/customer/display.html?nodeId=GSD587LKW72HKU2V",
    #     "https://www.amazon.com/gp/help/customer/display.html",
    #     "https://www.amazon.com/gp/help/customer/display.html?nodeId=200127470"
    # ]
    
    kb = KnowledgeBase()
    #kb.build_knowledge_base(amazon_urls)
    kb.build_knowledge_base_from_manual(manual_path="c:/Users/chaid/Jin/CSC525/knowledge_base_manual.txt")
    # Save all ingested content for quality check
    if kb.db is not None:
        kb.save_rag_context(query="", k=10000, out_path="rag_context.txt")  # <-- Add this line
        with open("all_rag_content.txt", "w", encoding="utf-8") as f:
            for i, doc in enumerate(kb.db.similarity_search("", k=10000), 1):
                src = doc.metadata.get("source", "unknown")
                f.write(f"[{i}] Source: {src}\n{doc.page_content}\n\n")
        print("Saved all RAG content to all_rag_content.txt")
    
    return kb


def main():
    """Main function to run the chatbot"""
    # Check if knowledge base exists, build if needed
    kb_path = "./knowledge_base"
    if not os.path.exists(os.path.join(kb_path, "index.faiss")):
        print("Building knowledge base from Amazon help pages...")
        setup_knowledge_base()
    else:
        print("Knowledge base already exists. Loading...")
    
    chatbot = CustomerSupportChatbot(knowledge_base_path=kb_path)
    
    # Verify the knowledge base is loaded
    if chatbot.knowledge_base.db is None:
        print("ERROR: Knowledge base failed to load!")
        return
    
    print(f"Knowledge base loaded successfully with {chatbot.knowledge_base.db.index.ntotal} vectors")
    chatbot.chat_session()


if __name__ == "__main__":
    main()