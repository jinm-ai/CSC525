"""Fine-tuned Amazon Customer Support Chatbot with RAG knowledge base.
Part two of the CSC 525 Week 8 Portfolio project.
This chatbot uses a fine-tuned language model and a RAG knowledge base built from Amazon help pages.
This implementation includes RAG: response generation, context validation and fallback mechanisms.
Run this script to start an interactive chatbot session in the console only after you have finished the training script CSC525_Week8_Portfolio RAFT.py.
steps to run:
Step0: Run this script after training the model with CSC525_Week8_Portfolio RAFT.py.
Step1: Install libraries and package
Step2: The knowledge base is present in the current directory as 'knowledge_base' folder.
Step3: Ensure the fine-tuned model is present at the 'trained-customer-support-bot-RAFT' directory.
Step4: Run this script to start the chatbot.
Step5: Interact with the chatbot in the console.
Step6: Type 'quit' to exit the chat session. Close the console window.
"""

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
import textstat
from langdetect import detect, LangDetectException
from wordfreq import zipf_frequency
import ftfy


class KnowledgeBase:
    def __init__(self, embedding_model="sentence-transformers/all-mpnet-base-v2", index_path="./knowledge_base"):
        """Initialize the knowledge base with an embedding model and storage location"""
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.index_path = index_path
        self.urls_crawled = set()
        self.db = None

        # Create the knowledge base directory 
        os.makedirs(index_path, exist_ok=True)
        
        # Try to load an existing index, fallback to building from manual file
        if os.path.exists(os.path.join(index_path, "index.faiss")):
            self.load_knowledge_base()
        else:
            print("No existing knowledge base found. Building from manual file...")
            self.build_knowledge_base("c:/Users/chaid/Jin/CSC525/knowledge_base_manual.txt")
    
    def ingest_RAG_file(self, manual_path):
        """Ingest the manual RAG file as a single document/LangChain Document instance"""
        print(f"Ingesting manual knowledge base from {manual_path}...")
        with open(manual_path, "r", encoding="utf-8") as f:
            content = f.read()
        # ingest as a LangChain Document instance which is standard for LangChain to store text
        from langchain.docstore.document import Document
        doc = Document(page_content=content, metadata={"source": manual_path})
        return [doc] #return a list

    def process_documents(self, documents):
        """Process and split documents into chunks, for semantic search and retrieval"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks #returns a list of chunks
    
    def save_knowledge_base(self):
        """Save the knowledge base to disk"""
        if self.db is not None:
            self.db.save_local(self.index_path) #FAISS class method to save the index
            print(f"Knowledge base saved to {self.index_path}")
            
            # deprecated code from web crawling 
            # with open(os.path.join(self.index_path, "crawled_urls.txt"), "w") as f:
            #     for url in self.urls_crawled:
            #         f.write(f"{url}\n")   

    def build_knowledge_base(self, manual_path):
        """The RAG function ingesting the knowledge base.
        Create vector embeddings and store in FAISS vector store. 
        Build the knowledge base from the manual file"""
        print("Starting to build knowledge base.")
        documents = self.ingest_RAG_file(manual_path)
        
        # Process documents split into chunks
        print(f"Processing {len(documents)} documents...")
        chunks = self.process_documents(documents)
        
        # Check if chunks were created
        if not chunks:
            print("No chunks were created from the manual file.")
            return False
        # Build vector store. This took a while on my machine.
        print(f"Creating vector store with {len(chunks)} chunks...")
        # FAISS.from_documents creates a vector db from a list of LangChain Document instances
        self.db = FAISS.from_documents(chunks, self.embedding_model)
        
        # Save the knowledge base
        self.save_knowledge_base()
        return True
    
    def load_knowledge_base(self):
        """Load the FAISS vector store index from disk using FAISS.load_local. """
        try:
            self.db = FAISS.load_local(
                self.index_path, 
                self.embedding_model, #"sentence-transformers/all-mpnet-base-v2"
                allow_dangerous_deserialization=True  # True if source of the index file is trusted
            )
            print(f"Knowledge base loaded from {self.index_path}")
            
            # # deprecated code from web crawling
            # crawled_urls_path = os.path.join(self.index_path, "crawled_urls.txt")
            # if os.path.exists(crawled_urls_path):
            #     with open(crawled_urls_path, "r") as f:
            #         self.urls_crawled = set(line.strip() for line in f)
            
            return True
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return False
    
    # def get_retriever(self, k=3):
    #     """Get a retriever for the knowledge base. for advanced LangChain's chains"""
    #     if self.db is None:
    #         print("Knowledge base not initialized. Please build or load a knowledge base first.")
    #         return None
    # #Converts the FAISS vector store (self.db) into a retriever interface.    
    #     return self.db.as_retriever(search_kwargs={"k": k}) 
 
    
    def search_documents(self, query, k=3):
        """Search for relevant documents based on a query.
        similarity search is based on semantic search.
        We used keyword search in RAFT and semantic search in RAG, to give more variety in responses, allowing model to generalize."""
        if self.db is None:
            print("Knowledge base is not initialized.")
            return []
        #similarity_search is a method of FAISS class vector store
        docs = self.db.similarity_search(query, k=k)
        if not docs:
            # Fallback: return top k chunks for generic queries
            docs = self.db.similarity_search("", k=k)
        return [doc.page_content for doc in docs]


class CustomerSupportChatbot:
    def __init__(self, model_path='c:/Users/chaid/Jin/trained-customer-support-bot-RAFT', # 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                 knowledge_base_path="./knowledge_base"):  
        """Initialize the chatbot with the trained model and tokenizer"""
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path) #loads tokenizer from the model
        
        # Set pad_token to eos_token for GPT-like models  
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(model_path)#Hugging Face class to load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}") 
        self.model.to(self.device)
        print("Model loaded successfully!")
        # Use a regex pattern to detect DM requests. I noticed lots of DM in the first version. This will remove the bad patterns learned from the training set.
        self.dm_pattern = re.compile(r'\b(dm|direct message)\b', re.IGNORECASE)

        self.history = []  # Store conversation history
        
        # Initialize a class instance of the knowledgebase and assigns it to self.knowledge_base attribute
        self.knowledge_base = KnowledgeBase(index_path=knowledge_base_path)

    def extract_answer_from_context(self, question, context_docs):
        """Extract QA answer directly from context
        extractive QA finds most relevant span of text from context without generating new sentences.
        This is useful for policy questions. 
        If generative model cannot product a good answer, we can fall back to extractive QA.
        This is to avoid hallucinations. The main retrieval is search_documents method.
        """
        from transformers import pipeline
        
        # Use a dedicated QA model for extraction
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
        # this model is trained to extract Q&A from given context. pipeline utility function creates a QA model and handles inference process.
        combined_context = " ".join(context_docs[:3])  # Use top 3 chunks
        result = qa_pipeline(question=question, context=combined_context)
        
        if result['score'] > 0.3:  # Confidence threshold
            return f"According to Amazon's policy: {result['answer']}"
        return None

    def validate_response_uses_context(self, response, context_docs):
        """Check if response aligns semantically with retrieved context"""
        if not re.search(r"\[\d+\]", response) or not context_docs: #retrieved content has [1], [2]
            return False

        embedder = getattr(self.knowledge_base, "embedding_model", None)
    #check if the embedding model is available for semantic similarity calculations
        """Condition 1: Semantic similarity check using embeddings."""
        if embedder:
            try:
                selected_docs = [doc[:750] for doc in context_docs[:3]] #limit character but may not be optimal for all cases.well below 512 tokens.
                doc_vectors = torch.tensor(embedder.embed_documents(selected_docs), dtype=torch.float32)
                response_vector = torch.tensor(embedder.embed_query(response), dtype=torch.float32)

                similarities = torch.nn.functional.cosine_similarity(
                    doc_vectors, response_vector.unsqueeze(0), dim=1
                ) #measuring semantic similarity between two vectors, cosine_similarity is the best for semantic similarity

                if torch.max(similarities).item() >= 0.6:
                    return True
            except Exception as exc:
                print(f"Semantic validation skipped: {exc}")
        """Condition 2: If there is no semantic similarity, it falls back to keyword overlap check. 
        collecting response words and context words, and checking for at least 3 overlapping keywords."""
        context_keywords = set() #matches overlap of keywords between context and response
        for doc in context_docs[:2]:
            words = re.findall(r'\b[a-z]{4,}\b', doc.lower())
            context_keywords.update(words[:20])

        response_words = set(re.findall(r'\b[a-z]{4,}\b', response.lower()))
        overlap = len(context_keywords & response_words)

        return overlap >= 3

    def generate_response(self, customer_message, max_length=250, temperature=0.8, max_retries=2):
        """Generate a response, with a retry mechanism and fallback logic."""
        # Append the latest customer message to history
        self.history.append(f"Customer: {customer_message}")
        
        # Build the prompt from the last N turns (e.g., 3 exchanges)
        history_prompt = "\n".join(self.history[-6:])  # 3 customer/support pairs
        
        # Get relevant context from knowledge base
        # For policy-related queries, retrieve more context
        k_context = 3
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
        embedder = getattr(self.knowledge_base, "embedding_model", None)
        if context_docs and embedder:
            try:
                doc_vectors = torch.tensor(embedder.embed_documents([doc[:750] for doc in context_docs]), dtype=torch.float32)
                query_vector = torch.tensor(embedder.embed_query(customer_message), dtype=torch.float32)
                similarities = torch.nn.functional.cosine_similarity(doc_vectors, query_vector.unsqueeze(0), dim=1)
                has_meaningful_context = torch.max(similarities).item() > 0.5
            except Exception as exc:
                #print(f"Context relevance check failed: {exc}")
                has_meaningful_context = any(len(doc.strip()) > 50 for doc in context_docs)
        else:
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
                f"{history_prompt}\n"
                f"Customer: {customer_message}\n"
                "Support (remember to cite [X] after each fact):"
            )#add conversation history to input text allows memory of past exchanges
        else:
            # No meaningful context - allow general knowledge
            input_text = (
                "You are an Amazon customer support agent. The customer asked a question that is not covered in our knowledge base.\n"
                "Provide a helpful answer using your general knowledge, but start your response with:\n"
                "'I don't have specific information about this in our current database, but generally speaking...'\n\n"
                f"{history_prompt}\n"
                f"Customer: {customer_message}\n"
                "Support:"
            ) #add conversation history to input text allows memory of past exchanges
        
        response = "" # Initialize response to handle cases where all retries fail
        for attempt in range(max_retries + 1):
            # **Lower temperature for more faithful copying**
            current_temp = max(0.3, temperature - (attempt * 0.1))
            # **Increase temperature to ensure a response is generated?**
            #current_temp = min(1.5, temperature + (attempt * 0.1))
            encoded_input = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1024)
            input_ids = encoded_input['input_ids'].to(self.device)
            attention_mask = encoded_input['attention_mask'].to(self.device)
            
            #loop for retries  
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
            
            # **remove citation markers**
            if "Support (remember to cite" in generated_text:
                response = generated_text.split("Support (remember to cite")[-1].split(":")[-1].strip()
            elif "Support:" in generated_text:
                response = generated_text.split("Support:")[-1].strip()
            else:
                response = generated_text[len(input_text):].strip()
        
            # Stop at first newline or when context is repeated
            # response = response.split('\n')[0].strip()
            
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
        ## implemented RAFT in training so QA knowledgebase is removed as it resembles training data 
        # # **FALLBACK 1: If model fails, try extractive QA for a precise answer**
        # if has_meaningful_context:
        #     print("⚠ Model failed all attempts. Trying extractive QA fallback...")
        #     extractive_answer = self.extract_answer_from_context(customer_message, context_docs)
        #     if extractive_answer:
        #         self.history.append(f"Support: {extractive_answer}")
        #         return extractive_answer

        # **FALLBACK 2: If all else fails, including general response, construct response manually**
        #print("⚠ Extractive QA failed. Using template fallback.")
        if has_meaningful_context:
            fallback = self.construct_fallback_response(customer_message, context_docs)
        else:
            fallback = "I don't have specific information about this in my knowledge base. Could you please provide more details or rephrase your question?"
        self.history.append(f"Support: {fallback}")
        return fallback

    def _select_best_context_snippet(self, question, context_docs):
        """ Semantic search (using embeddings) is more accurate, but it can fail if:
        keyword overlap is simpler and less precise, but it ensures the chatbot can still select a somewhat relevant context snippet, so the user always gets an answer"""
        question_terms = set(re.findall(r'\b[a-z]{3,}\b', question.lower()))
        best_index = 0
        best_score = 0
        for idx, doc in enumerate(context_docs):
            doc_terms = set(re.findall(r'\b[a-z]{3,}\b', doc.lower()))
            score = len(question_terms & doc_terms)
            if score > best_score:
                best_score = score
                best_index = idx
        return best_index, context_docs[best_index]

    def construct_fallback_response(self, question, context_docs):
        """Construct a response by extracting key facts from context"""
        if not context_docs: #semantic matching will general context_docs. if not there, then semantic matching failed. then try fallback keyword matching
            return "I don't have specific information about this in my knowledge base. However, I'm here to help - could you please provide more details?"
        #fallback to keyword overlap to select best context snippet
        idx, snippet = self._select_best_context_snippet(question, context_docs)
        #**format the snippet to extract key points**
        # Prioritize splitting by newlines to preserve list formatting
        lines = [line.strip() for line in snippet.split('\n') if line.strip()]
        
        merged_lines = []
        i = 0
        while i < len(lines):
            current = lines[i]
            if re.match(r'^(\d+[\).]?|[-*•])$', current) and i + 1 < len(lines):
                merged_lines.append(f"{current} {lines[i + 1]}")
                i += 2
                continue
            merged_lines.append(current)
            i += 1
        #ensure the chatbot’s fallback answer is informative and cites the source context, even when the initial extraction is too short.
        relevant_lines = []
        total_chars = 0
        for line in merged_lines:
            relevant_lines.append(line)
            total_chars += len(line)
            if len(relevant_lines) >= 6 or total_chars >= 400:
                break
        
        relevant_text = '\n'.join(relevant_lines).strip()
        if len(relevant_text) <= 20 and len(merged_lines) > len(relevant_lines):
            relevant_text = '\n'.join(merged_lines[:min(len(merged_lines), 8)]).strip()
        
        if len(relevant_text) > 20:
            return f"Based on our records: {relevant_text} [{idx+1}]"
        else:
            return "I don't have detailed information about that at the moment. Could you please rephrase your question?"

    def clean_response(self, text):
        """Post processing: Normalize artifacts and clean gibberish."""
        text = ftfy.fix_text(text or "").strip() #normalize text encoding issues
        text = re.sub(r'\[USER\]', '', text, flags=re.IGNORECASE) #this is from training data, a predominant pattern
        text = re.sub(r'\s*-[A-Z]{2,}\w*\b', '', text)
        text = re.sub(r'(.)\1{6,}', r'\1', text)
        text = ' '.join(text.split())

        if not text or len(text.split()) < 3: #too short
            return "I'm sorry, I'm having trouble with that request. Could you please rephrase?"

        words = re.findall(r"[A-Za-z']{3,}", text.lower()) #words with at least 3 letters
        if not words:
            return "I'm sorry, my response was unclear. Could you please rephrase your question?"

        scored = [w for w in words if zipf_frequency(w, "en") >= 2.5] #enough common english words.
        if len(scored) / len(words) < 0.5:
            return "I'm sorry, my response was unclear. Could you please rephrase your question?"

        return text

    def correct_grammar(self, text):
        """Use LanguageTool API to correct grammar on final response. Upholds response quality."""
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
    """Set up the knowledge base with a manual RAG file"""    
    kb = KnowledgeBase() 
    kb.build_knowledge_base(manual_path="c:/Users/chaid/Jin/CSC525/knowledge_base_manual.txt")
     
    if kb.db is not None:
        kb.save_rag_context(query="", k=10000, out_path="rag_context.txt")  
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