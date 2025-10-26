"""Amazon Customer Support Model Fine-Tuning with RAFT 
Part one of the CSC 525 Week 8 Portfolio project.
steps to run:
Step1: Install libraries and packages
Step2: download training dataset from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter and save twcs.csv in the current directory.
Step3: Copy policies and guides from amazon help pages. Create knowledge_base_manual.txt file in the current directory.
step4: Run the fine-tuning program using the amazon dataset and knowledge base. (RAFT: Retrieval Augmented Fine Tuning)
step5: The program could run for a few days on CPU. If interrupted, resume training from the last checkpoint by setting resume_training=True in main().
step6: The trained model will be saved in the 'trained-customer-support-bot-RAFT' directory.
step7: Now use the chatbot script to implement the RAG chatbot using the trained model.
"""
import pandas as pd
import numpy as np
import re
import torch
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import html 
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from langdetect import detect, LangDetectException
import contractions
from sentence_transformers import SentenceTransformer, util

def expand_contractions(text):
    """Expand common contractions to assist with NLP processing. 
    Originally I used a manually defined dictonary which took space here. 
    Now using the 'contractions' package for comprehensive coverage."""    
    # Regular expression to obtain contractions from conversation pairs
    # contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
    # def replace(match):
    #     return contractions[match.group(0)]
    # return contractions_re.sub(replace, text)    
    return contractions.fix(text)
def clean_tweet_text(text):
    """Clean and preprocess tweet text"""
    if pd.isna(text):
        return ""
    
    # 1. Decode HTML entities
    text = html.unescape(text)
    
    # 2. Replace emojis with text  
    text = replace_emojis(text)
    
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 4. Remove user mentions
    text = re.sub(r'@\w+', '[USER]', text)
    
    # 5. Expand contractions
    text = expand_contractions(text)
    
    # Remove special characters but keep punctuation and apostrophes for contractions
    text = re.sub(r"[^a-zA-Z0-9\s\.\!\?\,\-\:']", '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def replace_emojis(text):
    # Map emojis to text equivalents to make sense to the tokenizer
    emoji_dict = {
        "❤️": "[HEART]",
        "🤷": "[SHRUG]",
        "😊": "[SMILE]",
    }
    for emoji, replacement in emoji_dict.items():
        text = text.replace(emoji, replacement)
    return text

def create_conversation_pairs(df, company_ids):
    """Pair each customer tweet with agent reply using in_response_to_tweet_id    
    this function is called in feature_engineering()"""
    conversations = []
    # Build a mapping from tweet_id to cleaned_text and author_id
    tweet_map = df.set_index('tweet_id')
    for idx, row in df.iterrows():
        # Only consider customer tweets (inbound == True)
        if row['inbound']:
            # Find company reply: tweet where in_response_to_tweet_id == this tweet's tweet_id and inbound == False
            replies = df[(df['in_response_to_tweet_id'] == row['tweet_id']) & (df['inbound'] == False)]
            for _, reply in replies.iterrows():
                conversations.append({
                    'input': row['cleaned_text'],
                    'response': reply['cleaned_text'],
                    'customer_input': True,
                    'company': reply['author_id']
                })
    return pd.DataFrame(conversations)

def add_sentiment(df):
    """Add sentiment polarity scores to the dataframe using TextBlob
    Called in feature_engineering()"""
    df['sentiment'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df

def is_english(text):
    """Return True if text is detected as English, else False.
    I added this function after initial round of data cleaning. 
    Lots of non-English tweets that will create noise for training."""
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def feature_engineering(sample_size=10000):
    """Process and clean data, returning all necessary dataframes"""
    # Load the dataset with a reasonable sample size for training
    print(f"Loading dataset with {sample_size} samples...")
    df = pd.read_csv(r'C:\Users\chaid\Jin\CSC525\twcs.csv')
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    
    # Apply cleaning function
    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_tweet_text)

    # Filter for English only. Added after initial cleaning to reduce noise.
    print("Filtering for English language only...")
    df = df[df['cleaned_text'].apply(is_english)]
    print(f"After English filtering, dataset shape: {df.shape}")

    # Add sentiment scores for optional analysis
    print("Adding sentiment scores...")
    df = add_sentiment(df)

    # Filter out empty or meaningless messages
    df = df[df['cleaned_text'].str.len() > 10]
    print(f"After cleaning, dataset shape: {df.shape}")

    # Save the cleaned dataset to CSV for review
    print("Saving cleaned dataset to CSV...")
    df.to_csv('cleaned_twitter_data.csv', index=False)

    # Identify company accounts using the 'inbound' column 
    if 'inbound' in df.columns:
        company_ids = df[df['inbound'] == False]['author_id'].unique().tolist()
        print(f"Identified {len(company_ids)} company accounts using 'inbound' column")
    else:
        company_ids = df['author_id'].value_counts().head(20).index.tolist()
        print(f"Identified {len(company_ids)} company accounts using fallback method")

    # Create conversation pairs
    print("Creating conversation pairs...")
    conversation_df = create_conversation_pairs(df, company_ids)
    print(f"Created {len(conversation_df)} conversation pairs")
    
    # Save conversation pairs for reuse
    conversation_df.to_csv('cleaned_conversation_pairs.csv', index=False)
    print("Saved conversation pairs to CSV")
    
    return conversation_df, company_ids, df

def strip_user_prefix(text):
    # Remove 'USER' from the start. Initial model results showed it was causing confusion and model remembered to output 'USER' for every response.
    return re.sub(r'^(USER[\s\:\-\.\,]*)+', '', str(text), flags=re.IGNORECASE)

# def format_for_training(conversations_df, tokenizer):
#     """Formats the conversation dataframe for the dataset class.
#        commented out to add RAFT functionality"""
#     print("Formatting data for training...")
#     
#     conversations_df['input'] = conversations_df['input'].apply(strip_user_prefix)
#     conversations_df['response'] = conversations_df['response'].apply(strip_user_prefix)
#     conversations_df['input_text'] = "Customer: " + conversations_df['input'].astype(str) + "\nSupport:"
#     conversations_df['target_text'] = conversations_df['response'].astype(str)
#     formatted_df = conversations_df[['input_text', 'target_text', 'company']]
#     print(f"Created {len(formatted_df)} formatted training examples")
#     formatted_df.to_csv('formatted_training_data.csv', index=False)
#     return formatted_df

def format_for_training(conversations_df, tokenizer, use_raft=True, manual_path="c:/Users/chaid/Jin/CSC525/knowledge_base_manual.txt"):
    """Formats the conversation dataframe for the dataset class, with RAFT context using semantic search."""
    """Adding RAFT functionality to retrieve context from knowledge base manual.
    Zhang, T., et al. (2024). RAFT: Adapting Language Model to Domain Specific RAG. https://arxiv.org/pdf/2403.10131"""
    print("Formatting data for training with semantic retrieval...")

    if use_raft:
        # 1. Load a pre-trained sentence transformer model
        print("Loading sentence transformer model for semantic search...")
        retriever_model = SentenceTransformer('all-MiniLM-L6-v2')

        # 2. Load and chunk the knowledge base
        with open(manual_path, "r", encoding="utf-8") as f:
            manual = f.read()
        chunks = [p.strip() for p in manual.split('\n\n') if len(p.strip()) > 40]
        print(f"Loaded {len(chunks)} knowledge base chunks.")

        # 3. Encode the knowledge base chunks into embeddings (vectors)
        print("Encoding knowledge base chunks...")
        chunk_embeddings = retriever_model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
    
    conversations_df['input'] = conversations_df['input'].apply(strip_user_prefix)
    conversations_df['response'] = conversations_df['response'].apply(strip_user_prefix)
    
    input_texts = []
    # Use tqdm for progress tracking as this can be slow
    for _, row in tqdm(conversations_df.iterrows(), total=len(conversations_df), desc="Processing conversations"):
        customer_msg = str(row['input'])
        context = ""
        
        if use_raft:
            # 4. Encode the customer message
            question_embedding = retriever_model.encode(customer_msg, convert_to_tensor=True)
            
            # 5. Compute cosine similarity between the question and all chunks
            cos_scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
            
            # 6. Find the best-matching chunk
            best_chunk_idx = torch.argmax(cos_scores).item()
            best_score = cos_scores[best_chunk_idx].item()
            
            # 7. Retrieve context only if the similarity score is above a threshold
            if best_score > 0.5: # Threshold can be tuned
                context = chunks[best_chunk_idx]

        if context:
            prompt = f"Context:\n{context}\n\nCustomer: {customer_msg}\nSupport:"
        else:
            prompt = f"Customer: {customer_msg}\nSupport:"
            
        input_texts.append(prompt)

    conversations_df['input_text'] = input_texts
    conversations_df['target_text'] = conversations_df['response'].astype(str)
    formatted_df = conversations_df[['input_text', 'target_text', 'company']]
    
    print(f"Created {len(formatted_df)} formatted training examples (RAFT={use_raft})")
    formatted_df.to_csv('formatted_training_data_RAFT.csv', index=False)
    return formatted_df

class CustomerSupportDataset(Dataset):
    """Custom PyTorchDataset class training a language model with RAFT context and customer support dataset.
    Tokenizes both input and the output then concatenates them, masking input tokens so loss on computed on target tokens.
    Padding to fixed max_length for the model."""
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Combine input and target for causal LM training
        input_text = row['input_text']
        target_text = row['target_text']
        full_text = f"{input_text} {target_text}{self.tokenizer.eos_token}"

        # Tokenize input and target separately
        input_enc = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors='pt'
        )
        target_enc = self.tokenizer(
            f"{target_text}{self.tokenizer.eos_token}",
            truncation=True,
            max_length=self.max_length - input_enc['input_ids'].shape[1],
            add_special_tokens=False,
            return_tensors='pt'
        )

        # Concatenate input and target
        input_ids = torch.cat([input_enc['input_ids'], target_enc['input_ids']], dim=1).squeeze()
        attention_mask = torch.cat([input_enc['attention_mask'], target_enc['attention_mask']], dim=1).squeeze()

        # Create labels: mask input tokens with -100, keep target tokens as is
        labels = torch.cat([
            torch.full(input_enc['input_ids'].shape, -100),  # mask input
            target_enc['input_ids']
        ], dim=1).squeeze()

        # Pad to max_length if necessary
        pad_len = self.max_length - input_ids.shape[0]
        if pad_len > 0:
            input_ids = torch.cat([input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len)])
            labels = torch.cat([labels, torch.full((pad_len,), -100)])
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]

        return {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.long(),
            'labels': labels.long()
        }

def compute_metrics(eval_pred):
    """Compute perplexity from loss."""
    loss = eval_pred.loss if hasattr(eval_pred, 'loss') else eval_pred[0]
    perplexity = math.exp(loss) if loss < 100 else float('inf')
    return {"eval_loss": loss, "perplexity": perplexity}

def train_model(train_dataset, val_dataset, model, tokenizer, checkpoint_dir=None):
    """Configure and run the training process"""
    print("Configuring training...")

    # Training arguments for CPU environment
    training_args = TrainingArguments(
        output_dir='./customer-support-chatbot',
        num_train_epochs=1,  # Reduce epochs for less resource usage
        per_device_train_batch_size=8, #small batch size for CPU  
        per_device_eval_batch_size=8,
        warmup_steps=0, #No warmup for simplicity
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_steps=500, #reasonable intervals to avoid I/O overhead
        save_steps=1000, # about 25% of an epoch
        save_total_limit=1,
        load_best_model_at_end=False, #True,
        gradient_accumulation_steps=4,
        fp16=False,#only for GPU
        do_eval=True,
        dataloader_pin_memory=False,#only for GPU
        dataloader_num_workers=0, # no parallel data loading, avoiding overhead on CPU
       # evaluation_strategy="steps",   
    )

    # utility from huggingface to dynamically pads batches of text and prepares them for language modeling tasks
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    ) #Prepares batches for training.

    # Initialize trainer with parameters.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,   
    )

    # Train the model; if checkpoint_dir is available, resume from there instead of starting over
    #resume_training=False, then will skip this part
    print("Starting training...")
    if checkpoint_dir:
        print(f"Resuming training from checkpoint: {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        trainer.train()
        
    # Save the final model 
    print("Saving trained model...")
    model.save_pretrained('./trained-customer-support-bot-RAFT')
    tokenizer.save_pretrained('./trained-customer-support-bot-RAFT')
    print("Model saved to ./trained-customer-support-bot-RAFT")


def sample_top_companies(conversation_df, top_companies, total_samples=80000, random_state=42):
    """
    Filter conversation_df to only top_companies and sample evenly from each.
    Although only Amazon is used, we keep the function flexible for future use.
    """
    filtered_df = conversation_df[conversation_df['company'].isin(top_companies)]
    print(f"Found {len(filtered_df)} total conversations from the top {len(top_companies)} companies")
    
    # Calculate target samples per company
    samples_per_company = total_samples // len(top_companies)
    
    sampled_dfs = []
    company_counts = {}
    
    # Sample evenly from each company
    for company in top_companies:
        company_df = filtered_df[filtered_df['company'] == company]
        company_counts[company] = len(company_df)
        # If not enough samples, take all available
        n_samples = min(samples_per_company, len(company_df))
        if n_samples > 0:
            sampled_dfs.append(company_df.sample(n=n_samples, random_state=random_state))
            print(f"Sampled {n_samples} from {company} (available: {len(company_df)})")
    
    # Combine all samples
 



    result_df = pd.concat(sampled_dfs) if sampled_dfs else pd.DataFrame()
    # If we have fewer than total_samples, sample additional records to fill the quota
    if len(result_df) < total_samples:
        additional = filtered_df[~filtered_df.index.isin(result_df.index)]
        n_needed = total_samples - len(result_df)
        if len(additional) > 0:
            additional_samples = additional.sample(n=min(n_needed, len(additional)), random_state=random_state)
            result_df = pd.concat([result_df, additional_samples])

            print(f"Added {len(additional_samples)} additional samples to reach target")
    
    # Shuffle final result
    result_df = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"Final dataset: {len(result_df)} conversation pairs from top companies")
    
    return result_df

def main(run_training=False, sample_size=None, resume_training=True, conversation_sample_size=80000):
    #toggle resume_training for checkpoint loading
    print("=== Starting Customer Service Chatbot Pipeline ===")
    #load the two cleaned data files from training program.
    conversation_file = 'cleaned_conversation_pairs.csv'
    formatted_file = 'formatted_training_data_RAFT.csv'
     
    # Feature Engineering if there is no existing Data
    if os.path.exists(conversation_file):
        print(f"Loading existing conversation pairs from {conversation_file}...")
        conversation_df = pd.read_csv(conversation_file)
        print("Conversation pairs loaded.")
    else:
        print("No existing conversation file found. Running feature engineering...")
        conversation_df, _, _ = feature_engineering(sample_size)
        print("Feature engineering complete.")
    # Define top companies so I can select RAG training data from them.
    top_company = ["AmazonHelp",] 
    # originally set multiple companies: "AppleSupport", "Uber_Support", "Delta","SpotifyCares","Tesco", "AmericanAir", "comcastcares"
    # And filter from top companies. After initial experiments, I decided to focus on Amazon for better domain focus. This will match the Amazon knowledge base.
    conversation_df = sample_top_companies(
        conversation_df,
        top_company,
        total_samples=conversation_sample_size
    )
    
    # Stop and review the data. Resume training when ready.
    if not run_training:
        print("Data preparation complete. Review the CSV files before training.")
        return
    
    # 2. Prepare for training with a consistent model choice
    print("Preparing for model training...")
    model_name ='EleutherAI/gpt-neo-125m'
    
    # Check for checkpoint
    checkpoint_dir = None
    if resume_training:
        # Look for the latest checkpoint in the output directory
        checkpoint_dirs = [d for d in os.listdir('./customer-support-chatbot') 
                          if d.startswith('checkpoint-') and os.path.isdir(os.path.join('./customer-support-chatbot', d))]
        if checkpoint_dirs:
            latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))[-1]
            checkpoint_dir = os.path.join('./customer-support-chatbot', latest_checkpoint)
            print(f"Found checkpoint: {checkpoint_dir}")
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            print(f"Loaded model and tokenizer from checkpoint: {checkpoint_dir}")
        else:
            print("No checkpoint found, starting from scratch.")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    tokenizer.pad_token = tokenizer.eos_token
    #This line was commented out to avoid reformatting every time
    #training_data = format_for_training(conversation_df, tokenizer)

    # Format data for training
    #RAFT implementation in format_for_training() function. Retrieving context with keywords appending to agent responses.
    #Note I used semantic matching to retrieve RAG for Chatbot program. using different retrieval methods for training and inference (keyword for RAFT training and semantic for chatbot RAG) can help the model see a wider variety of context styles and may improve generalization to more diverse queries.
    # Comment out formatting if formatted_training_data_RAFT.csv exists
    if os.path.exists(formatted_file):
        print(f"Loading formatted training data from {formatted_file}...")
        training_data = pd.read_csv(formatted_file)
        print("Formatted training data loaded.")
    else:
        print("Formatting data for training...")
        training_data = format_for_training(conversation_df, tokenizer)
        print("Formatted training data created and saved.")

    # 4. Split into train/validation
    train_df, val_df = train_test_split(training_data, test_size=0.2, random_state=42)
    print(f"Split data into {len(train_df)} training and {len(val_df)} validation examples")
    
    # Create datasets ready for training
    train_dataset = CustomerSupportDataset(train_df, tokenizer)
    val_dataset = CustomerSupportDataset(val_df, tokenizer)
    
    # 6. Train the model
    train_model(train_dataset, val_dataset, model, tokenizer, checkpoint_dir)
    
    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    # Set run_training=True to run the training pipeline
    # Set resume_training=True to resume from checkpoint
    # Resume_training=False to start fresh
    main(run_training=True, sample_size=50000, resume_training=False, conversation_sample_size=50000)

