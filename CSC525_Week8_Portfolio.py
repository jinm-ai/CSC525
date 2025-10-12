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
 

contractions = {
        "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
        "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have",
        "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
        "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
        "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is",
        "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
        "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
        "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it would",
        "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
        "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
        "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
        "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
        "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
        "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
        "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
        "she's": "she is", "should've": "should have", "shouldn't": "should not",
        "shouldn't've": "should not have", "so've": "so have", "so's": "so is", "that'd": "that would",
        "that'd've": "that would have", "that's": "that is", "there'd": "there would",
        "there'd've": "there would have", "there's": "there is", "they'd": "they would",
        "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
        "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
        "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
        "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
        "what'll've": "what will have", "what're": "what are", "what's": "what is",
        "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
        "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
        "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
        "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
        "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
        "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
        "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have",
        "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"
    }

def expand_contractions(text):
    """Expand common contractions to assist with NLP processing"""    
    # Regular expression to obtain contractions from conversation pairs
    contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

def clean_tweet_text(text):
    """Clean and preprocess tweet text"""
    if pd.isna(text):
        return ""
    
    # 1. Decode HTML entities
    text = html.unescape(text)
    
    # 2. Replace emojis with text equivalents
    text = replace_emojis(text)
    
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 4. Remove user mentions
    text = re.sub(r'@\w+', '[USER]', text)
    
    # 5. Expand contractions
    text = expand_contractions(text)
    
    # 6. Remove special characters but keep punctuation and apostrophes for contractions
    text = re.sub(r"[^a-zA-Z0-9\s\.\!\?\,\-\:']", '', text)
    
    # 7. Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()

def replace_emojis(text):
    # Map common emojis to text equivalents
    emoji_dict = {
        "❤️": "[HEART]",
        "🤷": "[SHRUG]",
        # Add more as needed
    }
    for emoji, replacement in emoji_dict.items():
        text = text.replace(emoji, replacement)
    return text

def create_conversation_pairs(df, company_ids):
    """Pair each customer tweet with its direct company reply using in_response_to_tweet_id"""
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
    df['sentiment'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df

def feature_engineering(sample_size=10000):
    """Process and clean data, returning all necessary dataframes"""
    # Load the dataset with a reasonable sample size for training
    print(f"Loading dataset with {sample_size} samples...")
    df = pd.read_csv(r'C:\Users\chaid\Jin\CSC525\twcs.csv')
    #df = pd.read_csv('twcs.csv')
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Dataset shape: {df.shape}")
    print(df.head())
    
    # Apply cleaning
    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_tweet_text)

    # Add sentiment scores
    print("Adding sentiment scores...")
    df = add_sentiment(df)

    # Filter out empty or meaningless messages
    df = df[df['cleaned_text'].str.len() > 10]
    print(f"After cleaning, dataset shape: {df.shape}")

    # Save the cleaned dataset to CSV for review
    print("Saving cleaned dataset to CSV...")
    df.to_csv('cleaned_twitter_data.csv', index=False)

    # Identify company accounts using the 'inbound' column which distinguishes user and company
    if 'inbound' in df.columns:
        company_ids = df[df['inbound'] == False]['author_id'].unique().tolist()
        print(f"Identified {len(company_ids)} company accounts using 'inbound' column")
    else:
        # Fallback to previous method if 'inbound' is missing. Top 20 most active author_ids are assumed to be company accounts.
        company_ids = df['author_id'].value_counts().head(20).index.tolist()
        print(f"Identified {len(company_ids)} company accounts using fallback method")

    # Create conversation pairs
    print("Creating conversation pairs...")
    conversation_df = create_conversation_pairs(df, company_ids)
    print(f"Created {len(conversation_df)} conversation pairs")
    
    # Save conversation pairs for review
    conversation_df.to_csv('cleaned_conversation_pairs.csv', index=False)
    print("Saved conversation pairs to CSV")
    
    return conversation_df, company_ids, df

def strip_user_prefix(text):
    # Remove 'USER' from the start, followed by optional punctuation/space
    return re.sub(r'^(USER[\s\:\-\.\,]*)+', '', str(text), flags=re.IGNORECASE)

def format_for_training(conversations_df, tokenizer):
    """Formats the conversation dataframe for the dataset class."""
    print("Formatting data for training...")
    # Strip 'USER' from the beginning of input and response
    conversations_df['input'] = conversations_df['input'].apply(strip_user_prefix)
    conversations_df['response'] = conversations_df['response'].apply(strip_user_prefix)
    conversations_df['input_text'] = "Customer: " + conversations_df['input'].astype(str) + "\nSupport:"
    conversations_df['target_text'] = conversations_df['response'].astype(str)
    formatted_df = conversations_df[['input_text', 'target_text', 'company']]
    print(f"Created {len(formatted_df)} formatted training examples")
    formatted_df.to_csv('formatted_training_data.csv', index=False)
    return formatted_df

class CustomerSupportDataset(Dataset):
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
    """Compute perplexity from eval loss."""
    loss = eval_pred.loss if hasattr(eval_pred, 'loss') else eval_pred[0]
    perplexity = math.exp(loss) if loss < 100 else float('inf')
    return {"eval_loss": loss, "perplexity": perplexity}

def train_model(train_dataset, val_dataset, model, tokenizer, checkpoint_dir=None):
    """Configure and run the training process"""
    print("Configuring training...")
    
    # Training arguments for CPU  a
    training_args = TrainingArguments(
        output_dir='./customer-support-chatbot',
        num_train_epochs=1,  # Reduce epochs for less resource usage
        per_device_train_batch_size=8,   
        per_device_eval_batch_size=8,
        warmup_steps=0,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_steps=500,
        save_steps=1000,
        save_total_limit=1,
        load_best_model_at_end=False, #True,
        gradient_accumulation_steps=4,
        fp16=False,
        do_eval=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0, # 4,
       # evaluation_strategy="steps",   
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,   
    )

    # Train the model
    print("Starting training...")
    if checkpoint_dir:
        print(f"Resuming training from checkpoint: {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        trainer.train()
        
    # Save the final model
    print("Saving trained model...")
    model.save_pretrained('./trained-customer-support-bot')
    tokenizer.save_pretrained('./trained-customer-support-bot')
    print("Model saved to ./trained-customer-support-bot")

class CustomerSupportChatbot:
    def __init__(self, model_path='./trained-customer-support-bot'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def generate_response(self, customer_message, max_length=150, temperature=0.7):
        """Generate a support response to customer message"""
        
        # Format input
        input_text = f"Customer: {customer_message}\nSupport:"
        
        # Tokenize
        input_ids = self.tokenizer.encode(
            input_text, 
            return_tensors='pt'
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=input_ids.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract response
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = generated_text[len(input_text):].strip()
        
        return response
    
    def chat_session(self):
        """Interactive chat session"""
        print("Customer Support Chatbot - Type 'quit' to exit")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
                
            response = self.generate_response(user_input)
            print(f"Support: {response}")

def sample_top_companies(conversation_df, top_companies, total_samples=80000, random_state=42):
    """
    Filter conversation_df to only top_companies and sample evenly from each.
    Returns a DataFrame with total_samples rows (or as close as possible).
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
    print("=== Starting Customer Service Chatbot Pipeline ===")
    
    conversation_file = 'cleaned_conversation_pairs.csv'
     
    
    # 1. Feature Engineering or Load Existing Data
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
     # "AppleSupport", "Uber_Support", "Delta","SpotifyCares","Tesco", "AmericanAir", "comcastcares"
    # Filter and sample from top companies
    conversation_df = sample_top_companies(
        conversation_df,
        top_company,
        total_samples=conversation_sample_size
    )
    
    # Stop here if we just want to review the data
    if not run_training:
        print("Data preparation complete. Review the CSV files before training.")
        return
    
    # 2. Prepare for training with a consistent model choice
    print("Preparing for model training...")
    model_name ='EleutherAI/gpt-neo-125m' # "distilgpt2" #'microsoft/DialoGPT-small' #"TinyLlama/TinyLlama-1.1B-Chat-v0.2"
    
    # Check for checkpoint
    checkpoint_dir = None
    if resume_training:
        # Look for the latest checkpoint in the output directory
        checkpoint_dirs = [d for d in os.listdir('./customer-support-chatbot') 
                          if d.startswith('checkpoint-') and os.path.isdir(os.path.join('./customer-support-chatbot', d))]
        if checkpoint_dirs:
            # Get the latest checkpoint by number
            latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))[-1]
            checkpoint_dir = os.path.join('./customer-support-chatbot', latest_checkpoint)
            print(f"Found checkpoint: {checkpoint_dir}")
            
            # Load tokenizer and model from checkpoint
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
            model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
            print(f"Loaded model and tokenizer from checkpoint: {checkpoint_dir}")
        else:
            print("No checkpoint found, starting from scratch.")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        # Start from scratch
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Ensure the pad token is set
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Format data for training
    training_data = format_for_training(conversation_df, tokenizer)
    
    # 4. Split into train/validation
    train_df, val_df = train_test_split(training_data, test_size=0.2, random_state=42)
    print(f"Split data into {len(train_df)} training and {len(val_df)} validation examples")
    
    # 5. Load model and prepare datasets, uncomment if this is the first run
    #model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = CustomerSupportDataset(train_df, tokenizer)
    val_dataset = CustomerSupportDataset(val_df, tokenizer)
    
    # 6. Train the model
    train_model(train_dataset, val_dataset, model, tokenizer, checkpoint_dir)
    
    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    # Set run_training=True to run the training pipeline
    # Set resume_training=True to resume from checkpoint
    main(run_training=True, sample_size=160000, resume_training=False, conversation_sample_size=160000)
    # resume_training=False to start fresh
    # Uncomment to run the chatbot after training
    # chatbot = CustomerSupportChatbot()
    # chatbot.chat_session()