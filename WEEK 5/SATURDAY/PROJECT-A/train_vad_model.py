import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import numpy as np
import json
from tqdm import tqdm
import os

# --- Emotion Dataset with VAD labels ---
class EmotionVADDataset(Dataset):
    def __init__(self, texts, valences, arousals, tokenizer, max_length=128):
        self.texts = texts
        self.valences = valences
        self.arousals = arousals
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        valence = self.valences[idx]
        arousal = self.arousals[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'valence': torch.tensor(valence, dtype=torch.float32),
            'arousal': torch.tensor(arousal, dtype=torch.float32)
        }

# --- VAD Loss Function ---
class VADLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        # predictions: [batch_size, 2] (valence, arousal)
        # targets: [batch_size, 2] (valence, arousal)
        return self.mse_loss(predictions, targets)

# --- Training Function ---
def train_vad_model(model, train_dataloader, val_dataloader, device, tokenizer, epochs=5):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = VADLoss()
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps * 0.1, 
        num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in train_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = torch.stack([batch['valence'], batch['arousal']], dim=1).to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs.logits)
            
            loss = criterion(predictions, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_dataloader, desc="Validation")
        
        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = torch.stack([batch['valence'], batch['arousal']], dim=1).to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.sigmoid(outputs.logits)
                
                loss = criterion(predictions, targets)
                val_loss += loss.item()
                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("trained_vad_model", exist_ok=True)
            model.save_pretrained("trained_vad_model")
            tokenizer.save_pretrained("trained_vad_model")
            torch.save(model.state_dict(), "trained_vad_model/vad_model.pt")
            print("💾 Best model saved!")

# --- Generate Synthetic Training Data ---
def generate_synthetic_data():
    """Generate synthetic emotion data with VAD labels for training"""
    
    # Positive emotions (high valence)
    positive_texts = [
        "I'm feeling great today!",
        "This is amazing news!",
        "I'm so happy and excited!",
        "What a wonderful day!",
        "I'm feeling optimistic about the future",
        "This makes me so joyful!",
        "I'm thrilled with the results!",
        "I feel blessed and grateful",
        "I'm on cloud nine!",
        "This is absolutely fantastic!"
    ]
    
    # Negative emotions (low valence)
    negative_texts = [
        "I'm feeling sad and depressed",
        "This is terrible news",
        "I'm so frustrated and angry",
        "I feel hopeless right now",
        "This is really disappointing",
        "I'm feeling anxious and worried",
        "I'm so stressed out",
        "I feel lonely and isolated",
        "This is heartbreaking",
        "I'm feeling overwhelmed"
    ]
    
    # Calm emotions (low arousal)
    calm_texts = [
        "I'm feeling peaceful and calm",
        "I feel relaxed and content",
        "I'm in a meditative state",
        "I feel serene and tranquil",
        "I'm feeling mellow today",
        "I feel at peace with myself",
        "I'm feeling gentle and quiet",
        "I feel balanced and centered",
        "I'm feeling soft and gentle",
        "I feel calm and collected"
    ]
    
    # Excited emotions (high arousal)
    excited_texts = [
        "I'm feeling pumped and energized!",
        "I'm so hyped up right now!",
        "I feel electrified and alive!",
        "I'm bursting with energy!",
        "I feel so motivated and driven!",
        "I'm feeling fierce and powerful!",
        "I feel so passionate about this!",
        "I'm feeling wild and free!",
        "I feel so dynamic and vibrant!",
        "I'm feeling unstoppable!"
    ]
    
    texts = []
    valences = []
    arousals = []
    
    # Add positive emotions (high valence, variable arousal)
    for text in positive_texts:
        texts.append(text)
        valences.append(0.8 + np.random.uniform(0.0, 0.2))  # 0.8-1.0
        arousals.append(np.random.uniform(0.3, 0.9))  # Variable arousal
    
    # Add negative emotions (low valence, variable arousal)
    for text in negative_texts:
        texts.append(text)
        valences.append(np.random.uniform(0.0, 0.3))  # 0.0-0.3
        arousals.append(np.random.uniform(0.3, 0.9))  # Variable arousal
    
    # Add calm emotions (medium valence, low arousal)
    for text in calm_texts:
        texts.append(text)
        valences.append(0.4 + np.random.uniform(0.0, 0.3))  # 0.4-0.7
        arousals.append(np.random.uniform(0.1, 0.4))  # Low arousal
    
    # Add excited emotions (medium-high valence, high arousal)
    for text in excited_texts:
        texts.append(text)
        valences.append(0.6 + np.random.uniform(0.0, 0.3))  # 0.6-0.9
        arousals.append(0.7 + np.random.uniform(0.0, 0.3))  # High arousal
    
    # Shuffle the data
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    valences = [valences[i] for i in indices]
    arousals = [arousals[i] for i in indices]
    
    return texts, valences, arousals

# --- Main Training Script ---
def main():
    print("🚀 Starting VAD Model Training...")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2, 
        problem_type="regression"
    )
    model.to(device)
    
    # Generate synthetic training data
    print("📊 Generating synthetic training data...")
    texts, valences, arousals = generate_synthetic_data()
    
    # Split data (80% train, 20% validation)
    split_idx = int(0.8 * len(texts))
    train_texts = texts[:split_idx]
    train_valences = valences[:split_idx]
    train_arousals = arousals[:split_idx]
    
    val_texts = texts[split_idx:]
    val_valences = valences[split_idx:]
    val_arousals = arousals[split_idx:]
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # Create datasets
    train_dataset = EmotionVADDataset(train_texts, train_valences, train_arousals, tokenizer)
    val_dataset = EmotionVADDataset(val_texts, val_valences, val_arousals, tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Train the model
    print("🎯 Training model...")
    train_vad_model(model, train_dataloader, val_dataloader, device, tokenizer, epochs=5)
    
    print("✅ Training complete! Model saved to 'trained_vad_model/' directory")
    print("🎉 You can now use the trained model in your empath bot!")

if __name__ == "__main__":
    main()
