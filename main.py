import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import re

class ToxicityDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  
            ngram_range=(1,3),   
            strip_accents='unicode',
            min_df=3             
        )
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=1.0
        )
        
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        # Convert to string if not already
        text = str(text)
        # Convert to lowercase
        text = text.lower()
        # Keep some special characters that might indicate toxicity
        text = re.sub(r'[^\w\s!?*#@$%]', '', text)
        # Replace multiple exclamations/question marks with single ones
        text = re.sub(r'(!)\1+', r'\1', text)
        text = re.sub(r'(\?)\1+', r'\1', text)
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Add space around special characters to treat them as tokens
        text = re.sub(r'([!?*#@$%])', r' \1 ', text)
        return text.strip()
    
    def train(self, X_train, y_train):
        # Preprocess training data
        X_train_processed = [self.preprocess_text(text) for text in X_train]
        
        # Vectorize text
        X_train_vectorized = self.vectorizer.fit_transform(X_train_processed)
        
        # Train model
        self.model.fit(X_train_vectorized, y_train)
        
    def predict(self, texts):
        # Preprocess input texts
        texts_processed = [self.preprocess_text(text) for text in texts]
        
        # Vectorize
        texts_vectorized = self.vectorizer.transform(texts_processed)
        
        # Get predictions and probabilities
        predictions = self.model.predict(texts_vectorized)
        probabilities = self.model.predict_proba(texts_vectorized)
        
        # Get feature importance for explanation
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        explanations = []
        for idx, text_vec in enumerate(texts_vectorized):
            if predictions[idx] == 1:  # If predicted toxic
                # Get coefficients
                coef = self.model.coef_[0]
                # Get non-zero features for this text
                features = text_vec.nonzero()[1]
                # Get importance scores
                scores = coef[features] * text_vec[0, features].toarray()[0]
                # Sort by absolute importance
                important_idx = np.argsort(np.abs(scores))[-5:]  # Top 5 features
                toxic_features = [(feature_names[features[i]], scores[i]) for i in important_idx]
                explanations.append(toxic_features)
            else:
                explanations.append([])
        
        return predictions, probabilities, explanations
    
    def evaluate(self, X_test, y_test):
        # Get predictions
        predictions, probabilities, _ = self.predict(X_test)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'model': self.model}, f)
    
    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as f:
            components = pickle.load(f)
            
        detector = cls()
        detector.vectorizer = components['vectorizer']
        detector.model = components['model']
        return detector


def main():
    df = pd.read_csv('train.csv', nrows=100000)  
    print(f"Loaded {len(df)} comments")
    
    # Convert target to binary (0 if < 0.5, 1 if >= 0.5)
    df['target_binary'] = (df['target'] >= 0.5).astype(int)
    
    print("\nClass distribution:")
    print(df['target_binary'].value_counts(normalize=True))
    
    # Get features and target
    X = df['comment_text'].values
    y = df['target_binary'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    detector = ToxicityDetector()
    print("\nTraining model...")
    detector.train(X_train, y_train)
    
    print("\nEvaluating model...")
    detector.evaluate(X_test, y_test)
    
    detector.save_model('toxicity_detector.pkl')
    print("\nModel saved as 'toxicity_detector.pkl'")
    
    # Test custom texts
    texts = [
        "you're amazing!",
        "i hate you noob",
        "gg wp",
        "uninstall the game idiot",
        "get off the game and make me a sandwich"
    ]
    predictions, probabilities, explanations = detector.predict(texts)
    
    for text, pred, prob, expl in zip(texts, predictions, probabilities, explanations):
        print(f"\nText: {text}")
        print(f"Prediction: {'Toxic' if pred == 1 else 'Non-toxic'}")
        print(f"Confidence: {max(prob):.2f}")
        if expl:  # If toxic, show why
            print("Top contributing factors:")
            for feature, score in expl:
                print(f"- {feature}: {score:.3f}")

if __name__ == "__main__":
    main()
