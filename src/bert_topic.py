import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# --- Auto-encodeur pour la réduction ---
class TopicReducer(nn.Module): # Petit réseau de neuronne pour compresser l'information
    def __init__(self, d_model=256, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim) # Vecteur compressé
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction
    
# --- Clustering ---
def Kmeans(x, num_clusters, iterations=50):
    # Initialisation aléatoire des centroïdes
    n_samples, n_features = x.size()
    indices = torch.randperm(n_samples)[:num_clusters]
    centroids = x[indices]

    for i in range(iterations):
        # Calcul des distances entre points et centroïdes
        distances = torch.cdist(x, centroids) # (n_samples, num_clusters)
        cluster_assignments = torch.argmin(distances, dim=1)

        # Mise à jour des centroïdes (moyenne des points du cluster)
        new_centroids = torch.stack([
            x[cluster_assignments == k].mean(0) if (cluster_assignments == k).any() 
            else centroids[k]
            for k in range(num_clusters)
        ])
        centroids = new_centroids
        
    return cluster_assignments, centroids

# --- Architecture Complète BERTopic ---
class BERTopic(nn.Module):
    def __init__(self, trained_bert_wrapper, tokenizer, n_topics=10, latent_dim=10):
        super().__init__()

        self.bert_engine = trained_bert_wrapper.bert
        self.tokenizer = tokenizer
        self.n_topics = n_topics

        d_model = self.bert_engine.embeddings.token_embeddings.embedding_dim

        self.reducer = TopicReducer(
            d_model=d_model, 
            latent_dim=latent_dim
        )
        
        self.topic_keywords = {}
        self.cluster_ids = None
    
    def get_embeddings(self, texts):
        """Transforme les textes en vecteurs"""
        # Extraction des Embeddings avec le modèle BERT
        self.bert_engine.eval()
        device = next(self.parameters()).device # Pour récupérer le device actuel

        with torch.no_grad():
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        
            # Utilisation de Bert
            mask_4d = inputs['attention_mask'].unsqueeze(1).unsqueeze(2) # Masque 4D pour la MultiHeadAttention
            output = self.bert_engine(inputs['input_ids'], mask=mask_4d) # Sortie brute du Transformer (Batch, Seq, d_model)

            # Pooling: Moyenne pondérée par le masque (pour ignorer les [PAD])
            mask_3d = inputs['attention_mask'].unsqueeze(-1)
            embeddings = (output * mask_3d).sum(1) / mask_3d.sum(1)
            
        return embeddings
    
    def fit(self, texts, epochs=100):
        """Entraîne l'auto-encodeur, calcule les clusters et les mots-clés"""
        device = next(self.parameters()).device
        
        # 1. Embeddings
        embeddings = self.get_embeddings(texts)

        # 2. Entraînement du TopicReducer (Auto-encodeur)
        self.reducer.to(device)
        self.reducer.train()
        optimizer = torch.optim.Adam(self.reducer.parameters(), lr=1e-3)
        
        pbar = tqdm(range(epochs), desc="Training")
        for epoch in range(epochs):
            latent, reconstruction = self.reducer(embeddings)
            loss = F.mse_loss(reconstruction, embeddings)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        # 3. Réduction finale et Clustering
        print("Clustering...")
        self.reducer.eval()
        with torch.no_grad():
            compressed_embeddings, _ = self.reducer(embeddings)
            
        self.cluster_ids, _ = Kmeans(compressed_embeddings, num_clusters=self.n_topics)

        # 4. Extraction des mots-clés (c-TF-IDF)
        df_temp = pd.DataFrame({
            "cleaned_text": texts,
            "topic_id": self.cluster_ids.cpu().numpy()
        })
        self.topic_keywords = self._compute_topic_keywords(df_temp)
        
        return self.cluster_ids

    def _compute_topic_keywords(self, df, top_n=10):
        """Regarde les mots fréquents dans un cluster mais rare dans les autres avec c-TF-IDF"""
        docs_per_topic = df.groupby(['topic_id'], as_index=False).agg({'cleaned_text': ' '.join})
        
        # Calcul des fréquences de mots par cluster
        vectorizer = CountVectorizer(stop_words='english')
        count_matrix = vectorizer.fit_transform(docs_per_topic.cleaned_text)
        words = vectorizer.get_feature_names_out()
        
        # Calcul du c-TF-IDF simplifié
        # On normalise la fréquence par la taille du cluster et la rareté du mot ailleurs
        tf = np.array(count_matrix.toarray())
        word_counts = np.array(count_matrix.sum(axis=0)).flatten()
        idf = np.log(1 + (len(docs_per_topic) / (word_counts + 1)))
        ctfidf = tf * idf
        
        topic_keywords = {}
        for i in range(len(docs_per_topic)):
            topic_id = docs_per_topic.iloc[i]['topic_id']
            top_indices = ctfidf[i].argsort()[-top_n:][::-1]
            topic_keywords[topic_id] = [(words[idx], ctfidf[i][idx]) for idx in top_indices]
            
        return topic_keywords

    def visualize_barchart(self, n_topics=4):
        # On limite l'affichage pour que ce soit lisible
        cols = 2
        rows = (n_topics // cols) + (n_topics % cols > 0)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
        axes = axes.flatten()

        for i in range(n_topics):
            if i in self.topic_keywords:
                words = [x[0] for x in self.topic_keywords[i]]
                scores = [x[1] for x in self.topic_keywords[i]]
                
                sns.barplot(x=scores, y=words, ax=axes[i], palette='viridis')
                axes[i].set_title(f'Topic {i}', fontsize=15)
                axes[i].set_xlabel('Score c-TF-IDF')
            else:
                axes[i].axis('off')

        # On cache les graphiques vides
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
    
    def get_topic_info(self):
        """Retourne un DataFrame avec l'ID du topic, le nombre de documents et les mots-clés"""
        if self.cluster_ids is None:
            print("Erreur: Vous devez d'abord appeler .fit()")
            return None

        # Calcul du décompte par Topic
        topic_counts = pd.Series(self.cluster_ids.cpu().numpy()).value_counts().sort_index()
        
        info_list = []
        for topic_id, count in topic_counts.items():
            # Récupère les 4 premiers mots-clés pour le nom explicite
            name = "_".join([word for word, _ in self.topic_keywords[topic_id][:4]]) 
            info_list.append({
                "Topic": topic_id,
                "Count": count,
                "Name": f"{topic_id}_{name}"
            })
            
        return pd.DataFrame(info_list)
