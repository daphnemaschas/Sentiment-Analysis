import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Blocs d'aide (Attention, FFN, Positional Encoding) ---
class MultiHeadAttention(nn.Module):
    """Implémente le mécanisme Multi-Head Self-Attention"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads # Dimension de la tête (d_k)
        self.num_heads = num_heads
        self.d_model = d_model

        # Couches linéaires pour Q, K, V et la sortie
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. Couches linéaires et redimensionnement pour les têtes
        q = self.q_linear(q).view(q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2) # Shape: Batch Size, Seq Length, d_model et Avant le transpose: Batch Size, Seq Length, num_heads, d_model
        k = self.k_linear(k).view(k.size(0), -1, self.num_heads, self.d_k).transpose(1, 2) # Shape: Batch Size, Seq Length, d_model et Avant le transpose: Batch Size, Seq Length, num_heads, d_model
        v = self.v_linear(v).view(v.size(0), -1, self.num_heads, self.d_k).transpose(1, 2) # Shape: Batch Size, Seq Length, d_model et Avant le transpose: Batch Size, Seq Length, num_heads, d_model
        # C'est équivalent à faire: 
        # Mais reshape peut rendre le code plus court (en supprimant le besoin de .contiguous() après un transpose), 
        # mais elle masque la création potentielle d'une copie de données. 
        # Dans un code critique pour la performance comme le Transformer, 
        # beaucoup de développeurs préfèrent l'approche explicite de .view() et .contiguous() 
        # pour garder le contrôle sur les allocations de mémoire.
        # q = self.q_linear(q).reshape(q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2) 
        # k = self.k_linear(k).reshape(k.size(0), -1, self.num_heads, self.d_k).transpose(1, 2) 
        # v = self.v_linear(v).reshape(v.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Calcul de l'Attention (Scaled Dot-Product Attention)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # Shape: (Batch, num_heads, SeqLen, SeqLen)

        # Application du masque (pour le Décodeur)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        scores = F.softmax(scores, dim=-1) # 

        # 3. Pondération des Valeurs (V)
        output = torch.matmul(scores, v)

        # 4. Concaténation et Couche de Sortie
        output = output.transpose(1, 2).contiguous().view(output.size(0), -1, self.d_model)
        output = self.out(output)

        # Le calcul est en O(Seq Len ** 2): on comprend l'intérêt des window size qu'il faut calculer au training
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)  # batch_size x sentence size x dim_inp
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.Dropout(dropout),
            nn.GELU(), # Bert utilise GELU au lieu de ReLU
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Sous-couche 1: Attention + Residual
        x = x + self.dropout(self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        # Sous-couche 2: FFN + Residual
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

# -- Embeddings --
class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model) # (vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model) # Contrairement au Transformer Vanilla, BERT apprend où se trouvent les mots. (max_len, d_model)
        self.segment_embeddings = nn.Embedding(2, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        
        # Création des indices de position [0, 1, 2...]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Si on ne fournit pas de token_type_ids, on assume que tout est "phrase A" (0)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Somme des embeddings
        # L'Embedding de Token te donne son identité
        # L'Embedding de Position te donne son emplacement
        # L'Embedding de Segment te donne son contexte
        embeddings = self.token_embeddings(input_ids) + \
                     self.position_embeddings(position_ids) + \
                     self.segment_embeddings(token_type_ids) # E_{total} = E_{token} + E_{segment} + E_{position}
        # En les additionnant, on fusionne ces trois informations dans un seul vecteur de taille d_model. 
        # Le modèle ne voit pas trois vecteurs séparés, il voit un seul vecteur "augmenté" 
        # qui contient toutes les caractéristiques en même temps.
        embeddings = self.layer_norm(embeddings) # Normalisation
        return self.dropout(embeddings)
    
# -- Architecture complète du modèle ---
class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=4, num_heads=8, d_ff=512, dropout=0.1):
        super().__init__()
        self.embeddings = BERTEmbeddings(vocab_size, d_model, dropout=dropout)
        # On empile les couches d'encodeur
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, input_ids, token_type_ids=None, mask=None):
        # Embeddings
        x = self.embeddings(input_ids, token_type_ids=token_type_ids)
        
        # Passage dans les couches d'encodeur
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        return x
    
# -- Wrapper for Sequence Classification ---
class BERTForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, num_labels, d_model=256, num_layers=4, num_heads=8):
        super().__init__()
        self.bert = BERT(vocab_size, d_model, num_layers, num_heads)
        self.classifier = nn.Linear(d_model, num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Gestion du masque pour l'attention
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2) # Pour l'adapter aux têtes d'attention
        else:
            mask = None

        # Passage dans BERT
        output = self.bert(input_ids, mask=mask) # (Batch, Seq, d_model)
        
        # Pooling : Moyenne sur la séquence
        if mask is not None:
            mask = attention_mask.unsqueeze(-1)  # (Batch, SeqLen, 1)
            pooled_output = (output * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled_output = output.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Calcul de la Loss
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
            
        return {"loss": loss, "logits": logits}