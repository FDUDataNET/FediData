import torch
import torch.nn as nn
import torch.nn.functional as F

class BECEModel(nn.Module):
    def __init__(self, image_dim, tweet_dim, num_prop_dim, category_dim, hidden_dim, num_classes, num_edge_types=3):
        super(BECEModel, self).__init__()
        # Image encoding
        self.image_linear = nn.Linear(image_dim, hidden_dim)
        # Tweet encoding
        self.tweet_linear = nn.Linear(tweet_dim, hidden_dim)
        # Numerical property encoding
        self.num_prop_linear = nn.Linear(num_prop_dim, hidden_dim)
        # Categorical property encoding
        self.category_linear = nn.Linear(category_dim, hidden_dim)
        
        # Feature fusion after concatenation
        self.fusion_linear = nn.Linear(4 * hidden_dim, hidden_dim)
        
        # Edge type embedding
        self.edge_type_embed = nn.Embedding(num_embeddings=num_edge_types, embedding_dim=hidden_dim*2)
        
        # Gaussian parameters (mean and log variance)
        self.mu_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.logvar_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Edge confidence prediction MLP
        self.edge_conf_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Final node classification
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode_user(self, image_tensor, tweets_tensor, num_prop, category_prop):
        image_feat = F.relu(self.image_linear(image_tensor))
        tweet_feat = F.relu(self.tweet_linear(tweets_tensor))
        num_feat = F.relu(self.num_prop_linear(num_prop))
        category_feat = F.relu(self.category_linear(category_prop))
        combined = torch.cat([image_feat, tweet_feat, num_feat, category_feat], dim=-1)
        user_repr = F.relu(self.fusion_linear(combined))
        return user_repr

    def construct_edge(self, u_i, u_j, edge_type=None):
        edge_feat = torch.cat([u_i, u_j], dim=-1)
        # edge_feat = torch.abs(u_i - u_j)

        if edge_type is not None:
            edge_type_emb = self.edge_type_embed(edge_type)
            edge_feat = edge_feat + edge_type_emb
        return F.relu(edge_feat)

    def parameterized_gaussian(self, edge_emb):
        mu = self.mu_layer(edge_emb)
        log_var = self.logvar_layer(edge_emb)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, log_var

    def compute_edge_confidence(self, edge_emb):
        logits = self.edge_conf_mlp(edge_emb).squeeze(-1)
        p = torch.sigmoid(logits)
        return p

    def forward(self, image_tensor, tweets_tensor, num_prop, category_prop,
                edge_index, edge_type, labels=None, train_idx=None, val_idx=None, test_idx=None):
        """
        Use Bernoulli sampling to select edges
        """
        # Encode node features
        node_features = self.encode_user(image_tensor, tweets_tensor, num_prop, category_prop)  # [num_nodes, hidden_dim]
        
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        u_i = node_features[src_idx]
        u_j = node_features[dst_idx]
        edge_emb = self.construct_edge(u_i, u_j, edge_type)  # [num_edges, hidden_dim*2]
        
        # Compute Gaussian parameters (optional)
        z, mu, log_var = self.parameterized_gaussian(edge_emb)
        
        # Predict edge confidence probability
        p_ij = self.compute_edge_confidence(edge_emb)  # [num_edges]
        
        # Bernoulli sampling
        bernoulli_dist = torch.distributions.Bernoulli(probs=p_ij)
        s_ij = bernoulli_dist.sample()  # [num_edges], 0 or 1
        
        # Only keep edges sampled as 1
        trusted_edges_mask = s_ij.bool()
        trusted_src_idx = src_idx[trusted_edges_mask]
        trusted_dst_idx = dst_idx[trusted_edges_mask]
        trusted_edge_type = edge_type[trusted_edges_mask] if edge_type is not None else None
        trusted_p = p_ij[trusted_edges_mask]
        
        # Neighbor aggregation (example: mean of neighbor features)
        num_nodes = node_features.size(0)
        aggregated_features = torch.zeros_like(node_features)
        for i in range(num_nodes):
            neighbor_mask = (trusted_src_idx == i)
            neighbors = trusted_dst_idx[neighbor_mask]
            if len(neighbors) > 0:
                neighbor_feats = node_features[neighbors]
                aggregated_features[i] = neighbor_feats.mean(dim=0)
            else:
                aggregated_features[i] = node_features[i]
        
        # Final classification
        logits = self.classifier(aggregated_features)  # [num_nodes, num_classes]
        return logits