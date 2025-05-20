import torch
import torch.nn as nn
import torch.nn.functional as F


class CVCLoss(nn.Module):
    """
    Computes graph-level and node-level contrastive loss for two views of a batch of graphs.

    Args:
        temperature (float): Temperature scaling factor.
        use_cosine_similarity (bool): If True, use cosine similarity; else use dot product.
    """
    def __init__(self, temperature: float = 0.25, use_cosine_similarity: bool = True):
        super().__init__()
        self.temperature = temperature
        self.use_cosine = use_cosine_similarity
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _similarity(self, x, y):
        if self.use_cosine:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        return torch.matmul(x, y.transpose(-1, -2))

    def _contrastive_loss(self, z1, z2):
        """
        Contrastive loss between graph representations of two views.
        z1, z2: [B, D]
        """
        batch_size = z1.size(0)
        representations = torch.cat([z1, z2], dim=0)  # [2B, D]
        similarity_matrix = self._similarity(representations, representations)  # [2B, 2B]

        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels, labels], dim=0)

        positive_indices = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~torch.eye(2 * batch_size, device=z1.device, dtype=torch.bool))
        positives = similarity_matrix[positive_indices].view(2 * batch_size, 1)

        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        negatives = similarity_matrix[mask].view(2 * batch_size, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature

        targets = torch.zeros(2 * batch_size, dtype=torch.long, device=z1.device)
        loss = self.criterion(logits, targets)
        return loss / (2 * batch_size)

    def _node_contrastive_single(self, z1, z2):
        """
        Node-level contrastive loss in one direction.
        z1, z2: [B, N, D]
        """
        B, N, D = z1.shape
        z1_flat = z1.view(B * N, D)
        z2_flat = z2.view(B * N, D)

        similarity_matrix = self._similarity(z1_flat, z2_flat)  # [BN, BN]
        positives = torch.diag(similarity_matrix).unsqueeze(1)
        negatives = similarity_matrix[~torch.eye(B * N, dtype=torch.bool, device=z1.device)].view(B * N, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature
        targets = torch.zeros(B * N, dtype=torch.long, device=z1.device)
        loss = self.criterion(logits, targets)
        return loss

    def _node_contrastive_loss(self, z1, z2):
        """
        Bidirectional node-level contrastive loss.
        z1, z2: [B, N, D]
        """
        B, N, _ = z1.shape
        loss_forward = self._node_contrastive_single(z1, z2)
        loss_backward = self._node_contrastive_single(z2, z1)
        return (loss_forward + loss_backward) / (2 * B * N)

    def forward(self, view1, view2):
        """
        view1, view2: [B, N, D] feature matrices from two views of the graph
        """
        # Node-level contrastive loss
        loss_node = self._node_contrastive_loss(view1, view2)

        # Graph-level contrastive loss
        graph1 = view1.mean(dim=1)
        graph2 = view2.mean(dim=1)
        loss_graph = self._contrastive_loss(graph1, graph2)

        return loss_graph, loss_node
