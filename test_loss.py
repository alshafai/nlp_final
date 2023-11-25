import torch
from torch.nn import functional as F

def forward(logits, teacher_probs, labels):
        logits = logits.float()  # In case we were in fp16 mode
        loss = F.cross_entropy(logits, labels, reduction='none')
        print(f"loss {loss}")
        print(torch.eye(logits.size(1))[0])
        one_hot_labels = torch.eye(logits.size(1))[labels]
        print(f"one_hot_labels {one_hot_labels}")

        weights = 1 - (one_hot_labels * teacher_probs).sum(1)
        print(f"weights {weights}")
        
        # weights = weights ** theta
        
        return (weights * loss).sum() / weights.sum()

if __name__ == "__main__":
        lo = torch.tensor([[-1,5]])
        print(lo.shape)
        te = torch.tensor([[.9,.1]])
        # te = F.softmax(te)
        labels = torch.tensor([0])
        print(f"logits {lo} \n teacher {te}")
        print(forward(lo,te,labels))