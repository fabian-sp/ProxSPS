import torch 

# Metrics
def get_metric_function(metric):
    
    if metric == "softmax_loss":
        return softmax_loss
        
    elif metric == "logistic_loss":
        return logistic_loss
    
    elif metric == "squared_loss":
        return squared_loss

    elif metric == "logistic_accuracy":
        return logistic_accuracy

    elif metric == "softmax_accuracy":
        return softmax_accuracy
    
    elif metric == 'rmse':
        return rmse

    else:
        raise NotImplementedError("Unknown metric function.")


# out = model output, computed beforehand
# all metrics by default use mean (over batch)    


def softmax_loss(out, labels, backwards=False):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(out, labels.long().view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def logistic_loss(out, labels, backwards=False):
    criterion = torch.nn.SoftMarginLoss()
    loss = criterion(out.view(-1), labels.float().view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss


def squared_loss(out, labels, backwards=False):
    criterion = torch.nn.MSELoss()
    loss = criterion(out.view(-1), labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss



##
# Accuracy
# ==========================

def logistic_accuracy(out, labels):
    logits = torch.sigmoid(out).view(-1)
    pred_labels = ((logits>=0.5)*2-1).view(-1) # map to{-1,1}
    acc = (pred_labels == labels).float().mean()

    return acc

def softmax_accuracy(out, labels):
    pred_labels = out.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc

def rmse(out, labels):
    criterion = torch.nn.MSELoss()
    loss = criterion(out.view(-1), labels.view(-1))
    return torch.sqrt(loss)
