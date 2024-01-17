import numpy as np

def get_accuracy(acc_preds, acc_labels,flag2='not_print'):
    preds_np = acc_preds.detach().cpu().numpy()
    preds = np.array([ i.argmax() for i in preds_np])
    labels = np.array(acc_labels.cpu())
    correct_count = (preds== labels).sum().item()
    total_count = len(preds)

    accuracy = correct_count / total_count
    if flag2=='print':
        print(f"Accuracy: {accuracy:.3f}")
    return accuracy


def get_precision_recall(preds, labels,flag='true',flag2='not_print'):
    preds_np = preds.detach().cpu().numpy()
    preds = np.array([ i.argmax() for i in preds_np])
    labels = np.array(labels)
    # print(preds)
    # print(labels)

    true_positives = ((preds == 1) & (labels == 1)).sum().item()
    false_positives = ((preds == 1) & (labels == 0)).sum().item()
    false_negatives = ((preds == 0) & (labels == 1)).sum().item()
    true_negatives = ((preds == 0) & (labels == 0)).sum().item()

    if flag=='true':
        if flag2=='print':
            print('---positive is plausible')
        precision = true_positives / (true_positives + false_positives) 
        recall = true_positives / (true_positives + false_negatives) 
    elif flag=='false':
        if flag2=='print':
            print('---positive is implausible')
        precision = true_negatives  / (true_negatives  + false_negatives ) 
        recall = true_negatives  / (true_negatives  +false_positives ) 

    f_score = (2 * precision * recall) / (precision + recall)
    if flag2=='print':
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F-score: {f_score:.3f}")
    
    return precision, recall, f_score
