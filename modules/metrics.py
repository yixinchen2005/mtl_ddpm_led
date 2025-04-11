def eval_result(true_labels, pred_result, rel2id, logger, use_name=False):
    correct = 0
    total = len(true_labels)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    neg = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
        if name in rel2id:
            if use_name:
                neg = name
            else:
                neg = rel2id[name]
            break
    for i in range(total):
        if use_name:
            golden = true_labels[i]
        else:
            golden = true_labels[i]

        if golden == pred_result[i]:
            correct += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if pred_result[i] != neg:
            pred_positive += 1
    acc = float(correct) / float(total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0

    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
    logger.info('Evaluation result: {}.'.format(result))
    return result

def comp_score(true_issues, detected_issues):
    """
    Compute the f1_score of the error detection results.
    
    Parameters:
    true_issues: a list of tuples, (seq_idx, label_idx), obtained by comparing given ground-truth labels with
        corrected ground-truth labels.
    detected_issues: a list of tuples, (seq_idx, label_idx), obtained by calling another algorithm.
    
    Returns:
    f1_score: the f1 score.
    """
    true_positive, false_positive, false_negative = 0, 0, 0
    for issue in detected_issues:
        if issue in true_issues:
            true_positive += 1
        else:
            false_positive += 1
    for issue in true_issues:
        if issue not in detected_issues:
            false_negative += 1
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1_score = (2 * true_positive) / (2 * true_positive + false_positive + false_negative + 1e-8)
    return precision, recall, f1_score

def comp_f1_score(targets_new, targets_old, pred_forward_old, pred_reverse_new):
        true_issues, detect_issues_forward, detect_issues_reverse = [], [], []
        for i, (seq_n, seq_o) in enumerate(zip(targets_new, targets_old)):
            for j, (l_n, l_o) in enumerate(zip(seq_n, seq_o)):
                if l_n != l_o:
                    true_issues.append((i, j))
        for i, (seq_p, seq_t) in enumerate(zip(pred_forward_old, targets_old)):
            for j, (l_p, l_t) in enumerate(zip(seq_p, seq_t)):
                if l_p != l_t:
                    detect_issues_forward.append((i, j))
        for i, (seq_p, seq_t) in enumerate(zip(pred_reverse_new, targets_new)):
            for j, (l_p, l_t) in enumerate(zip(seq_p, seq_t)):
                if l_p != l_t:
                    detect_issues_reverse.append((i, j))
        precision_forward, recall_forward, f1_score_forward = comp_score(true_issues, detect_issues_forward)
        precision_reverse, recall_reverse, f1_score_reverse = comp_score(true_issues, detect_issues_reverse)
        return precision_forward, recall_forward, f1_score_forward, precision_reverse, recall_reverse, f1_score_reverse