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

def comp_f1_score(targets_new, targets_old, pred_new):
    """
    Compute the F1 score for error detection by comparing true errors (targets_new != targets_old)
    with detected errors (pred_new != targets_old).

    Parameters:
    targets_new: Tensor or array [bsz, seq_len], corrected ground-truth labels.
    targets_old: Tensor or array [bsz, seq_len], noisy ground-truth labels.
    pred_new: Tensor or array [bsz, seq_len], predicted denoised labels.

    Returns:
    precision: Precision of error detection.
    recall: Recall of error detection.
    f1_score: F1 score of error detection.
    """
    true_issues, detect_issues = [], []
    # Identify true errors
    for i, (seq_n, seq_o) in enumerate(zip(targets_new, targets_old)):
        for j, (l_n, l_o) in enumerate(zip(seq_n, seq_o)):
            if l_n != l_o:
                true_issues.append((i, j))
    # Identify detected errors
    for i, (seq_p, seq_o) in enumerate(zip(pred_new, targets_old)):
        for j, (l_p, l_o) in enumerate(zip(seq_p, seq_o)):
            if l_p != l_o:
                detect_issues.append((i, j))
    
    precision, recall, f1_score = comp_score(true_issues, detect_issues)
    return precision, recall, f1_score

def comp_score(true_issues, detected_issues):
    """
    Compute precision, recall, and F1 score for error detection.

    Parameters:
    true_issues: List of tuples (seq_idx, label_idx), true error positions.
    detected_issues: List of tuples (seq_idx, label_idx), detected error positions.

    Returns:
    precision, recall, f1_score: Error detection metrics.
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