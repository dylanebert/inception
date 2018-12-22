true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

with open('model/gmc/classifier_entailment', 'r') as f:
    for line in f:
        w1, w2, v1, v2 = line.rstrip().split('\t')
        true_positives += int(v1)
        false_negatives += int(v2)

with open('model/gmc/classifier_entailment_false', 'r') as f:
    for line in f:
        w1, w2, v1, v2 = line.rstrip().split('\t')
        false_positives += int(v1)
        true_negatives += int(v1)

recall = float(true_positives) / (true_positives + false_negatives)
precision = float(true_positives) / (true_positives + false_positives)
print(recall)
print(precision)
