import json


def ev_match(qed_ev, dev_ev):
    match = False
    if len(dev_ev) == 1:
        same_art = qed_ev[0] == dev_ev[0][2]
        same_sent = qed_ev[1] == dev_ev[0][3]
        if same_art and same_sent:
            match = True
    return match


dev_filename = 'shared_task_dev.jsonl'
qed_filename = 'test_predictions.jsonl'

dev_json = json.load(open(dev_filename))
qed_json = json.load(open(qed_filename))

true_pos = 0
false_pos = 0  # in qed but not dev
false_neg = 0  # in dev but not qed
gold = 0
for claim in range(len(dev_json)):
    dev_label = dev_json[claim]['label']
    qed_label = qed_json[claim]['predicted_label']
    dev_evidence = dev_json[claim]['evidence']
    qed_evidence = qed_json[claim]['predicted_evidence']

    one_ev = False
    for dev_ev in dev_evidence:
        in_qed = False
        for qed_ev in qed_evidence:
            if ev_match(qed_ev, dev_ev):
                in_qed = True
        if in_qed:
            true_pos += 1
            one_ev = True
        else:
            false_neg += 1

    for qed_ev in qed_evidence:
        in_dev = False
        for dev_ev in dev_evidence:
            if ev_match(qed_ev, dev_ev):
                in_dev = True
        if not in_dev:
            false_pos += 1

    if one_ev and dev_label == qed_label:
        gold += 1

fever = float(gold) / len(dev_json)

precision = float(true_pos) / (true_pos + false_pos)
recall = float(true_pos) / (true_pos + false_neg)
f1 = 2 * precision * recall / (precision + recall)

print('Precision: ' + str(precision))
print('Recall: ' + str(recall))
print('F1: ' + str(f1))
print('FEVER: ' + str(fever))



