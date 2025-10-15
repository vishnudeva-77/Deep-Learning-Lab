import numpy as np
import pandas as pd
import json
import functools as fc
import os
from sklearn.metrics import accuracy_score
os.makedirs('output', exist_ok=True)

train = pd.read_csv('data1/train', sep='\t', names=['index', 'word', 'POS'])

word = train['word'].values.tolist()
index = train['index'].values.tolist()
pos = train['POS'].values.tolist()

vocab = {}
for w in word:
    vocab[w] = vocab.get(w, 0) + 1

vocab2 = {}
num_unk = 0
for w in vocab:
    if vocab[w] >= 3:
        vocab2[w] = vocab[w]
    else:
        num_unk += vocab[w]

vocab_sorted = sorted(vocab.items(), key=lambda item: item[1], reverse=True)

with open('output/vocab_frequent.txt', 'w') as vocab_file:
    vocab_file.write('<unk>\t0\t' + str(num_unk) + '\n')
    for i, (w, count) in enumerate(vocab_sorted):
        vocab_file.write(f"{w}\t{i+1}\t{count}\n")

print(f'Total vocabulary size: {len(vocab_sorted)}')
print(f'Total <unk> occurrences: {num_unk}')

vocab_ls = list(vocab2.keys())

with open('output/vocab_frequent.txt', 'w') as output:
    for word in vocab_ls:
        output.write(word + '\n')

word = ['<unk>' if w not in vocab_ls else w for w in word]

ss = {}
sx = {}
for i in range(len(word)-1):
    if index[i] < index[i+1]:
        key_ss = f"{pos[i+1]}|{pos[i]}"
        ss[key_ss] = ss.get(key_ss, 0) + 1

        key_sx = f"{word[i]}|{pos[i]}"
        sx[key_sx] = sx.get(key_sx, 0) + 1

for i in range(len(word)):
    if index[i] == 1:
        key = f"{pos[i]}|<s>"
        ss[key] = ss.get(key, 0) + 1

emission = {}
transition = {}

count_pos = {}
for p in pos:
    count_pos[p] = count_pos.get(p, 0) + 1

count_pos['<s>'] = index.count(1)

for sx_pair in sx:
    tag = sx_pair.split('|')[1]
    emission[sx_pair] = sx[sx_pair] / count_pos[tag]

for ss_pair in ss:
    tag = ss_pair.split('|')[1]
    transition[ss_pair] = ss[ss_pair] / count_pos[tag]

print(f'Transition parameters: {len(transition)}')
print(f'Emission parameters: {len(emission)}')

with open('output/hmm.json', 'w') as output:
    json.dump([emission, transition], output)

pos_distinct = list(count_pos.keys())
with open('output/pos.txt', 'w') as pos_output:
    for p in pos_distinct:
        pos_output.write(p + '\n')

def greedy(sentence):
    pos_sequence = []
    sentence = ['<unk>' if w not in vocab_ls else w for w in sentence]

    max_prob = 0
    best_pos = 'UNK'
    for p in pos_distinct:
        try:
            prob = emission.get(f"{sentence[0]}|{p}", 0) * transition.get(f"{p}|<s>", 0)
            if prob > max_prob:
                max_prob = prob
                best_pos = p
        except:
            continue
    pos_sequence.append(best_pos)

    for i in range(1, len(sentence)):
        max_prob = 0
        best_pos_i = 'UNK'
        for p in pos_distinct:
            try:
                prob = emission.get(f"{sentence[i]}|{p}", 0) * transition.get(f"{p}|{pos_sequence[-1]}", 0)
                if prob > max_prob:
                    max_prob = prob
                    best_pos_i = p
            except:
                continue
        pos_sequence.append(best_pos_i)

    return pos_sequence

dev = pd.read_csv('data1/dev', sep='\t', names=['index', 'word', 'POS'])
index_dev = dev['index'].tolist()
word_dev = dev['word'].tolist()
pos_dev = dev['POS'].tolist()

word_dev2, pos_dev2 = [], []
temp_words, temp_pos = [], []

for i in range(len(dev)):
    temp_words.append(word_dev[i])
    temp_pos.append(pos_dev[i])
    if i == len(dev)-1 or index_dev[i] >= index_dev[i+1]:
        word_dev2.append(temp_words)
        pos_dev2.append(temp_pos)
        temp_words, temp_pos = [], []

pos_pred = [greedy(s) for s in word_dev2]
pos_pred_flat = fc.reduce(lambda a, b: a + b, pos_pred)
pos_dev_flat = fc.reduce(lambda a, b: a + b, pos_dev2)

acc = accuracy_score(pos_dev_flat, pos_pred_flat)
print(f'Greedy decoding accuracy: {acc * 100:.2f}%')

def viterbi(sentence):
    sentence = ['<unk>' if w not in vocab_ls else w for w in sentence]
    seq = {i: {} for i in range(len(sentence))}
    pre_pos = {i: {} for i in range(len(sentence))}

    for p in pos_distinct:
        try:
            seq[0][p] = transition.get(f"{p}|<s>", 0) * emission.get(f"{sentence[0]}|{p}", 0)
            pre_pos[0][p] = '<s>'
        except:
            seq[0][p] = 0

    for i in range(1, len(sentence)):
        for p in seq[i-1]:
            for p_prime in pos_distinct:
                key_trans = f"{p_prime}|{p}"
                key_emit = f"{sentence[i]}|{p_prime}"
                prob = seq[i-1][p] * transition.get(key_trans, 0) * emission.get(key_emit, 0)

                if prob > seq[i].get(p_prime, 0):
                    seq[i][p_prime] = prob
                    pre_pos[i][p_prime] = p

    if not seq[len(sentence)-1]:
        return ['UNK'] * len(sentence)

    last_pos = max(seq[len(sentence)-1], key=seq[len(sentence)-1].get)
    seq_predict = [last_pos]
    for i in range(len(sentence)-1, 0, -1):
        last_pos = pre_pos[i].get(last_pos, 'UNK')
        seq_predict.append(last_pos)
    seq_predict.reverse()
    return seq_predict

pos_pred_viterbi = [viterbi(s) for s in word_dev2]
pos_pred_viterbi_flat = fc.reduce(lambda a, b: a + b, pos_pred_viterbi)
acc_viterbi = accuracy_score(pos_dev_flat, pos_pred_viterbi_flat)
print(f'Viterbi decoding accuracy: {acc_viterbi * 100:.2f}%')
