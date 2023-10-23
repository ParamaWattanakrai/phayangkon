# from experimental import training_data
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from utils import DataLoader, embed
embedded = DataLoader()

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    return "yes"

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def reset_weights(m):
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence, input):
        self.hidden = self.init_hidden()
        embeds = embed("one_hot", input, embedded).unsqueeze(1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(input), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags, input):
        feats = self._get_lstm_features(sentence,input)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence, input):
        lstm_feats = self._get_lstm_features(sentence,input)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 68
HIDDEN_DIM = 48

num_epoch = 112
num_kfold = 5

word_to_ix = {}
results = {}

best_accuracy = 0.0
best_model_state_dict = None
best_fold = 0

# for sentence, tags in training_data:
#     for word in sentence:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
kfold = KFold(n_splits=num_kfold,random_state=True ,shuffle=True)

# print('--------------------------------') #comment to load start
# for fold, (train_index, test_index) in enumerate(kfold.split(training_data)):
#     print(f'FOLD {fold+1}')
#     print('--------------------------------')
#     train_data = [training_data[i] for i in train_index]
#     test_data = [training_data[i] for i in test_index]

#     print("Training Data:", len(train_data))
#     print("Test Data:", len(test_data))

#     model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
#     model.apply(reset_weights)
    
#     optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

#     for epoch in range(num_epoch):
#         total_loss = 0
#         for sentence, tags in train_data:
#             model.zero_grad()
#             sentence_in = prepare_sequence(sentence, word_to_ix)
#             targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
#             loss = model.neg_log_likelihood(sentence_in, targets, sentence)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Fold {fold + 1}, Epoch [{epoch+1}], Loss: {total_loss/1000 :.2f}")
   
#     total_correct = 0
#     total_samples = 0
#     ix_to_tag = {v: k for k, v in tag_to_ix.items()}  
#     for sentence, tags in test_data:
#         precheck_sent = prepare_sequence(sentence, word_to_ix)
#         _, predicted_tags = model(precheck_sent,sentence)
#         predicted_tags = [ix_to_tag[ix] for ix in predicted_tags]
#         total_correct += sum(p == t for p, t in zip(predicted_tags, tags))
#         total_samples += len(tags)

#     accuracy = total_correct / total_samples
#     if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             best_model_state_dict = model.state_dict()
#             best_fold = fold

#     print('Accuracy for fold %d: %d %%' % (fold+1, 100.0 * accuracy))
#     print('--------------------------------')
#     results[fold] = 100.0 * accuracy

# print('Training on entire dataset as the final fold...')
# print('--------------------------------')

# model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
# model.apply(reset_weights)
# print('Best fold :',best_fold ,' Best accuracy: ',best_accuracy )

# optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
# model.load_state_dict(best_model_state_dict)

# for epoch in range(num_epoch):
#     total_loss = 0
#     for sentence, tags in training_data:
#         model.zero_grad()
#         sentence_in = prepare_sequence(sentence, word_to_ix)
#         targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
#         loss = model.neg_log_likelihood(sentence_in, targets, sentence)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Final, Epoch [{epoch+1}], Loss: {total_loss/1000 :.2f}")

# total_correct = 0
# total_samples = 0
# for sentence, tags in training_data:
#     precheck_sent = prepare_sequence(sentence, word_to_ix)
#     _, predicted_tags = model(precheck_sent,sentence)
#     predicted_tags = [ix_to_tag[ix] for ix in predicted_tags]
#     total_correct += sum(p == t for p, t in zip(predicted_tags, tags))
#     total_samples += len(tags)

# accuracy = total_correct / total_samples
# print('Accuracy for the final fold: {:.2f}%'.format(100.0 * accuracy))
# print('--------------------------------')

# print(f'K-FOLD CROSS VALIDATION RESULTS FOR {num_kfold} FOLDS')
# print('--------------------------------')

# sum = 0.0
# for key, value in results.items():
#     print(f'Fold {key}: {value:.2f} %')
#     sum += value
# print(f'Average: {sum/len(results.items()):.2f} %')

# torch.save(model.state_dict(), 'trained_BiLSTM_CR_model.pth') #comment to load end
model = BiLSTM_CRF(65, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load('trained_BiLSTM_CR_48_Hidden.pth'))
model.eval()

def segment_syllables(sentence):
    result = []
    boi = model('yes', sentence)
    boi = boi[1]
    b_indices = [i for i, element in enumerate(boi) if element == 0]

    start = 0
    for index in b_indices:
        result.append(sentence[start:index])
        start = index
    result.append(sentence[start:])
    result.pop(0)
    return result

if __name__ == '__main__':
    while True:
        input_sentence = input("Enter a sentence: ")
        precheck_sent = prepare_sequence(input_sentence, word_to_ix)
        print(model(precheck_sent, input_sentence))