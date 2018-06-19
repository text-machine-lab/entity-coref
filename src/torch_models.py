import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import time
import os
import sys
import numpy as np
import pickle

from src.build_data import build_dataFrame, DataGen, slice_data, EMBEDDING_DIM
from src.evaluator import TriadEvaluator


class CorefTagger(nn.Module):
    def __init__(self, vocab_size, pos_size, word_embeddings=None):
        super(CorefTagger, self).__init__()
        self.vocab_size = vocab_size
        self.pos_size = pos_size

        self.WordEmbedding = nn.Embedding(self.vocab_size + 1, EMBEDDING_DIM)
        if word_embeddings is not None:
            self.WordEmbedding.weight = nn.Parameter(torch.from_numpy(word_embeddings).type(torch.cuda.FloatTensor))
        # print("word embedding size:", self.WordEmbedding.weight.size())
        self.WordLSTM = nn.LSTM(EMBEDDING_DIM, 128, num_layers=1, batch_first=True, bidirectional=True)

        self.PosEmbedding = nn.Embedding(self.pos_size + 1, self.pos_size + 1)
        self.PosEmbedding.weight = nn.Parameter(torch.eye(self.pos_size + 1).type(torch.cuda.FloatTensor))
        self.PosLSTM = nn.LSTM(self.pos_size + 1, 8, num_layers=1, batch_first=True, bidirectional=True)

        self.PairHidden_1 = weight_norm(nn.Linear(2*(256+16)+1+1, 256), name='weight')
        self.PairHidden_2 = weight_norm(nn.Linear(256, 128), name='weight')
        self.Context = nn.Linear(128*2, 128)
        self.Decoder = nn.Linear(256, 64)
        self.Harmonize = nn.Linear(64*3, 16)
        self.Out = nn.Linear(16, 3)

        self.label_constraint = torch.nn.Sequential(
            torch.nn.Linear(3,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,8),
            torch.nn.ReLU(),
            torch.nn.Linear(8,1),
            torch.nn.Sigmoid()).cuda()

        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        self.init_label_constraint()

    def init_label_constraint(self):
        print("Training label constraint model...")
        X = autograd.Variable(torch.cuda.FloatTensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                                              [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]))
        y = autograd.Variable(torch.cuda.FloatTensor([[0], [0], [0], [1], [0], [1], [1], [0]]))

        loss_fn = nn.BCELoss()
        optimizer = optim.SGD(self.label_constraint.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        while True:
            for e in range(2000):
                pred = self.label_constraint(X)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if loss.data.item() < 1e-2: break
            else:
                print(pred)
                print("Keep training label constraint...")
                sys.stdout.flush()

        print("Finished training lable constraint:")
        print(pred)

        for param in self.label_constraint.parameters():  # freeze the model
            param.requires_grad = False

    def process_words(self, input_words):
        word_emb_0 = self.WordEmbedding(input_words[0])
        word_emb_0 = F.dropout(word_emb_0, p=0.3)
        word_lstm_0, _ = self.WordLSTM(word_emb_0)  # (batch, seq, feature)
        word_lstm_0, _ = torch.max(word_lstm_0, dim=1, keepdim=False)  # (batch, feature)

        word_emb_1 = self.WordEmbedding(input_words[1])
        word_emb_1 = F.dropout(word_emb_1, p=0.3)
        word_lstm_1, _ = self.WordLSTM(word_emb_1)  # (batch, seq, feature)
        word_lstm_1, _ = torch.max(word_lstm_1, dim=1, keepdim=False)  # (batch, feature)

        word_emb_2 = self.WordEmbedding(input_words[2])
        word_emb_2 = F.dropout(word_emb_2, p=0.3)
        word_lstm_2, _ = self.WordLSTM(word_emb_2)  # (batch, seq, feature)
        word_lstm_2, _ = torch.max(word_lstm_2, dim=1, keepdim=False)  # (batch, feature)

        return word_lstm_0, word_lstm_1, word_lstm_2

    def process_pos_tags(self, input_pos_tags):
        pos_emb_0 = self.PosEmbedding(input_pos_tags[0])
        pos_lstm_0, _ = self.PosLSTM(pos_emb_0)  # (batch, seq, feature)
        pos_lstm_0, _ = torch.max(pos_lstm_0, dim=1, keepdim=False)

        pos_emb_1 = self.PosEmbedding(input_pos_tags[1])
        pos_lstm_1, _ = self.PosLSTM(pos_emb_1)  # (batch, seq, feature)
        pos_lstm_1, _ = torch.max(pos_lstm_1, dim=1, keepdim=False)

        pos_emb_2 = self.PosEmbedding(input_pos_tags[2])
        pos_lstm_2, _ = self.PosLSTM(pos_emb_2)  # (batch, seq, feature)
        pos_lstm_2, _ = torch.max(pos_lstm_2, dim=1, keepdim=False)

        return pos_lstm_0, pos_lstm_1, pos_lstm_2

    def forward(self, X):
        input_distances = [X[i].type(torch.cuda.FloatTensor) for i in range(3)]
        input_speakers = [X[i].type(torch.cuda.FloatTensor) for i in range(3, 6)]
        input_words = [X[i] for i in range(6, 9)]
        input_pos_tags = [X[i] for i in range(9, 12)]

        word_lstms = self.process_words(input_words)
        pos_lstms = self.process_pos_tags(input_pos_tags)

        concat01 = torch.cat(
            [input_distances[0], input_speakers[0], word_lstms[0], pos_lstms[0], word_lstms[1], pos_lstms[1]], -1)
        hidden01_1 = F.relu(F.dropout(self.PairHidden_1(concat01), p=0.3))
        hidden01_2 = F.relu(F.dropout(self.PairHidden_2(hidden01_1), p=0.3))

        concat12 = torch.cat(
            [input_distances[1], input_speakers[1], word_lstms[1], pos_lstms[1], word_lstms[2], pos_lstms[2]], -1)
        hidden12_1 = F.relu(F.dropout(self.PairHidden_1(concat12), p=0.3))
        hidden12_2 = F.relu(F.dropout(self.PairHidden_2(hidden12_1), p=0.3))

        concat20 = torch.cat(
            [input_distances[2], input_speakers[2], word_lstms[0], pos_lstms[0], word_lstms[2], pos_lstms[2]], -1)
        hidden20_1 = F.relu(F.dropout(self.PairHidden_1(concat20), p=0.3))
        hidden20_2 = F.relu(F.dropout(self.PairHidden_2(hidden20_1), p=0.3))

        hidden_shared = torch.cat((hidden01_2 + hidden12_2 + hidden20_2, hidden01_2 * hidden12_2 * hidden20_2), -1)
        context = F.relu(self.Context(hidden_shared))

        decoder0 = F.tanh(self.Decoder(torch.cat([hidden01_2, context], -1)))
        decoder1 = F.tanh(self.Decoder(torch.cat([hidden12_2, context], -1)))
        decoder2 = F.tanh(self.Decoder(torch.cat([hidden20_2, context], -1)))
        harmonized = F.tanh(self.Harmonize(torch.cat([decoder0, decoder1, decoder2], -1)))
        output = F.sigmoid(self.Out(harmonized))

        return output

    @staticmethod
    def sharpen(x, alpha=5.0):
        return F.softmax(x**alpha)

    def criterion(self, pred, truth):
        individual_loss = nn.BCELoss()(pred, truth)

        transitivity_loss = 5 * self.label_constraint(CorefTagger.sharpen(pred)).sum() / len(pred)

        return individual_loss, transitivity_loss

    def fit(self, X, y):
        pred = self.forward(X)
        individual_loss, transitivity_loss = self.criterion(pred, y)
        loss = individual_loss + transitivity_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        acc = (pred.round() == y).sum().type(torch.cuda.FloatTensor) / (len(y)*3)

        return individual_loss.data.item(), transitivity_loss.data.item(), acc

    def evaluate(self, X, y):
        with torch.no_grad():
            pred = self.forward(X)
            individual_loss, transitivity_loss = self.criterion(pred, y)
            acc = (pred.round() == y).sum().type(torch.cuda.FloatTensor) / (len(y) * 3)

        return individual_loss.data.item(), transitivity_loss.data.item(), acc

    def predict(self, X_np):
        """Takes numpy array as input"""

        with torch.no_grad():
            X = [autograd.Variable(torch.from_numpy(x).type(torch.cuda.LongTensor)) for x in X_np]
            pred = self.forward(X)

        return pred.data.cpu().numpy()

class CorefTaggerReview(CorefTagger):
    def __init__(self, coref_tagger):
        super(CorefTagger, self).__init__()
        self.coref_tagger = coref_tagger
        # for param in self.coref_tagger.parameters():  # freeze the model
        #     param.requires_grad = False

        decoder_size = self.coref_tagger.Decoder.out_features
        self.Review = nn.RNNCell(decoder_size * 3, 64, nonlinearity='tanh')
        self.h0 = nn.Parameter(torch.zeros(64).type(torch.cuda.FloatTensor))
        self.Harmonize = nn.Linear(64, 3)

        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=0.005, momentum=0.9, weight_decay=1e-4)
        self.label_constraint = self.coref_tagger.label_constraint

    def decoder_forward(self, X):
        input_distances = [X[i].type(torch.cuda.FloatTensor) for i in range(3)]
        input_speakers = [X[i].type(torch.cuda.FloatTensor) for i in range(3, 6)]
        input_words = [X[i] for i in range(6, 9)]
        input_pos_tags = [X[i] for i in range(9, 12)]

        word_lstms = self.coref_tagger.process_words(input_words)
        pos_lstms = self.coref_tagger.process_pos_tags(input_pos_tags)

        concat01 = torch.cat(
            [input_distances[0], input_speakers[0], word_lstms[0], pos_lstms[0], word_lstms[1], pos_lstms[1]], -1)
        hidden01_1 = F.relu(F.dropout(self.coref_tagger.PairHidden_1(concat01), p=0.3))
        hidden01_2 = F.relu(F.dropout(self.coref_tagger.PairHidden_2(hidden01_1), p=0.3))

        concat12 = torch.cat(
            [input_distances[1], input_speakers[1], word_lstms[1], pos_lstms[1], word_lstms[2], pos_lstms[2]], -1)
        hidden12_1 = F.relu(F.dropout(self.coref_tagger.PairHidden_1(concat12), p=0.3))
        hidden12_2 = F.relu(F.dropout(self.coref_tagger.PairHidden_2(hidden12_1), p=0.3))

        concat20 = torch.cat(
            [input_distances[2], input_speakers[2], word_lstms[0], pos_lstms[0], word_lstms[2], pos_lstms[2]], -1)
        hidden20_1 = F.relu(F.dropout(self.coref_tagger.PairHidden_1(concat20), p=0.3))
        hidden20_2 = F.relu(F.dropout(self.coref_tagger.PairHidden_2(hidden20_1), p=0.3))

        hidden_shared = torch.cat((hidden01_2 + hidden12_2 + hidden20_2, hidden01_2 * hidden12_2 * hidden20_2), -1)
        context = F.relu(self.coref_tagger.Context(hidden_shared))

        decoder0 = F.tanh(self.coref_tagger.Decoder(torch.cat([hidden01_2, context], -1)))
        decoder1 = F.tanh(self.coref_tagger.Decoder(torch.cat([hidden12_2, context], -1)))
        decoder2 = F.tanh(self.coref_tagger.Decoder(torch.cat([hidden20_2, context], -1)))
        harmonized = F.tanh(self.coref_tagger.Harmonize(torch.cat([decoder0, decoder1, decoder2], -1)))
        output = F.sigmoid(self.coref_tagger.Out(harmonized))
        return output, torch.cat([decoder0, decoder1, decoder2], -1)

    # def forward(self, X, steps=2):
    #     single_out = self.coref_tagger.forward(X)  # (batch, 3)
    #     h = single_out
    #     for t in range(steps):
    #         h = F.sigmoid(self.Review(single_out, h))
    #         # print("reviewed results", t, h)
    #
    #     return h

    def forward(self, X, steps=3):
        _, decoder_out = self.decoder_forward(X)
        batch_size = X[0].size()[0]
        h = self.h0.expand(batch_size, 64)
        for t in range(steps):
            h = self.Review(decoder_out, h)
            # print("reviewed results", t, h)

        out = F.sigmoid(self.Harmonize(h))

        return out

    def fit(self, X, y):
        pred = self.forward(X)
        individual_loss, transitivity_loss = self.criterion(pred, y)
        loss = individual_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        acc = (pred.round() == y).sum().type(torch.cuda.FloatTensor) / (len(y) * 3)

        return individual_loss.data.item(), transitivity_loss.data.item(), acc


def train(**kwargs):
    train_gen = kwargs['train_gen']
    val_dir = kwargs['val_dir']
    model_destination = kwargs['model_destination']
    epochs = kwargs['epochs']
    load_model = kwargs['load_model']
    review = kwargs.get('review', False)

    group_size = 100
    train_input_gen = train_gen.generate_triad_input(file_batch=50, threads=3)

    assert torch.cuda.is_available()
    if review:
        if load_model:
            model = torch.load(os.path.join(model_destination, 'review', 'model.pt'))
        else:
            model = CorefTaggerReview(torch.load(os.path.join(model_destination, 'model.pt')))
        model_destination = os.path.join(model_destination, 'review/')
        if not os.path.exists(model_destination):
            os.makedirs(model_destination)
    elif load_model:
        model = torch.load(os.path.join(model_destination, 'model.pt'))
    else:
        model = CorefTagger(len(train_gen.word_indexes), len(train_gen.pos_tags), word_embeddings=train_gen.embedding_matrix)
    model = model.cuda()
    print("Model loaded successfully.")
    training_history = []
    evaluator = None

    if val_dir is not None:
        # Need the same word indexes and pos indexes for training and test data
        val_gen = DataGen(build_dataFrame(val_dir, threads=1), train_gen.word_indexes, train_gen.pos_tags)
        val_input_gen = val_gen.generate_triad_input(file_batch=10, looping=True, threads=2)  # file_batch is the # files to use
        print("val_input_gen created.")
        # just get data from 1 for try
        val_data_q = next(val_input_gen)
        print("val_data_q created.")
        val_data = val_data_q[0]
        # val_X, val_y = next(group_data(val_data, group_size, batch_size=None))
        val_X, val_y = val_data
        val_X = [autograd.Variable(torch.from_numpy(x).type(torch.cuda.LongTensor)) for x in val_X]
        val_y = autograd.Variable(torch.from_numpy(val_y).type(torch.cuda.FloatTensor))
        print("val data created.")

    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=1e-4)
    for epoch in range(epochs):
        sys.stdout.write('\n')
        # train_data_q = subproc_queue.get()
        train_data_q = next(train_input_gen)
        n_training_files = len(train_data_q)
        # epoch_history = []
        start = time.time()
        model.train()
        history = {'acc': [], 'loss': [], 'trans_loss': [], 'val_acc': [], 'val_loss': [], 'val_trans_loss': []}
        for n, data in enumerate(train_data_q):
            for X, y in slice_data(data, group_size):  # create batches
                if not y.any(): continue

                X = [autograd.Variable(torch.from_numpy(x).type(torch.cuda.LongTensor)) for x in X]
                y = autograd.Variable(torch.from_numpy(y).type(torch.cuda.FloatTensor))

                loss, trans_loss, acc = model.fit(X, y)
                val_loss, val_trans_loss, val_acc = model.evaluate(val_X, val_y)

                history['loss'].append(loss)
                history['trans_loss'].append(trans_loss)
                history['acc'].append(acc)
                history['val_loss'].append(val_loss)
                history['val_trans_loss'].append(val_trans_loss)
                history['val_acc'].append(val_acc)

            acc = np.mean(history['acc'])
            loss = np.mean(history['loss'])
            trans_loss = np.mean(history['trans_loss'])
            val_acc = np.mean(history['val_acc'])
            val_loss = np.mean(history['val_loss'])
            val_trans_loss = np.mean(history['val_trans_loss'])

            sys.stdout.write(
                "epoch %d after training file %d/%d--- -%ds - loss : %.4f - %.4f - acc : %.4f - val_loss : %.4f - %.4f - val_acc : %.4f\r" % (
                    epoch + 1, n + 1, n_training_files, int(time.time() - start), loss, trans_loss, acc, val_loss, val_trans_loss, val_acc))
            sys.stdout.flush()

        training_history.append({'categorical_accuracy': acc, 'loss': loss})
        if (epoch +1) % 10 == 0:
            torch.save(model, os.path.join(model_destination, 'model.pt'))

            if val_dir:
                if evaluator is None:
                    model.eval()
                    evaluator = TriadEvaluator(model, val_input_gen)
                    # evaluator.data_available = True
                    # filler = multiprocessing.Process(target=evaluator.fill_q_store, args=())
                    # filler.daemon = True
                    # filler.start()
                else:
                    evaluator.model = model
                    evaluator.model.eval()

                eval_results = evaluator.fast_eval()
                # print("\nlabel constraint factor:", model.c)
                print(eval_results)

            if epoch + 1 == 200:
                for g in model.optimizer.param_groups:
                    g['lr'] = 0.005
            if epoch + 1 == 400:
                for g in model.optimizer.param_groups:
                    g['lr'] = 0.001

    eval_results = evaluator.fast_eval()
    print(eval_results)
    # with open(os.path.join(args.model_destination, 'results.pkl'), 'w') as f:
    #     pickle.dump(eval_results, f)
    with open(os.path.join(model_destination, 'history.pkl'), 'wb') as f:
        pickle.dump(training_history, f)
    print("Done!")

