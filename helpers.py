import json


class Vocabulary:
    def __init__(self):
        self.word2index = dict()
        self.num_words = 0

    def get_only_chars(self, word):
        result = ''.join(c for c in word if c.isalpha())
        return result.strip().lower()

    def add_word(self, word):
        word = self.get_only_chars(word)
        if word not in self.word2index:
            self.num_words += 1
            self.word2index[word] = self.num_words
        return self.word2index[word]

    def __getitem__(self, word):
        word = self.get_only_chars(word)
        return self.word2index[word]

    def __len__(self):
        return len(self.word2index.keys())

    def save(self, filename):
        w_file = open(filename, "w")
        json.dump(self.word2index, w_file)
        w_file.close()

    def load(self, filename):
        a_file = open(filename, "r")
        output = json.loads(a_file.read())
        self.word2index = output
        self.num_words = len(self.word2index.keys())
