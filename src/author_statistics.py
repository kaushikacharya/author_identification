from author_identification import *
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os


class AuthorStatistics:
    def __init__(self, set_of_sentences, train_folder, test_folder):
        self.author_identity = AuthorIdentity(set_of_sentences=set_of_sentences, train_folder=train_folder,
                                              test_folder=test_folder)

    def load_train_test_data(self, train_csv_file, test_csv_file):
        train_folder = self.author_identity.train_folder
        self.author_identity.load_train_data(train_file=os.path.join(train_folder, train_csv_file))

        test_folder = self.author_identity.test_folder
        self.author_identity.load_test_data(test_file=os.path.join(test_folder, test_csv_file))

    def pre_process_data(self):
        self.author_identity.replace_newline_extra_spaces(df=self.author_identity.train_df)
        self.author_identity.replace_newline_extra_spaces(df=self.author_identity.test_df)

    def compute_sentence_count_per_sample_statistics(self):
        train_sentence_count_arr = np.zeros(len(self.author_identity.train_df.index), dtype=int)
        for i in range(len(self.author_identity.train_df.index)):
            text_line = self.author_identity.train_df.loc[i, "text"]
            sentence_arr = nltk.sent_tokenize(text=text_line)
            train_sentence_count_arr[i] = len(sentence_arr)

            if i % 100 == 99:
                print("[Train] i:{0}".format(i))

        plt.hist(x=train_sentence_count_arr, bins=20)
        plt.show()

        # https://stackoverflow.com/questions/23461713/obtaining-values-used-in-boxplot-using-python-and-matplotlib
        # Use this to identify whisker
        plt.boxplot(x=train_sentence_count_arr)
        plt.show()

        test_sentence_count_arr = np.zeros(len(self.author_identity.test_df.index), dtype=int)
        for i in range(len(self.author_identity.test_df.index)):
            text_line = self.author_identity.test_df.loc[i, "text"]
            sentence_arr = nltk.sent_tokenize(text=text_line)
            test_sentence_count_arr[i] = len(sentence_arr)

            if i % 100 == 99:
                print("[Test] i:{0}".format(i))

        plt.hist(x=test_sentence_count_arr, bins=20)
        plt.show()

        plt.boxplot(x=test_sentence_count_arr)
        plt.show()

if __name__ == '__main__':
    author_statistics = AuthorStatistics(set_of_sentences=True, train_folder="../data/machinehack/train",
                                         test_folder="../data/machinehack/test")
    author_statistics.load_train_test_data(train_csv_file='TRAIN.csv', test_csv_file='TEST.csv')
    author_statistics.pre_process_data()
    author_statistics.compute_sentence_count_per_sample_statistics()
