#!/usr/bin/env python

from collections import namedtuple
from extract_features import *
import io
import json
import numpy as np
import os
import pandas as pd
import random
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm


class AuthorIdentity:
    def __init__(self, set_of_sentences, train_folder, test_folder):
        self.set_of_sentences = set_of_sentences  # whether each sample is a sentence or set of sentences
        self.train_folder = train_folder
        self.train_df = None
        self.author_count_df = None
        self.output_train_folder = None
        self.author_count_filename = "author_count.csv"

        self.test_folder = test_folder
        self.test_df = None
        self.output_test_folder = None

        self.output_validation_folder = None

        self.train_indices_dict = dict()
        self.validation_indices_dict = dict()

        # TODO (Later) Think of including numerical param groups. Probably dict with keys 'text', 'numerical' would help.
        self.feature_groups = []
        self.feature_groups.append('bag_of_words')
        self.feature_groups.append('bag_of_pos')
        self.feature_groups.append('bag_of_dependency_triplet')
        self.feature_groups.append('bag_of_syntactic_n_gram_lemma')
        self.feature_groups.append('bag_of_syntactic_n_gram_tag')

        # TODO Also include train_i which represents the instance number of that class. This will be helpful in debugging individual samples.
        self.train_struct = namedtuple(typename='train_struct', field_names='data target')
        self.train_struct.data = dict()
        for feature_group in self.feature_groups:
            self.train_struct.data[feature_group] = []
        self.train_struct.target = []

        self.validation_struct = namedtuple(typename='validation_struct', field_names='data target')
        self.validation_struct.data = dict()
        for feature_group in self.feature_groups:
            self.validation_struct.data[feature_group] = []
        self.validation_struct.target = []

        # usually we won't know the target, but in case we know
        self.test_struct = namedtuple(typename='test_struct', field_names='data target')
        self.test_struct.data = dict()
        for feature_group in self.feature_groups:
            self.test_struct.data[feature_group] = []
        self.test_struct.target = []

        # self.count_vec = CountVectorizer(encoding="utf-8", decode_error="ignore")
        # self.tfidf_transformer = TfidfTransformer()
        self.tfidf_vec = dict()
        for feature_group in self.feature_groups:
            if feature_group == "bag_of_words":
                self.tfidf_vec[feature_group] = TfidfVectorizer(encoding="utf-8", decode_error="ignore", binary=False)
            else:
                self.tfidf_vec[feature_group] = TfidfVectorizer(encoding="utf-8", decode_error="ignore",
                                                                token_pattern=r"\S+", binary=False)
        # Note: default token_pattern in TfidfVectorizer removes the punctuation. We want to split only at space.

        # classifier models
        self.clf_svm = None
        # Following dict will have the svm classifier model for each param group
        self.clf_svm_param_group = dict()

        self.parser = SyntacticParser()

        self.create_output_train_folder()
        self.create_output_test_folder()
        self.create_output_validation_folder()

    def load_train_data(self, train_file, verbose=True):
        self.train_df = pd.read_csv(filepath_or_buffer=train_file, encoding="utf-8")

        if verbose:
            # https://stackoverflow.com/questions/36462852/how-to-read-utf-8-files-with-pandas (Sam's answer)
            print("data type of train_df:")
            print(self.train_df.apply(lambda x: pd.api.types.infer_dtype(x.values)))
            # self.train_df.apply(lambda x: pd.lib.infer_dtype(x.values))

    def load_test_data(self, test_file, verbose=True):
        self.test_df = pd.read_csv(filepath_or_buffer=test_file, encoding="utf-8")

        if verbose:
            print("data type of test_df:")
            print(self.test_df.apply(lambda x: pd.api.types.infer_dtype(x.values)))

    def create_output_train_folder(self):
        # https://stackoverflow.com/questions/3167154/how-to-split-a-dos-path-into-its-components-in-python
        #   Tompa's answer
        path_list = os.path.normpath(self.train_folder).split(os.sep)
        # assumption: train_folder is a relative path
        path_list[1] = "output"
        # https://stackoverflow.com/questions/14826888/python-os-path-join-on-a-list (ATOzTOA's answer)
        self.output_train_folder = os.path.join(*path_list)

        if not os.path.exists(self.output_train_folder):
            os.makedirs(self.output_train_folder)

    def create_output_test_folder(self):
        path_list = os.path.normpath(self.test_folder).split(os.sep)
        path_list[1] = "output"
        self.output_test_folder = os.path.join(*path_list)

        if not os.path.exists(self.output_test_folder):
            os.makedirs(self.output_test_folder)

    def create_output_validation_folder(self):
        path_parts = os.path.split(self.output_train_folder)
        self.output_validation_folder = os.path.join(path_parts[0], "validation")

        if not os.path.exists(self.output_validation_folder):
            os.makedirs(self.output_validation_folder)

    @staticmethod
    def replace_newline_extra_spaces(df, verbose=True):
        """Replace newlines and extra spaces in the sentence.
            This is found in the dataset provided by MachineHack.
        """
        # TODO Experiment with a) apply function   b) .str operator. Check if these can make this function faster.
        for i in df.index:
            """
            if isinstance(df.loc[i, "text"], str):
                df.loc[i, "text"] = unicode(df.loc[i, "text"], "utf-8")
            """
            # df.loc[i, "text"] = re.sub(r'[\n]+', ' ', df.loc[i, "text"])
            # df.loc[i, "text"] = re.sub(r'[" "]{2,}', ' ', df.loc[i, "text"])
            df.loc[i, "text"] = re.sub(r'[\s]+', ' ', df.loc[i, "text"])
            # Each sample in MachineHack is consisting of sentences which are separated by ","
            # Convert this into usual text with sentences.
            # ?? can we combine the following two commands into a single one?
            df.loc[i, "text"] = re.sub(r'[.],', '.', df.loc[i, "text"])
            df.loc[i, "text"] = re.sub(r'[?],', '?', df.loc[i, "text"])

        if verbose:
            print("Replaced extra newline and spaces")

    def compute_count_by_author(self, verbose=True):
        """Compute count by author"""
        # count by author
        # https://stackoverflow.com/questions/19384532/how-to-count-number-of-rows-per-group-and-other-statistics-in-pandas-group-by
        # (Pedro M Duarte's answer to convert series into dataframe)
        self.author_count_df = self.train_df.groupby(by=['author']).size().reset_index(name='count')

        author_count_csv = os.path.join(self.output_train_folder, self.author_count_filename)
        with open(author_count_csv, mode="w") as fd:
            self.author_count_df.to_csv(fd, index=False)

        if verbose:
            print("Number of train samples for each author:")
            for author_i in range(len(self.author_count_df.index)):
                print("{0}: {1}".format(self.author_count_df.ix[author_i, "author"], self.author_count_df.ix[author_i, "count"]))

    def load_count_by_author(self):
        author_count_csv = os.path.join(self.output_train_folder, self.author_count_filename)
        assert os.path.exists(author_count_csv), "author count filename not found"
        self.author_count_df = pd.read_csv(author_count_csv)

    def write_by_author(self):
        """Dump train samples in text files(according to author).
            Assumed that preprocessing of text has already been done.
        """
        if self.author_count_df is None:
            self.compute_count_by_author()

        for author_i in range(len(self.author_count_df.index)):
            author_name = self.author_count_df.ix[author_i, "author"]
            author_row_index_arr = self.train_df.index[self.train_df['author'] == author_name].tolist()
            author_file_name = str(author_name) + ".txt"

            with open(os.path.join(self.output_train_folder, author_file_name), mode="w") as fd:
                for sample_index in author_row_index_arr:
                    fd.write("{0}\n".format(self.train_df.loc[sample_index, "text"].encode("utf-8")))

    def write_test_data(self):
        """Write text data in csv file to text file.
            Before this text has been pre-processed to remove extra spaces and newlines using the function replace_newline_extra_spaces()
        """
        with open(os.path.join(self.output_test_folder, "test.txt"), mode="w") as fd:
            for i in range(len(self.test_df.index)):
                fd.write("{0}\n".format(self.test_df.loc[i, "text"].encode("utf-8")))

    # TODO Move the common portion of write_pos_train_data_by_author() and write_pos_test_data() into a common function
    def write_pos_train_data_by_author(self):
        """Store parser output in corresponding files for author. This is avoid running slow parser in runtime."""
        if self.author_count_df is None:
            self.compute_count_by_author()

        output_train_parser_folder = os.path.join(self.output_train_folder, "parser")
        if not os.path.exists(output_train_parser_folder):
            os.makedirs(output_train_parser_folder)

        for author_i in range(1, 2): # len(self.author_count_df.index)
            author_name = self.author_count_df.ix[author_i, "author"]
            print("author_i: {0} :: name: {1}".format(author_i, author_name))
            author_row_index_arr = self.train_df.index[self.train_df['author'] == author_name].tolist()
            # author_json_file_name = str(author_name) + ".json"
            author_parser_file_name = str(author_name) + ".csv"

            with open(os.path.join(output_train_parser_folder, author_parser_file_name), mode="w") as fd:
                # pos_file_arr = []
                parser_df = pd.DataFrame(columns=['line_index', 'sentence_index', 'word_index', 'text', 'lemma', 'pos',
                                                  'tag', 'dep', 'head_index'])
                line_i = 0
                for sample_index in author_row_index_arr:
                    line_parser_df = self.parser.get_pos_for_line(text_line=self.train_df.loc[sample_index, "text"], n_max_sentence=100)
                    # pos_file_arr.append(pos_line_arr)
                    line_parser_df['line_index'] = line_i
                    parser_df = parser_df.append(line_parser_df, ignore_index=True)

                    if line_i % 50 == 49:
                        print("\tline_i: {0}".format(line_i))
                    line_i += 1

                # json.dump(obj=pos_file_arr, fp=fd)
                parser_df.to_csv(path_or_buf=fd, encoding="utf-8", index=False)

    def write_pos_test_data(self):
        """Store parser output in a file to avoid running slow parser in runtime."""
        # json_file_name = "test.json"
        parser_file_name = "test.csv"

        output_test_parser_folder = os.path.join(self.output_test_folder, "parser")
        if not os.path.exists(output_test_parser_folder):
            os.makedirs(output_test_parser_folder)

        with open(os.path.join(output_test_parser_folder, parser_file_name), mode="w") as fd:
            # pos_file_arr = []
            parser_df = pd.DataFrame(columns=['line_index', 'sentence_index', 'word_index', 'text', 'lemma', 'pos',
                                              'tag', 'dep', 'head_index'])
            for i in range(len(self.test_df.index)):
                # print("i: {0}".format(i))
                line_parser_df = self.parser.get_pos_for_line(text_line=self.test_df.loc[i, "text"], n_max_sentence=100)
                # pos_file_arr.append(pos_line_arr)
                line_parser_df['line_index'] = i
                parser_df = parser_df.append(line_parser_df, ignore_index=True)

                if i % 20 == 19:
                    print("i: {0}".format(i))

            # json.dump(obj=pos_file_arr, fp=fd)
            parser_df.to_csv(path_or_buf=fd, index=False, encoding="utf-8")

    def summary_train_data(self):
        print("Number of train samples: {0}".format(len(self.train_df.index)))

        if self.author_count_df is None:
            self.compute_count_by_author()

        flag_id_column_present = 'id' in self.author_count_df.columns
        print("columns: {0}".format(list(self.author_count_df.columns)))

        # Now display few random text from each author
        n_random_sample = 10
        for author_i in range(len(self.author_count_df.index)):
            author_name = self.author_count_df.ix[author_i, "author"]
            print("Author: {0}".format(author_name))
            # https://stackoverflow.com/questions/21800169/python-pandas-get-index-of-rows-which-column-matches-certain-value
            # (unutbu's answer)
            author_row_index_arr = self.train_df.index[self.train_df['author'] == author_name].tolist()
            for sample_index in random.sample(author_row_index_arr, n_random_sample):
                if flag_id_column_present:
                    print("{0}: {1}".format(self.train_df.loc[sample_index, "id"], self.train_df.loc[sample_index, "text"]))
                else:
                    print("{0}: {1}".format(sample_index, self.train_df.loc[sample_index, "text"].encode("utf-8")))  #

    def split_train_validation_data(self, train_size, random_state=None):
        assert self.author_count_df is not None, "load_count_by_author() should be run before."
        for author_i in range(len(self.author_count_df.index)):
            author_name = self.author_count_df.loc[author_i, "author"]
            author_count = self.author_count_df.loc[author_i, "count"]
            train_indices, validation_indices = train_test_split(range(author_count), train_size=train_size, random_state=random_state)
            # sort the indices as they are returned in random order
            train_indices = sorted(train_indices)
            validation_indices = sorted(validation_indices)
            self.train_indices_dict[author_name] = train_indices
            self.validation_indices_dict[author_name] = validation_indices

    def train(self, flag_load_features=False):
        """Compute and aggregate features for train and validation set."""
        print("------- Train ---------")
        flag_load_parser_csv = True
        if flag_load_features:
            flag_load_parser_csv = False
        for author_i in range(len(self.author_count_df.index)):
            print("\nauthor_i: {0}".format(author_i))
            author_name = self.author_count_df.loc[author_i, "author"]
            author_count = self.author_count_df.loc[author_i, "count"]
            print("\tauthor name: {0} :: count: {1}".format(author_name, author_count))

            parser_df = None
            if flag_load_parser_csv:
                author_parser_csv_file_name = str(author_name) + ".csv"
                author_parser_csv_file_path = os.path.join(self.output_train_folder, author_parser_csv_file_name)
                with io.open(author_parser_csv_file_path, mode="r", encoding="utf-8") as fd:
                    parser_df = pd.read_csv(filepath_or_buffer=fd)
                print("\t#rows in parser_df: {0}".format(len(parser_df.index)))

            feature_group_to_text_dict_for_author = dict()
            if flag_load_features:
                for feature_group in self.feature_groups:
                    author_feature_group_csv_file_name = str(author_name) + ".csv"
                    author_feature_group_csv_file_path = os.path.join(self.output_train_folder, "features",
                                                                      feature_group, author_feature_group_csv_file_name)
                    with io.open(author_feature_group_csv_file_path, mode="r", encoding="utf-8") as fd:
                        feature_group_to_text_dict_for_author[feature_group] = pd.read_csv(filepath_or_buffer=fd)

            # train/validation set relative indices
            # Note: These indices are relative to the indices pertaining to the current author
            #       In other words, these represent the line index in author text file created by write_by_author()
            train_rel_indices = self.train_indices_dict[author_name]
            validation_rel_indices = self.validation_indices_dict[author_name]

            train_author_df = self.train_df[self.train_df['author'] == author_name].reset_index()

            line_i = 0
            for line_index in train_rel_indices:
                if flag_load_features:
                    feature_group_to_text_dict = dict()
                    for feature_group in self.feature_groups:
                        feature_group_to_text_dict[feature_group] = feature_group_to_text_dict_for_author[feature_group].loc[line_index, 'feature_text']
                else:
                    line = train_author_df.loc[line_index, 'text']
                    line = line.strip()

                    parser_line_df = parser_df.loc[parser_df['line_index'] == line_index]
                    feature_group_to_text_dict = self.parser.get_feature_group_to_text(feature_groups=self.feature_groups,
                                                                                       line_text=line, parser_line_df=parser_line_df)

                for feature_group in self.feature_groups:
                    self.train_struct.data[feature_group].append(feature_group_to_text_dict[feature_group])

                self.train_struct.target.append(author_i)

                line_i += 1

            line_i = 0
            for line_index in validation_rel_indices:
                if flag_load_features:
                    feature_group_to_text_dict = dict()
                    for feature_group in self.feature_groups:
                        feature_group_to_text_dict[feature_group] = feature_group_to_text_dict_for_author[feature_group].loc[line_index, 'feature_text']
                else:
                    line = train_author_df.loc[line_index, 'text']
                    line = line.strip()

                    parser_line_df = parser_df.loc[parser_df['line_index'] == line_index]
                    feature_group_to_text_dict = self.parser.get_feature_group_to_text(feature_groups=self.feature_groups,
                                                                                       line_text=line,
                                                                                       parser_line_df=parser_line_df)

                for feature_group in self.feature_groups:
                    self.validation_struct.data[feature_group].append(feature_group_to_text_dict[feature_group])

                self.validation_struct.target.append(author_i)

                line_i += 1

    def test(self, flag_load_features=False):
        print("------- Test ------------")
        flag_load_parser_csv = True
        if flag_load_features:
            flag_load_parser_csv = False

        parser_df = None
        if flag_load_parser_csv:
            with io.open(os.path.join(self.output_test_folder, "test.csv"), mode="r", encoding="utf-8") as fd:
                parser_df = pd.read_csv(filepath_or_buffer=fd)
            print("\t#rows in parser_df: {0}".format(len(parser_df.index)))

        feature_group_to_text_dict_for_test = dict()
        if flag_load_features:
            for feature_group in self.feature_groups:
                test_feature_group_csv_file_path = os.path.join(self.output_test_folder, "features", feature_group, "test.csv")
                with io.open(test_feature_group_csv_file_path, mode="r", encoding="utf-8") as fd:
                    feature_group_to_text_dict_for_test[feature_group] = pd.read_csv(filepath_or_buffer=fd)

        sample_index = 0
        for line_index in self.test_df.index:
            if flag_load_features:
                feature_group_to_text_dict = dict()
                for feature_group in self.feature_groups:
                    feature_group_to_text_dict[feature_group] = feature_group_to_text_dict_for_test[feature_group].loc[line_index, 'feature_text']
            else:
                line = self.test_df.loc[line_index, 'text']
                line = line.strip()

                parser_line_df = parser_df.loc[parser_df['line_index'] == line_index]
                feature_group_to_text_dict = self.parser.get_feature_group_to_text(feature_groups=self.feature_groups,
                                                                                   line_text=line,
                                                                                   parser_line_df=parser_line_df)

            for feature_group in self.feature_groups:
                self.test_struct.data[feature_group].append(feature_group_to_text_dict[feature_group])

            if sample_index % 100 == 99:
                print('\ttest sample index: {0}'.format(sample_index))

            sample_index += 1

    def train_old(self):
        """Compute and aggregate features for train and validation set.
            In this old version, reading the text from text file.
        """
        print("------- Train ---------")
        # flag_load_pos_json = True
        flag_load_parser_csv = True
        for author_i in range(len(self.author_count_df.index)):
            print("\nauthor_i: {0}".format(author_i))
            author_name = self.author_count_df.loc[author_i, "author"]
            author_count = self.author_count_df.loc[author_i, "count"]
            author_file_name = str(author_name) + ".txt"
            print("\tauthor name: {0} :: count: {1}".format(author_name, author_count))

            """
            if flag_load_pos_json:
                author_pos_json_file_name = str(author_name) + ".json"
                with open(os.path.join(self.output_train_folder, author_pos_json_file_name), mode="r") as fd:
                    author_pos_json = json.load(fp=fd)
            """
            parser_df = None
            if flag_load_parser_csv:
                author_parser_csv_file_name = str(author_name) + ".csv"
                author_parser_csv_file_path = os.path.join(self.output_train_folder, author_parser_csv_file_name)
                with io.open(author_parser_csv_file_path, mode="r", encoding="utf-8") as fd:
                    parser_df = pd.read_csv(filepath_or_buffer=fd)

            train_indices = self.train_indices_dict[author_name]
            validation_indices = self.validation_indices_dict[author_name]

            with io.open(os.path.join(self.output_train_folder, author_file_name), mode="r", encoding="utf-8") as fd:
                train_i = 0
                validation_i = 0
                sample_index = 0
                n_train = len(train_indices)
                n_validation = len(validation_indices)

                for line in fd:
                    line = line.strip()
                    # print("\tsample_index: {0}".format(sample_index))

                    param_name_to_text_dict = dict()
                    # 1. extract features
                    # 2. convert list of strings into space separated string
                    if 'bag_of_words' in self.feature_groups:
                        bag_of_words = extract_bag_of_words(line)
                        param_text_bag_of_words = " ".join(bag_of_words)
                        param_name_to_text_dict['bag_of_words'] = param_text_bag_of_words

                    parser_line_df = None
                    if flag_load_parser_csv:
                        parser_line_df = parser_df.loc[parser_df['line_index'] == sample_index]

                    if 'bag_of_pos' in self.feature_groups:
                        if flag_load_parser_csv:
                            pos_line_arr = []
                            for sentence_index in parser_line_df['sentence_index'].unique():
                                pos_line_arr.append(parser_line_df.loc[parser_line_df['sentence_index'] == sentence_index, 'pos'].tolist())
                            bag_of_pos = self.parser.get_bag_of_pos(pos_line_arr=pos_line_arr)
                        else:
                            bag_of_pos = self.parser.extract_bag_of_pos(text_line=line)
                        param_text_bag_of_pos = " ".join(bag_of_pos)
                        param_name_to_text_dict['bag_of_pos'] = param_text_bag_of_pos

                    if 'bag_of_dependency_triplet' in self.feature_groups:
                        if flag_load_parser_csv:
                            bag_of_dependency_triplet = self.parser.get_bag_of_dependency_triplet(parser_line_df=parser_line_df)
                            param_text_bag_of_dependency_triplet = " ".join(bag_of_dependency_triplet)
                            param_name_to_text_dict['bag_of_dependency_triplet'] = param_text_bag_of_dependency_triplet
                        else:
                            print("Dependency triplet computing function not yet added.")

                    if train_i < n_train and sample_index == train_indices[train_i]:
                        # sample_index belongs to train set
                        for feature_group in self.feature_groups:
                            self.train_struct.data[feature_group].append(param_name_to_text_dict[feature_group])

                        self.train_struct.target.append(author_i)

                        if train_i % 100 == 99:
                            print("\ttrain_i: {0}".format(train_i))

                        # shift forward
                        train_i += 1
                    elif validation_i < n_validation:
                        # sample_index belongs to validation set
                        assert sample_index == validation_indices[validation_i], \
                            "sample_index: {0} should be part of validation indices".format(sample_index)

                        for feature_group in self.feature_groups:
                            self.validation_struct.data[feature_group].append(param_name_to_text_dict[feature_group])

                        self.validation_struct.target.append(author_i)

                        # shift forward
                        validation_i += 1
                    else:
                        assert False, "Something wrong in handling train_i or validation_i"

                    # update to next sample
                    sample_index += 1

                    if sample_index == author_count:
                        break

    def test_old(self, test_file_name):
        print("------- Test ------------")
        # flag_load_pos_json = True
        flag_load_parser_csv = True

        """
        if flag_load_pos_json:
            test_pos_json_file_name = "test.json"
            with open(os.path.join(self.output_test_folder, test_pos_json_file_name), mode="r") as fd:
                test_pos_json = json.load(fp=fd)
        """

        parser_df = None
        if flag_load_parser_csv:
            test_parser_csv_file_name = "test.csv"
            with io.open(os.path.join(self.output_test_folder, test_parser_csv_file_name), mode="r", encoding="utf-8") as fd:
                parser_df = pd.read_csv(filepath_or_buffer=fd)

        with io.open(os.path.join(self.output_test_folder, test_file_name), mode="r", encoding="utf-8") as fd:
            sample_index = 0
            for line in fd:
                line = line.strip()
                if line == "":
                    continue
                param_name_to_text_dict = dict()
                # 1. extract features
                # 2. convert list of strings into space separated string
                if 'bag_of_words' in self.feature_groups:
                    bag_of_words = extract_bag_of_words(line)
                    param_text_bag_of_words = " ".join(bag_of_words)
                    param_name_to_text_dict['bag_of_words'] = param_text_bag_of_words

                parser_line_df = None
                if flag_load_parser_csv:
                    parser_line_df = parser_df.loc[parser_df['line_index'] == sample_index]

                if 'bag_of_pos' in self.feature_groups:
                    if flag_load_parser_csv:
                        pos_line_arr = []
                        for sentence_index in parser_line_df['sentence_index'].unique():
                            pos_line_arr.append(parser_line_df.loc[parser_line_df['sentence_index'] == sentence_index, 'pos'].tolist())
                        bag_of_pos = self.parser.get_bag_of_pos(pos_line_arr=pos_line_arr)
                    else:
                        bag_of_pos = self.parser.extract_bag_of_pos(text_line=line)
                    param_text_bag_of_pos = " ".join(bag_of_pos)
                    param_name_to_text_dict['bag_of_pos'] = param_text_bag_of_pos

                if 'bag_of_dependency_triplet' in self.feature_groups:
                    if flag_load_parser_csv:
                        bag_of_dependency_triplet = self.parser.get_bag_of_dependency_triplet(
                            parser_line_df=parser_line_df)
                        param_text_bag_of_dependency_triplet = " ".join(bag_of_dependency_triplet)
                        param_name_to_text_dict['bag_of_dependency_triplet'] = param_text_bag_of_dependency_triplet
                    else:
                        print("Dependency triplet computing function not yet added.")

                for feature_group in self.feature_groups:
                    self.test_struct.data[feature_group].append(param_name_to_text_dict[feature_group])

                if sample_index % 100 == 99:
                    print('\ttest sample index: {0}'.format(sample_index))

                # update to next sample
                sample_index += 1

    def dump_train_features(self):
        flag_load_parser_csv = True
        for author_i in range(len(self.author_count_df.index)):
            print("\nauthor_i: {0}".format(author_i))
            author_name = self.author_count_df.loc[author_i, "author"]
            author_count = self.author_count_df.loc[author_i, "count"]
            print("\tauthor name: {0} :: count: {1}".format(author_name, author_count))

            parser_df = None
            if flag_load_parser_csv:
                author_parser_csv_file_name = str(author_name) + ".csv"
                parser_folder = os.path.join(self.output_train_folder, "parser")
                author_parser_csv_file_path = os.path.join(parser_folder, author_parser_csv_file_name)
                with io.open(author_parser_csv_file_path, mode="r", encoding="utf-8") as fd:
                    parser_df = pd.read_csv(filepath_or_buffer=fd)
                print("\t#rows in parser_df: {0}".format(len(parser_df.index)))

            train_author_df = self.train_df[self.train_df['author'] == author_name].reset_index()

            author_feature_group_df_dict = dict()
            for feature_group in self.feature_groups:
                author_feature_group_df_dict[feature_group] = pd.DataFrame(columns=['line_index', 'feature_text'])

            # Note: line_index is wrt to the lines belonging to the current author and not wrt the csv filename
            #  containing the train data for all authors
            line_i = 0
            for line_index in train_author_df.index:
                line = train_author_df.loc[line_index, 'text']
                line = line.strip()

                parser_line_df = parser_df.loc[parser_df['line_index'] == line_index]
                feature_group_to_text_dict = self.parser.get_feature_group_to_text(feature_groups=self.feature_groups,
                                                                                   line_text=line,
                                                                                   parser_line_df=parser_line_df)

                for feature_group in self.feature_groups:
                    author_feature_group_df_dict[feature_group].loc[line_index, 'line_index'] = line_index
                    author_feature_group_df_dict[feature_group].loc[line_index, 'feature_text'] = feature_group_to_text_dict[feature_group]

                if line_i % 100 == 99:
                    print("\tline_i: {0}".format(line_i))

                line_i += 1

            # Now dump into csv file
            for feature_group in self.feature_groups:
                feature_folder = os.path.join(self.output_train_folder, 'features', feature_group)
                if not os.path.exists(feature_folder):
                    os.makedirs(feature_folder)
                author_feature_group_csv_file_path = os.path.join(feature_folder, str(author_name)+".csv")
                with open(name=author_feature_group_csv_file_path, mode='w') as fd:
                    author_feature_group_df_dict[feature_group].to_csv(path_or_buf=fd, index=False, encoding='utf-8')

    def dump_test_features(self):
        flag_load_parser_csv = True
        print("\ntest")

        parser_df = None
        if flag_load_parser_csv:
            test_parser_csv_file_name = "test.csv"
            parser_folder = os.path.join(self.output_test_folder, "parser")
            with io.open(os.path.join(parser_folder, test_parser_csv_file_name), mode="r",
                         encoding="utf-8") as fd:
                parser_df = pd.read_csv(filepath_or_buffer=fd)
            print("\t#rows in parser_df: {0}".format(len(parser_df.index)))

        test_feature_group_text_df_dict = dict()
        # create empty dataframe for each feature group
        for feature_group in self.feature_groups:
            test_feature_group_text_df_dict[feature_group] = pd.DataFrame(columns=['line_index', 'feature_text'])

        line_i = 0
        for line_index in self.test_df.index:
            line = self.test_df.loc[line_index, 'text']
            line = line.strip()

            parser_line_df = parser_df.loc[parser_df['line_index'] == line_index]
            feature_group_to_text_dict = self.parser.get_feature_group_to_text(feature_groups=self.feature_groups,
                                                                               line_text=line,
                                                                               parser_line_df=parser_line_df)

            for feature_group in self.feature_groups:
                test_feature_group_text_df_dict[feature_group].loc[line_index, 'line_index'] = line_index
                test_feature_group_text_df_dict[feature_group].loc[line_index, 'feature_text'] = feature_group_to_text_dict[feature_group]

            if line_i % 100 == 99:
                print("\tline_i: {0}".format(line_i))

            line_i += 1

        # Now dump into csv file
        for feature_group in self.feature_groups:
            feature_folder = os.path.join(self.output_test_folder, 'features', feature_group)
            if not os.path.exists(feature_folder):
                os.makedirs(feature_folder)
            test_feature_group_csv_file_path = os.path.join(feature_folder, "test.csv")
            with open(name=test_feature_group_csv_file_path, mode='w') as fd:
                test_feature_group_text_df_dict[feature_group].to_csv(path_or_buf=fd, index=False, encoding='utf-8')

    # TODO refactor train_svm(). predict_svm() so that a) saving feature groups prob, b) loading feature group prob etc can be done both for train and validation set
    def train_svm(self, save_feature_groups_prob=False, load_feature_groups_prob=False, prob_feature_groups_csv_filename=None, save_result_prob=False):
        # Train svm model for each individual param group
        proba_matrix = None
        header_arr = []
        if load_feature_groups_prob is False:
            for feature_group in self.feature_groups:
                print("\nTrain model for Param Group: {0}".format(feature_group))
                X_train_tfidf = self.tfidf_vec[feature_group].fit_transform(self.train_struct.data[feature_group])
                self.clf_svm_param_group[feature_group] = svm.LinearSVC().fit(X=X_train_tfidf, y=self.train_struct.target)
                print_topk_features(clf=self.clf_svm_param_group[feature_group], k=40, vectorizer=self.tfidf_vec[feature_group])
                proba = self.clf_svm_param_group[feature_group].decision_function(X=X_train_tfidf)
                print("classes: {0}".format(self.clf_svm_param_group[feature_group].classes_))

                if proba_matrix is None:
                    proba_matrix = proba
                else:
                    # print("proba_matrix.shape (before): {0}".format(proba_matrix.shape))
                    proba_matrix = np.hstack((proba_matrix, proba))
                    # print("proba_matrix.shape (after): {0}".format(proba_matrix.shape))

                header_arr.extend([feature_group + '_' + str(x) for x in self.clf_svm_param_group[feature_group].classes_])
        else:
            # Load the feature groups probability from the csv file
            prob_feature_groups_csv_file_path = os.path.join(self.output_train_folder, "prob", prob_feature_groups_csv_filename)
            assert prob_feature_groups_csv_file_path is not None, "prob_feature_groups_csv_filename should be passed if load_feature_groups_prob=True"
            with open(name=prob_feature_groups_csv_file_path, mode='r') as fd:
                proba_df = pd.read_csv(filepath_or_buffer=fd)
            print("Loaded prob feature group from {0}".format(prob_feature_groups_csv_file_path))

            assert self.train_struct.target == proba_df['target'].tolist(), "train_struct.target is not matching target column of dataframe."

            # TODO drop columns author_0, etc if present
            proba_df = proba_df.drop(['target'], axis=1)
            proba_matrix = proba_df.values

        if len(self.feature_groups) > 1:
            # self.clf_svm = svm.LinearSVC().fit(X=proba_matrix, y=self.train_struct.target)
            self.clf_svm = svm.SVC(C=1.0, probability=True).fit(X=proba_matrix, y=self.train_struct.target)
            # self.clf_svm = LogisticRegression(random_state=100).fit(X=proba_matrix, y=self.train_struct.target)
            print("Combined model classes: {0}".format(self.clf_svm.classes_))
            # print_topk_features(clf=self.clf_svm, k=20)  # Won't work for non-linear kernel
        else:
            feature_group = list(self.feature_groups)[0]
            self.clf_svm = self.clf_svm_param_group[feature_group]

        if proba_matrix is not None:
            predicted = self.clf_svm.predict(X=proba_matrix)

            n_class = len(self.author_count_df.index)
            confusion_matrix = pd.DataFrame(data=np.zeros((n_class, n_class), dtype=int), index=range(n_class),
                                            columns=range(n_class))

            for i in range(len(self.train_struct.target)):
                predicted_class_index = predicted[i]
                truth_class_index = self.train_struct.target[i]

                confusion_matrix.loc[predicted_class_index, truth_class_index] += 1

            print("\n[Train] Confusion Matrix: Columns -> truth class, Rows -> predicted class")
            print(confusion_matrix)

            n_correct_samples = sum(map(lambda class_i: confusion_matrix.loc[class_i, class_i], range(n_class)))
            print("accuracy: {0} :: n_instances: {1}".format(n_correct_samples * 1.0 / len(predicted), len(predicted)))

        prob_folder = os.path.join(self.output_train_folder, "prob")
        if not os.path.exists(prob_folder):
            os.makedirs(prob_folder)

        if save_feature_groups_prob and proba_matrix is not None:
            # Now prepend target values as 1st column on proba_matrix
            proba_matrix = np.hstack(
                (np.array(self.train_struct.target).reshape(len(self.train_struct.target), 1), proba_matrix))
            header_str = 'target,' + ','.join(header_arr)

            prob_file_name = "train_prob_bag_of_words_pos_dependency_triplet_syntactic_n_gram_lemma_tag.csv"
            np.savetxt(fname=os.path.join(prob_folder, prob_file_name), X=proba_matrix, delimiter=",",
                       header=header_str, comments='')
        elif save_result_prob and proba_matrix is not None:
            proba = self.clf_svm.predict_proba(X=proba_matrix)
            dist_arr = self.clf_svm.decision_function(X=proba_matrix)

            proba_matrix = proba
            header_arr.extend(['author_' + str(x) for x in self.clf_svm.classes_])
            # Now prepend target values as 1st column on proba_matrix
            proba_matrix = np.hstack(
                (np.array(self.train_struct.target).reshape(len(self.train_struct.target), 1), proba_matrix))
            header_str = 'target,' + ','.join(header_arr)

            prob_file_name = "train_prob_result_C_1.csv"
            np.savetxt(fname=os.path.join(prob_folder, prob_file_name), X=proba_matrix, delimiter=",",
                   header=header_str, comments='')

            # Similarly dump the distance matrix
            dist_matrix = dist_arr
            dist_matrix = np.hstack((np.array(self.train_struct.target).reshape(len(self.train_struct.target), 1), dist_matrix))

            dist_file_name = "train_dist_result_C_1.csv"
            np.savetxt(fname=os.path.join(prob_folder, dist_file_name), X=dist_matrix, delimiter=",",
                       header=header_str, comments='')

    # TODO Make it generic to run for both train and validation set
    def predict_svm(self, save_feature_groups_prob=False, load_feature_groups_prob=False, prob_feature_groups_csv_filename=None, save_result_prob=False):
        proba_matrix = None
        header_arr = []
        if load_feature_groups_prob is False:
            if len(self.feature_groups) > 1:
                for feature_group in self.feature_groups:
                    X_validation_tfidf = self.tfidf_vec[feature_group].transform(self.validation_struct.data[feature_group])
                    proba = self.clf_svm_param_group[feature_group].decision_function(X=X_validation_tfidf)

                    if proba_matrix is None:
                        proba_matrix = proba
                    else:
                        proba_matrix = np.hstack((proba_matrix, proba))

                    header_arr.extend([feature_group + '_' + str(x) for x in self.clf_svm_param_group[feature_group].classes_])
            else:
                feature_group = self.feature_groups[0]
                X_validation_tfidf = self.tfidf_vec[feature_group].transform(self.validation_struct.data[feature_group])
                proba = self.clf_svm_param_group[feature_group].decision_function(X=X_validation_tfidf)
                proba_matrix = proba
                header_arr.extend(
                    [feature_group + '_' + str(x) for x in self.clf_svm_param_group[feature_group].classes_])
        else:
            # Load the feature groups probability from the csv file
            prob_feature_groups_csv_file_path = os.path.join(self.output_validation_folder,
                                                             prob_feature_groups_csv_filename)
            assert prob_feature_groups_csv_file_path is not None, "prob_feature_groups_csv_filename should be passed if load_feature_groups_prob=True"
            with open(name=prob_feature_groups_csv_file_path, mode='r') as fd:
                proba_df = pd.read_csv(filepath_or_buffer=fd)
            print("Loaded prob feature group from {0}".format(prob_feature_groups_csv_file_path))

            assert self.validation_struct.target == proba_df['target'].tolist(), "mismatch of target column with validation_struct.target"
            result_columns = []
            for author_i in range(10):
                result_columns.append('author_' + str(author_i))
            columns_to_drop = ['target']
            columns_to_drop.extend(result_columns)
            proba_df = proba_df.drop(columns_to_drop, axis=1)
            proba_matrix = proba_df.values

        """
        # Move the folder creation code into a function which should be called as part of init function
        path_parts = os.path.split(self.output_train_folder)
        validation_folder = os.path.join(path_parts[0], "validation")

        if not os.path.exists(validation_folder):
            os.makedirs(validation_folder)
        """

        if save_feature_groups_prob and proba_matrix is not None:
            # Now prepend target values as 1st column on proba_matrix
            proba_feature_groups_matrix = np.hstack((np.array(self.validation_struct.target).reshape(len(self.validation_struct.target),1), proba_matrix))
            header_str = 'target,' + ','.join(header_arr)

            # https://stackoverflow.com/questions/6081008/dump-a-numpy-array-into-a-csv-file (Jim Brissom's answer)
            np.savetxt(fname=os.path.join(self.output_validation_folder, "validation_prob_bag_of_words_pos_dependency_triplet_syntactic_n_gram_lemma_tag.csv"),
                       X=proba_feature_groups_matrix, delimiter=",", header=header_str, comments='')

        proba_result_matrix = None
        dist_result_matrix = None
        if len(self.feature_groups) > 1:
            predicted = self.clf_svm.predict(X=proba_matrix)
            proba = self.clf_svm.predict_proba(X=proba_matrix)
            proba_result_matrix = proba
            dist_result_matrix = self.clf_svm.decision_function(X=proba_matrix)
        else:
            feature_group = self.feature_groups[0]
            X_validation_tfidf = self.tfidf_vec[feature_group].transform(self.validation_struct.data[feature_group])
            predicted = self.clf_svm.predict(X=X_validation_tfidf)
            # LinearSVC doesn't have predict_proba()
            # proba_matrix = self.clf_svm.predict_proba(X=X_validation_tfidf)
            proba_result_matrix = self.clf_svm.decision_function(X=X_validation_tfidf)
            dist_result_matrix = proba_result_matrix

        if save_result_prob and proba_result_matrix is not None:
            header_arr = ['author_' + str(x) for x in self.clf_svm.classes_]
            proba_result_matrix = np.hstack(
                (np.array(self.validation_struct.target).reshape(len(self.validation_struct.target), 1), proba_result_matrix))
            header_str = 'target,' + ','.join(header_arr)

            np.savetxt(fname=os.path.join(self.output_validation_folder, "validation_prob_result_C_1.csv"),
                       X=proba_result_matrix, delimiter=",", header=header_str, comments='')

        if save_result_prob and dist_result_matrix is not None:
            header_arr = ['author_' + str(x) for x in self.clf_svm.classes_]
            dist_result_matrix = np.hstack(
                (np.array(self.validation_struct.target).reshape(len(self.validation_struct.target), 1), dist_result_matrix))
            header_str = 'target,' + ','.join(header_arr)

            np.savetxt(fname=os.path.join(self.output_validation_folder, "validation_dist_result_C_1.csv"),
                       X=dist_result_matrix, delimiter=",", header=header_str, comments='')

        n_class = len(self.author_count_df.index)
        confusion_matrix = pd.DataFrame(data=np.zeros((n_class, n_class), dtype=int), index=range(n_class), columns=range(n_class))

        for i in range(len(self.validation_struct.target)):
            predicted_class_index = predicted[i]
            truth_class_index = self.validation_struct.target[i]

            confusion_matrix.loc[predicted_class_index, truth_class_index] += 1

        print("\n[Validation] Confusion Matrix: Columns -> truth class, Rows -> predicted class")
        print(confusion_matrix)

        n_correct_samples = sum(map(lambda class_i: confusion_matrix.loc[class_i, class_i], range(n_class)))
        print("accuracy: {0} :: n_instances: {1}".format(n_correct_samples*1.0/len(predicted), len(predicted)))  # {0:.3f}

    def predict_test_svm(self):
        proba_matrix = None
        header_arr = []
        if len(self.feature_groups) > 1:
            for feature_group in self.feature_groups:
                X_test_tfidf = self.tfidf_vec[feature_group].transform(self.test_struct.data[feature_group])
                proba = self.clf_svm_param_group[feature_group].decision_function(X=X_test_tfidf)

                if proba_matrix is None:
                    proba_matrix = proba
                else:
                    proba_matrix = np.hstack((proba_matrix, proba))

                header_arr.extend([feature_group + '_' + str(x) for x in self.clf_svm_param_group[feature_group].classes_])

            predicted = self.clf_svm.predict(X=proba_matrix)
            # save the class probabilities
            proba = self.clf_svm.predict_proba(X=proba_matrix)
            proba_matrix = np.hstack((proba_matrix, proba))
            header_arr.extend(['author_' + str(x) for x in self.clf_svm.classes_])
        else:
            feature_group = list(self.feature_groups)[0]
            X_test_tfidf = self.tfidf_vec[feature_group].transform(self.test_struct.data[feature_group])
            predicted = self.clf_svm.predict(X=X_test_tfidf)

        predicted_df = pd.DataFrame(data=predicted, columns=['author'], dtype=int)
        result_dir = os.path.join(self.output_test_folder, "result")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        excel_writer = pd.ExcelWriter(path=os.path.join(result_dir, "author_bag_of_words_pos_dependency_triplet_syntactic_n_gram_lemma_tag.xlsx"))
        predicted_df.to_excel(excel_writer=excel_writer, index=False)
        excel_writer.save()

        if proba_matrix is not None:
            # Prepend the predicted author
            proba_matrix = np.hstack((np.array(predicted).reshape(len(predicted), 1), proba_matrix))
            header_str = 'predicted,' + ','.join(header_arr)

            np.savetxt(fname=os.path.join(self.output_test_folder, "result", "test_prob.csv"), X=proba_matrix,
                       delimiter=",", header=header_str, comments='')

    def compare_excel_results(self, excel_file_0, excel_file_1):
        """Compare test results submitted in excel format(in MachineHack)."""
        df_0 = pd.read_excel(excel_file_0)
        df_1 = pd.read_excel(excel_file_1)

        assert len(df_0.index) == len(df_1.index), "Both excel should have same number of instances"
        n_sample = len(df_0.index)

        self.load_test_data(test_file=os.path.join(self.test_folder, "TEST.csv"))
        self.replace_newline_extra_spaces(df=self.test_df)

        n_mismatch = 0
        mismatch_dict = dict()
        for i in range(n_sample):
            if df_0.loc[i, "author"] != df_1.loc[i, "author"]:
                sample_text = self.test_df.loc[i, "text"].encode("utf-8")
                print("mismatch #{0}: sample #{1} : predicted author: {2}(prev), {3}(cur) :: {4}".format(n_mismatch, i, df_0.loc[i, "author"], df_1.loc[i, "author"], sample_text))
                n_mismatch += 1
                if df_0.loc[i, "author"] not in mismatch_dict:
                    mismatch_dict[df_0.loc[i, "author"]] = dict()
                if df_1.loc[i, "author"] not in mismatch_dict[df_0.loc[i, "author"]]:
                    mismatch_dict[df_0.loc[i, "author"]][df_1.loc[i, "author"]] = 0
                mismatch_dict[df_0.loc[i, "author"]][df_1.loc[i, "author"]] += 1

        print("mismatch: {0} out of {1}".format(n_mismatch, n_sample))
        for author_0 in mismatch_dict:
            print("author_0: {0} :: {1}".format(author_0, sorted(mismatch_dict[author_0].iteritems(), key=lambda x: x[1], reverse=True)))


if __name__ == "__main__":
    # True: Load the data from csv, pre-process and write into text file.
    # False: Read the data from text file written during pre-processing
    flag_pre_process = False
    compare_results = False
    author_identity = AuthorIdentity(set_of_sentences=True, train_folder="../data/machinehack/train",
                                     test_folder="../data/machinehack/test")
    if flag_pre_process:
        author_identity.load_train_data(train_file=os.path.join(author_identity.train_folder, "TRAIN.csv"))
        author_identity.replace_newline_extra_spaces(df=author_identity.train_df)

        # author_identity.summary_train_data()
        # author_identity.write_by_author()
        author_identity.write_pos_train_data_by_author()

        author_identity.load_test_data(test_file=os.path.join(author_identity.test_folder, "TEST.csv"))
        author_identity.replace_newline_extra_spaces(df=author_identity.test_df)

        # author_identity.write_test_data()
        author_identity.write_pos_test_data()
    elif not compare_results:
        author_identity.load_train_data(train_file=os.path.join(author_identity.train_folder, "TRAIN.csv"))
        author_identity.replace_newline_extra_spaces(df=author_identity.train_df)

        author_identity.load_count_by_author()
        author_identity.split_train_validation_data(train_size=0.75, random_state=100)

        author_identity.load_test_data(test_file=os.path.join(author_identity.test_folder, "TEST.csv"))
        author_identity.replace_newline_extra_spaces(df=author_identity.test_df)

        flag_dump_features = False

        if flag_dump_features:
            author_identity.dump_train_features()
            author_identity.dump_test_features()
        else:
            author_identity.train(flag_load_features=True)

            author_identity.train_svm(save_feature_groups_prob=False, load_feature_groups_prob=True,
                                      prob_feature_groups_csv_filename="train_prob_bag_of_words_pos_dependency_triplet_syntactic_n_gram_lemma_tag.csv",
                                      save_result_prob=True)
            author_identity.predict_svm(save_feature_groups_prob=False, load_feature_groups_prob=True,
                                        prob_feature_groups_csv_filename="validation_prob_bag_of_words_pos_dependency_triplet_syntactic_n_gram_lemma_tag.csv",
                                        save_result_prob=True)

            # author_identity.test(flag_load_features=True)
            # author_identity.predict_test_svm()
    else:
        # TODO Better to move this section to result_analysis.py
        # This section is to compare the excel files created for submissions
        excel_file0 = os.path.join(author_identity.output_test_folder, "result", "author_1.xlsx")
        excel_file1 = os.path.join(author_identity.output_test_folder, "result", "author_bag_of_words_pos_dependency_triplet_syntactic_n_gram_lemma_tag.xlsx")
        author_identity.compare_excel_results(excel_file_0=excel_file0, excel_file_1=excel_file1)

"""
Problem Statement:
    1. https://www.kaggle.com/c/spooky-author-identification/
    2. https://www.machinehack.com/course/whose-line-is-it-anyway-identify-the-author-hackathon/
    3. https://pan.webis.de/tasks.html
            Several category of tasks.

Data description:
    1. Kaggle: Each sample is a single sentence.
    2. MachineHack: Each sample consists of a set of 10 sentences.
                    Though this is mentioned in the description, but not true in many cases. Its more than 10 in many cases.

Tasks:
    1. Exploratory data analysis
        https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial/notebook
    2. Feature engineering:
        https://www.kaggle.com/sudalairajkumar/simple-feature-engg-notebook-spooky-author

Resource:
    https://pandas.pydata.org/pandas-docs/stable/overview.html#mutability-and-copying-of-data
        All pandas data structures are value-mutable (the values they contain can be altered) but not always size-mutable.

TODO:
    1. https://stackoverflow.com/questions/47312432/attributeerrorlinearsvc-object-has-no-attribute-predict-proba?rq=1
        sascha suggests to create own pipeline of probability calibration using LinearSVC
    2. Handle author name different from numbers in natural order. Would definitely require this to run on Kaggle dataset.
    3. Results alternate to confusion matrix:
            Count the rank of true class in probability order. Marcia Fissette had mentioned this on page #5 of her thesis.
"""