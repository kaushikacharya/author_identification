#!/usr/bin/env python

import nltk
import numpy as np
import pandas as pd
import spacy


def extract_bag_of_words(text):
    bag_of_words = []
    sentence_arr = nltk.sent_tokenize(text=text)
    for sentence in sentence_arr:
        bag_of_words.extend(nltk.word_tokenize(sentence.lower()))

    return bag_of_words


def print_topk_features(clf, k, vectorizer=None):
    """Display topK features for linear classifier.
        Note: coef_ is only available when using a linear kernel
    """
    print("\nTop {0} features".format(k))
    feature_names = None
    if vectorizer:
        feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(clf.classes_):
        topK = np.argsort(clf.coef_[i])[::-1][:k]
        if vectorizer:
            print("{0}: {1} total features :: {2}".format(class_label, len(clf.coef_[i]),
                                                          [(feature_names[j], j, clf.coef_[i][j]) for j in topK]))
        else:
            print("{0}: {1} total features :: {2}".format(class_label, len(clf.coef_[i]), [(j, clf.coef_[i][j]) for j in topK]))


class SyntacticParser:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def extract_bag_of_pos(self, text_line):
        bag_of_pos = []
        sentence_arr = nltk.sent_tokenize(text=text_line)
        for sentence in sentence_arr:
            if isinstance(sentence, unicode):
                sentence_unicode = sentence
            else:
                sentence_unicode = unicode(sentence, encoding="utf-8")
            doc = self.nlp(sentence_unicode)

            pos_arr = []
            for token in doc:
                pos_arr.append(token.pos_)

            # now create bigram of POS
            pos_bigram_arr = map(lambda i: pos_arr[i]+"_"+pos_arr[i+1], range(len(pos_arr)-1))

            bag_of_pos.extend(pos_arr)
            bag_of_pos.extend(pos_bigram_arr)

        return bag_of_pos

    @staticmethod
    def get_bag_of_pos(pos_line_arr):
        """Create Bag of parts of speech.

            Parameters
            ----------
            pos_line_arr : list of list
                    pos_line_arr[i]: List of POS of sentence #i
        """
        bag_of_pos = []
        for sentence_i in range(len(pos_line_arr)):
            pos_sentence_arr = pos_line_arr[sentence_i]
            pos_bigram_arr = map(lambda i: pos_sentence_arr[i]+"_"+pos_sentence_arr[i+1], range(len(pos_sentence_arr)-1))
            pos_trigram_arr = map(lambda i: pos_sentence_arr[i]+"_"+pos_sentence_arr[i+1]+"_"+pos_sentence_arr[i+2], range(len(pos_sentence_arr)-2))

            bag_of_pos.extend(pos_sentence_arr)
            bag_of_pos.extend(pos_bigram_arr)
            bag_of_pos.extend(pos_trigram_arr)

        return bag_of_pos

    def get_pos_for_line(self, text_line, n_max_sentence=np.inf):
        """Extract parser output e.g. POS for the text line which may contain multiple sentences."""
        # pos_line_arr = []
        line_parser_df = pd.DataFrame(columns=['line_index', 'sentence_index', 'word_index', 'text', 'lemma', 'pos',
                                               'tag', 'dep', 'head_index'])
        # column "line_index" will be populated by the calling function/script
        sentence_arr = nltk.sent_tokenize(text=text_line)

        i = 0
        sentence_index = 0
        for sentence in sentence_arr:
            if isinstance(sentence, unicode):
                sentence_unicode = sentence
            else:
                sentence_unicode = unicode(sentence, encoding="utf-8")
            doc = self.nlp(sentence_unicode)
            # print("\tsentence_index: {0}".format(sentence_index))

            # pos_arr = []
            word_index = 0
            for token in doc:
                # pos_arr.append(token.pos_)
                line_parser_df.loc[i, 'sentence_index'] = sentence_index
                line_parser_df.loc[i, 'word_index'] = word_index
                line_parser_df.loc[i, 'text'] = token.text
                line_parser_df.loc[i, 'lemma'] = token.lemma_
                line_parser_df.loc[i, 'pos'] = token.pos_
                line_parser_df.loc[i, 'tag'] = token.tag_
                line_parser_df.loc[i, 'dep'] = token.dep_
                line_parser_df.loc[i, 'head_index'] = token.head.i

                # update for next iteration
                word_index += 1
                i += 1

            if sentence_index >= n_max_sentence:
                print("\tTotal sentences: {0} :: Stopping at sentence #{1}".format(len(sentence_arr), sentence_index))
                break

            # pos_line_arr.append(pos_arr)
            sentence_index += 1

        return line_parser_df

    @staticmethod
    def get_bag_of_dependency_triplet(parser_line_df):
        """Extract bag of dependency triplet features

            Parameters
            ----------
            parser_line_df : DataFrame
                            Parser data frame corresponding to the line of text which may contain multiple sentences.

            Notes
            -----
            Author identification in short texts by Marcia Fissette(2010)
        """
        bag_of_dependency_triplet = []
        for sentence_index in parser_line_df['sentence_index'].unique():
            # print("\tsentence_index: {0}".format(sentence_index))
            parser_sentence_df = parser_line_df.loc[parser_line_df['sentence_index'] == sentence_index]
            for row_index in parser_sentence_df.index:
                if parser_sentence_df.loc[row_index, 'word_index'] == parser_sentence_df.loc[row_index, 'head_index']:
                    # This one is root
                    continue
                if pd.isnull(parser_sentence_df.loc[row_index, 'pos']) or \
                    pd.isnull(parser_sentence_df.loc[row_index, 'text']) or \
                        pd.isnull(parser_sentence_df.loc[row_index, 'dep']):
                    continue
                dependency_triplet_str = parser_sentence_df.loc[row_index, 'pos'] + ":" + \
                                         parser_sentence_df.loc[row_index, 'text'].lower() + "_" + \
                                         parser_sentence_df.loc[row_index, 'dep'] + "_"
                # Identify the row index whose word_index column value matches head_index column value of current row_index
                """
                row_head_index = parser_sentence_df.index[parser_sentence_df['word_index'] ==
                                                          parser_sentence_df.loc[row_index, 'head_index']].tolist()
                assert len(row_head_index) == 1, "Multiple rows: {0} have same word_index.".format(row_head_index)
                row_head_index = row_head_index[0]
                """
                row_head_index = parser_sentence_df.index[parser_sentence_df.loc[row_index, 'head_index']]
                if pd.isnull(parser_sentence_df.loc[row_head_index, 'pos']) or \
                        pd.isnull( parser_sentence_df.loc[row_head_index, 'text']):
                    continue
                dependency_triplet_str += parser_sentence_df.loc[row_head_index, 'pos'] + ":" + \
                                          parser_sentence_df.loc[row_head_index, 'text'].lower()
                bag_of_dependency_triplet.append(dependency_triplet_str)

        return bag_of_dependency_triplet

    @staticmethod
    def get_bag_of_syntactic_n_gram_lemma(parser_line_df, n_gram=2):
        assert n_gram == 2, "Need to extend for n_gram > 2"
        bag_of_syntactic_n_gram_lemma = []
        for sentence_index in parser_line_df['sentence_index'].unique():
            parser_sentence_df = parser_line_df.loc[parser_line_df['sentence_index'] == sentence_index]
            for row_i in range(len(parser_sentence_df.index)):
                row_index = parser_sentence_df.index[row_i]
                if parser_sentence_df.loc[row_index, 'word_index'] == parser_sentence_df.loc[row_index, 'head_index']:
                    # This one is root
                    continue
                row_head_index = parser_sentence_df.index[parser_sentence_df.loc[row_index, 'head_index']]
                if pd.isnull(parser_sentence_df.loc[row_index, 'lemma']) or \
                        pd.isnull(parser_sentence_df.loc[row_head_index, 'lemma']):
                    continue
                syntactic_bi_gram_lemma_str = parser_sentence_df.loc[row_index, 'lemma'] + "_" + \
                                              parser_sentence_df.loc[row_head_index, 'lemma']
                bag_of_syntactic_n_gram_lemma.append(syntactic_bi_gram_lemma_str)

        return bag_of_syntactic_n_gram_lemma

    @staticmethod
    def get_bag_of_syntactic_n_gram_tag(parser_line_df, n_gram=2):
        """Tag represents parts of speech tag.
            Tag is fine grained whereas POS is of coarse grained.
        """
        assert n_gram == 2, "Need to extend for n_gram > 2"
        bag_of_syntactic_n_gram_tag = []
        for sentence_index in parser_line_df['sentence_index'].unique():
            parser_sentence_df = parser_line_df.loc[parser_line_df['sentence_index'] == sentence_index]
            for row_i in range(len(parser_sentence_df.index)):
                row_index = parser_sentence_df.index[row_i]
                if parser_sentence_df.loc[row_index, 'word_index'] == parser_sentence_df.loc[row_index, 'head_index']:
                    # This one is root
                    continue
                row_head_index = parser_sentence_df.index[parser_sentence_df.loc[row_index, 'head_index']]
                if pd.isnull(parser_sentence_df.loc[row_index, 'tag']) or \
                        pd.isnull(parser_sentence_df.loc[row_head_index, 'tag']):
                    continue
                syntactic_bi_gram_tag_str = parser_sentence_df.loc[row_index, 'tag'] + "_" + \
                                              parser_sentence_df.loc[row_head_index, 'tag']
                bag_of_syntactic_n_gram_tag.append(syntactic_bi_gram_tag_str)

        return bag_of_syntactic_n_gram_tag

    def get_feature_group_to_text(self, feature_groups, line_text, parser_line_df):
        # 1. extract features
        # 2. convert list of strings into space separated string
        feature_group_to_text_dict = dict()

        for feature_group in feature_groups:
            if feature_group == 'bag_of_words':
                bag_of_words = extract_bag_of_words(text=line_text)
                feature_text_bag_of_words = " ".join(bag_of_words)
                feature_group_to_text_dict[feature_group] = feature_text_bag_of_words
            elif feature_group == 'bag_of_pos':
                pos_line_arr = []
                for sentence_index in parser_line_df['sentence_index'].unique():
                    pos_line_arr.append(
                        parser_line_df.loc[parser_line_df['sentence_index'] == sentence_index, 'pos'].tolist())
                bag_of_pos = self.get_bag_of_pos(pos_line_arr=pos_line_arr)
                feature_text_bag_of_pos = " ".join(bag_of_pos)
                feature_group_to_text_dict[feature_group] = feature_text_bag_of_pos
            elif feature_group == 'bag_of_dependency_triplet':
                bag_of_dependency_triplet = self.get_bag_of_dependency_triplet(parser_line_df=parser_line_df)
                feature_text_bag_of_dependency_triplet = " ".join(bag_of_dependency_triplet)
                feature_group_to_text_dict[feature_group] = feature_text_bag_of_dependency_triplet
            elif feature_group == 'bag_of_syntactic_n_gram_lemma':
                bag_of_syntactic_n_gram_lemma = self.get_bag_of_syntactic_n_gram_lemma(parser_line_df=parser_line_df)
                feature_text_bag_of_syntactic_n_gram_lemma = " ".join(bag_of_syntactic_n_gram_lemma)
                feature_group_to_text_dict[feature_group] = feature_text_bag_of_syntactic_n_gram_lemma
            elif feature_group == "bag_of_syntactic_n_gram_tag":
                bag_of_syntactic_n_gram_tag = self.get_bag_of_syntactic_n_gram_tag(parser_line_df=parser_line_df)
                feature_text_bag_of_syntactic_n_gram_tag = " ".join(bag_of_syntactic_n_gram_tag)
                feature_group_to_text_dict[feature_group] = feature_text_bag_of_syntactic_n_gram_tag
            else:
                assert False, "Unknown feature group: {0}".format(feature_group)

        return feature_group_to_text_dict

"""
TODO:
    1. https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
        Visualization of top K features.
"""