#!/usr/bin/env python

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


class ResultAnalysis:
    def __init__(self):
        self.train_result_df = None
        self.validation_result_df = None
        self.test_result_df = None

    def load_train_result_data(self, filename):
        self.train_result_df = pd.read_csv(filepath_or_buffer=filename)

    def load_validation_result_data(self, filename):
        self.validation_result_df = pd.read_csv(filepath_or_buffer=filename)

    @staticmethod
    def plot_prob_score(df, img_file_name=None):
        target_class_arr = np.zeros(len(df.index))
        prob_target_class_arr = np.zeros(len(df.index))
        for i in range(len(df.index)):
            target_class = int(df.loc[i, "target"])
            prob_target_class = df.loc[i, "author_"+str(target_class)]
            target_class_arr[i] = target_class
            prob_target_class_arr[i] = prob_target_class

        n_class = 10
        fig = plt.figure()
        colors = cm.rainbow(np.linspace(0, 1, n_class))
        for class_i in range(n_class):
            plt.scatter(np.where(target_class_arr == class_i), prob_target_class_arr[target_class_arr == class_i], color=colors[class_i], label=class_i)

        plt.legend()
        # plt.plot(range(len(df.index)), prob_target_class_arr)

        if img_filename is not None:
            plt.savefig(img_file_name, dpi=fig.dpi)

        plt.show()

    @staticmethod
    def plot_true_vs_best_alternate_prob(df, img_file_name=None):
        target_class_arr = np.zeros(len(df.index))
        prob_target_class_arr = np.zeros(len(df.index))
        prob_best_alternate_class_arr = np.zeros(len(df.index))

        for i in range(len(df.index)):
            target_class = int(df.loc[i, "target"])
            prob_target_class = df.loc[i, "author_"+str(target_class)]
            target_class_arr[i] = target_class
            prob_target_class_arr[i] = prob_target_class

            best_alternate_class = None
            best_alternate_prob = None
            # TODO class names should be passed as input parameter
            for j in range(10):
                if j == target_class:
                    continue
                prob_cur_class = df.loc[i, "author_" + str(j)]
                if best_alternate_class is None:
                    best_alternate_class = j
                    best_alternate_prob = prob_cur_class
                elif prob_cur_class > best_alternate_prob:
                    best_alternate_class = j
                    best_alternate_prob = prob_cur_class

            prob_best_alternate_class_arr[i] = best_alternate_prob

        n_class = 10
        fig = plt.figure()
        # ax = fig.add_subplot(111)
        colors = cm.rainbow(np.linspace(0, 1, n_class))
        for class_i in range(n_class):
            plt.scatter(prob_target_class_arr[target_class_arr == class_i], prob_best_alternate_class_arr[target_class_arr == class_i],
                       color=colors[class_i], label=class_i)

        plt.xlabel('true class')
        plt.ylabel('best alternate class')
        plt.legend()
        # plt.plot(prob_target_class_arr, prob_best_alternate_class_arr, '*')

        if img_filename is not None:
            plt.savefig(img_file_name, dpi=fig.dpi)

        plt.show()

    def best_alternate_prob(self, prob_threshold, feature_groups):
        for i in range(len(self.validation_result_df.index)):
            target_class = int(self.validation_result_df.loc[i, "target"])
            prob_target_class = self.validation_result_df.loc[i, "author_" + str(target_class)]

            if prob_target_class > prob_threshold:
                continue

            # case #1: If target class has the highest probability among all the classes, then best alternate class
            #           is the one with next to highest probability.
            # case #2: Else best alternate class is the one with highest probability
            best_alternate_class = None
            best_alternate_prob = None
            # TODO class names should be passed as input parameter
            for j in range(10):
                if j == target_class:
                    continue
                prob_cur_class = self.validation_result_df.loc[i, "author_" + str(j)]
                if best_alternate_class is None:
                    best_alternate_class = j
                    best_alternate_prob = prob_cur_class
                elif prob_cur_class > best_alternate_prob:
                    best_alternate_class = j
                    best_alternate_prob = prob_cur_class

            print("Sample #{0} :: True target class: {1} : prob: {2} :: Best alternate class: {3} : prob: {4}".format(
                i, target_class, prob_target_class, best_alternate_class, best_alternate_prob))
            for feature_group in feature_groups:
                score_feature_group_target_class = self.validation_result_df.loc[i, feature_group+"_"+str(target_class)]
                score_feature_group_best_alternate_class = self.validation_result_df.loc[i, feature_group+"_"+str(best_alternate_class)]
                print("\tFeature group: {0} :: score(target class: {1}): {2} :: score(best alternate class: {3}): {4}"
                      .format(feature_group, target_class, score_feature_group_target_class, best_alternate_class, score_feature_group_best_alternate_class))

if __name__ == "__main__":
    result_analysis = ResultAnalysis()

    dataset_split_category = "validation"  # train, validation
    prob_dist_category = "prob"  # prob, dist
    hyper_parameter_detail = "C_1"

    if dataset_split_category == "train":
        result_filename = os.path.join("../output/machinehack/train/prob", dataset_split_category+"_"+
                                       prob_dist_category+"_result_"+hyper_parameter_detail+".csv")
    else:
        result_filename = os.path.join("../output/machinehack/validation",
                                       dataset_split_category + "_" + prob_dist_category + "_result_" + hyper_parameter_detail + ".csv")

    assert os.path.exists(result_filename), "result_filename: {0} doesn't exist".format(result_filename)
    img_filename = result_filename[:-3] + "png"

    if dataset_split_category == "train":
        result_analysis.load_train_result_data(filename=result_filename)
    else:
        result_analysis.load_validation_result_data(filename=result_filename)
    # result_analysis.load_train_result_data(filename=os.path.join("../output/machinehack/train/prob", "train_prob_result_C_1_by_10.csv"))
    # result_analysis.load_validation_result_data(filename=os.path.join("../output/machinehack/validation", "validation_prob_result_C_1_by_10.csv"))
    """
    result_analysis.load_validation_result_data(filename=os.path.join("../output/machinehack/validation",
                                                                      "validation_prob_bag_of_words_syntactic_n_gram_lemma_tag.csv"))
    """

    if dataset_split_category == "train":
        result_analysis.plot_prob_score(df=result_analysis.train_result_df, img_file_name=img_filename)
    else:
        result_analysis.plot_prob_score(df=result_analysis.validation_result_df, img_file_name=img_filename)

    # result_analysis.best_alternate_prob(prob_threshold=0.4, feature_groups=('bag_of_words', 'bag_of_pos'))

    img_filename = result_filename[:-4] + "_true_vs_best_alternate"+ ".png"
    if dataset_split_category == "train":
        result_analysis.plot_true_vs_best_alternate_prob(df=result_analysis.train_result_df, img_file_name=img_filename)
    else:
        result_analysis.plot_true_vs_best_alternate_prob(df=result_analysis.validation_result_df, img_file_name=img_filename)

"""
Reference:
    https://stackoverflow.com/questions/12236566/setting-different-color-for-each-series-in-scatter-plot-on-matplotlib

    https://chartio.com/resources/tutorials/how-to-save-a-plot-to-a-file-using-matplotlib/
    https://stackoverflow.com/questions/7906365/matplotlib-savefig-plots-different-from-show
        Parameter: dpi for savefig()
"""
