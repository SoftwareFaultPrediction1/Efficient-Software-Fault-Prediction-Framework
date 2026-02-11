from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
import cv2 as cv


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'RSA', 'TFMO', 'ZOA', 'POA', 'APO-AMMNet']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(2):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('--------------------------------------------------Dataset -', i + 1, 'Statistical Report ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='GTO-A-CDD-IDSANet')
        plt.plot(length, Conv_Graph[1, :], color='#aaff32', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='AOA-A-CDD-IDSANet')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='BCO-A-CDD-IDSANet')
        plt.plot(length, Conv_Graph[3, :], color='#ad03de', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='COA-A-CDD-IDSANet')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='PU-COA-A-CDD-IDSANet')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['ViT-GRU', 'LSTM', 'CNN', 'Dilated DenseNet', 'PU-COA-A-CDD-IDSANet']
    for a in range(2):  # For 5 Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')
        # Actual = np.load('Target.npy', allow_pickle=True)

        colors = cycle(["blue", "darkorange", "cornflowerblue", "deeppink", "black"])  # "aqua",
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


def plot_results_kfold():
    eval1 = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC', 'FOR', 'PT', 'CSI', "BA", 'FM', 'BM', 'MK', 'lrplus', 'lrminus', 'DOR', 'prevalence']
    Graph_Terms = [0, 4, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20]
    Algorithm = ['TERMS', 'MAO', 'TSO', 'BWO', 'CO', 'BWO+CO']
    Classifier = ['TERMS', 'LSTM', 'RNN', 'Resnet', 'HCAR-AM', 'Proposed']
    for i in range(eval1.shape[0]):
        value1 = eval1[i, 4, :, 4:]

        # Table = PrettyTable()
        # Table.add_column(Algorithm[0], Terms)
        # for j in range(len(Algorithm) - 1):
        #     Table.add_column(Algorithm[j + 1], value1[j, :])
        # print('-------------------------------------------------- Batch Size - 48-Dataset', i + 1,
        #       'Algorithm Comparison',
        #       '--------------------------------------------------')
        # print(Table)
        #
        # Table = PrettyTable()
        # Table.add_column(Classifier[0], Terms)
        # for j in range(len(Classifier) - 1):
        #     Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        # print('-------------------------------------------------- Batch Size - 48-Dataset', i + 1,
        #       'Classifier Comparison',
        #       '--------------------------------------------------')
        # print(Table)

    kfold = [100, 200, 300, 400, 500, 600]
    for i in range(eval1.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros((eval1.shape[1], eval1.shape[2]))
            for k in range(eval1.shape[1]):
                for l in range(eval1.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]

            plt.plot(kfold, Graph[:, 0], color='yellow', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                     label="GTO-A-CDD-IDSANet")
            plt.plot(kfold, Graph[:, 1], color='#aaff32', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                     label="BCO-A-CDD-IDSANet")
            plt.plot(kfold, Graph[:, 2], color='c', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                     label="AOA-A-CDD-IDSANet")
            plt.plot(kfold, Graph[:, 3], color='#ad03de', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                     label="COA-A-CDD-IDSANet")
            plt.plot(kfold, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                     label="PU-COA-A-CDD-IDSANet")
            plt.xlabel('Epoch')
            plt.xticks(kfold,('100', '200', '300', '400', '500', '600'))
            plt.ylabel(Terms[Graph_Terms[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_line.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(6)
            ax.bar(X + 0.00, Graph[:, 5], color='#ad03de', width=0.15, label="ViT-GRU")
            ax.bar(X + 0.15, Graph[:, 6], color='#aaff32', width=0.15, label="LSTM")
            ax.bar(X + 0.30, Graph[:, 7], color='c', width=0.15, label="CNN")
            ax.bar(X + 0.45, Graph[:, 8], color='yellow', width=0.15, label="Dilated DenseNet")
            ax.bar(X + 0.60, Graph[:, 9], color='k', width=0.15, label="PU-COA-A-CDD-IDSANet")
            plt.xticks(X + 0.10, ('100', '200', '300', '400', '500', '600'))
            plt.xlabel('Epoch')
            plt.ylabel(Terms[Graph_Terms[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_bar.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()


# import seaborn as sns
#
#
# def Plot_Confusion():
#     Actual = np.load('Actual.npy', allow_pickle=True)
#     Predict = np.load('Predict.npy', allow_pickle=True)
#     no_of_Dataset = 1
#     for n in range(no_of_Dataset):
#         ax = plt.subplot()
#         cm = confusion_matrix(np.asarray(Actual[0]).argmax(axis=1), np.asarray(Predict[0]).argmax(axis=1))
#         sns.heatmap(cm, annot=True, fmt='g',
#                     ax=ax)
#         path = "./Results/Confusion123_%s.png" % (n + 1)
#         plt.title('Accuracy')
#         plt.savefig(path)
#         plt.show()


from prettytable import PrettyTable
import numpy as np


def Table():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Algorithm = ['TERMS/Batch Size', 'REF1', 'REF2', 'REF3', 'REF4', 'REF5']
    Classifier = ['TERMS/Batch size', 'REF1', 'REF2', 'REF3', 'REF4', 'REF5']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Table_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    table_terms = [Terms[i] for i in Table_Terms]
    # Epoch = [50, 100, 150, 200, 250]
    Epoch = [16, 32, 48, 64, 80, 96]
    for i in range(2):
        for k in range(len(Epoch)):
            value = eval[i, :, :, 4:14]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Epoch)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, k])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Epoch)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, j+5, k])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)

            # Table = PrettyTable()
            # Table.add_column(Classifier[0], Terms[:10])
            # for j in range(len(Classifier) - 2):
            #     Table.add_column(Classifier[j + 1], value[k, len(Algorithm) + j - 1, :])
            # print('------------------------------- Dataset- ', i+1, '-KFOLD - ', Epoch[k], '  - State of the art Classifier Comparison',
            #       '---------------------------------------')
            # print(Table)

            # Table = PrettyTable()
            # Table.add_column(Classifier[0], Terms[:10])
            # for j in range(len(Classifier) - 2):
            #     Table.add_column(Classifier[j + 1], value[k,  j, :])
            # print('------------------------------- Dataset- ', i+1, '-KFOLD - ', Epoch[k], '  -State of the art Algorithm Comparison',
            #       '---------------------------------------')
            # print(Table)



if __name__ == '__main__':
    plot_results_kfold()
    Plot_ROC_Curve()
    plotConvResults()
    Table()
    # Plot_Confusion()
    # Sample_images()
    # Image_Results1()
    # plot_results()
    # plot_results_kfold1()
