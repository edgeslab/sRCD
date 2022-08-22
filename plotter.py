import os
import sys
import pdb
import argparse
import collections

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


xmap = {
    'num_entities'          :   'Number of entities',
    'num_dependencies'      :   'Number of dependencies',
    'num_feedback_loops'    :   'Number of feedback loops'
}

ymap = {
    's_precision'   :   'Skeleton Precision',
    's_recall'      :   'Skeleton Recall',
    'o_precision'   :   'Orientation Precision',
    'o_recall'      :   'Orientation Recall',
    'p_precision'   :   'isPossibleParent Precision',
    'p_recall'      :   'isPossibleParent Recall',
    'a_precision'   :   'isPossibleAncestor Precision',
    'a_recall'      :   'isPossibleAncestor Recall',
    'f_precision'   :   'isPossibleCycle Precision',
    'f_recall'      :   'isPossibleCycle Recall',
    'dummy_dummy'   :  'dummy'
}

lmap = {
    'RCD'       :   'RCD',
    'sRCD'      :   'cRCD',
    'd-RCD'     :   'd-RCD',
    'sigma-RCD' :   'sigma-RCD',
    'sigma-sRCD':   'sigma-sRCD',
    's-RCD'     :   'sigma-RCD',
    's-RCD-pr'  :   'sigma-RCD-PR',
    's-RCD-nr'  :   'sigma-RCD-NR',
    'RelFCI'    :   'RelFCI'
}


def plot_init(fsize, xlabel, ylabel):
    fig = plt.figure(figsize=(16,10))
    plt.rc('legend', fontsize=fsize)
    plt.rc('xtick',labelsize=fsize)
    plt.rc('ytick',labelsize=fsize)
    plt.rcParams["font.family"] = "Times New Roman"

    plt.xlabel(xlabel, fontsize=fsize+5)
    plt.ylabel(ylabel, fontsize=fsize+5)

    return fig


def draw_multi_y_column(df, num_plots, labels, xlabel, ylabel, filename, fmt='eps', fontsize=40, shadow_df=None):
    columns = list(df.columns)
    
    xcol = columns[0]
    ycols = columns[1:]

    fig = plot_init(fsize=fontsize, xlabel=xlabel, ylabel=ylabel)
    
    legend_handles = []
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ["o", "^", "s", "P", "D", ">"]
    # colors = ['blue', 'green', 'gold', 'red', 'purple', 'magenta']
    colors = ['blue', 'green', 'purple', 'red', 'gold', 'magenta']
    ls = 0
    for i in range(num_plots):
        # df[xcols[i]] = df[xcols[i]] * 60
        line, = plt.plot(xcol, ycols[i], data=df, linewidth=3, linestyle=linestyles[ls], color=colors[ls], marker=markers[ls], markersize=16)
        legend_handles.append(line)

        if shadow_df is not None:
            line, = plt.plot(xcol, ycols[i].replace('ii', 'i'), data=shadow_df, linewidth=3, linestyle='dashed', color=colors[ls], marker=markers[ls], markersize=16)
            legend_handles.append(line)

        ls += 1

    axes = plt.gca()
    legend_loc = 'upper right'
    
    axes.set_xticks(df[xcol])
    # if 'Precision' in ylabel:
    #     axes.set_ylim([0.55, 1.05])
    #     axes.set_yticks([0.60, 0.70, 0.80, 0.90, 1.0])
    # else:
    axes.set_ylim([0.25, 1.05])
    axes.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # plt.legend(handles=legend_handles, labels=labels, loc=legend_loc, prop={'size': 32}, ncol=2)
    if 'time' in filename:
        plt.legend(handles=legend_handles, labels=labels, prop={'size': fontsize-10}, ncol=1, loc='upper left', fancybox=True, framealpha=0.5)
    elif 'ltm' in filename:
        # pltlegend = plt.legend(handles=legend_handles, bbox_to_anchor=(0.50, 1.11), labels=labels, prop={'size': fontsize-5}, ncol=4, loc='upper center')
        pltlegend = plt.legend(handles=legend_handles, labels=labels, prop={'size': fontsize-10}, ncol=2, loc='upper right')

    if fmt == 'eps':
        plt.savefig(filename, format='eps', dpi=2000, bbox_inches='tight')
    else:
        print(fmt, filename)
        plt.savefig(filename, format=fmt, bbox_inches='tight')

    if ('time' not in filename) and ('ltm' not in filename):
        # figlegend = plt.figure(figsize=(58, 3.5))
        figlegend = plt.figure(figsize=(24, 3.5))
        # plt.rc('legend', fontsize=84)
        # figlegend.legend(handles=legend_handles, labels=labels, bbox_to_anchor=(1.0, 1.0), prop={'size': 80}, ncol=4, loc='upper right')
        figlegend.legend(handles=legend_handles, labels=labels, bbox_to_anchor=(1.0, 1.0), fontsize=100, ncol=4, loc='upper right')
        # figlegend.savefig('plots/fig_1_2_legend.eps', dpi=3000, format='eps')
        figlegend.savefig('plots/fig_1_2_legend.png', format='png')


def plot_stacked_bar(result_file):
    df = pd.read_csv(result_file)
    pdb.set_trace()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-res', default='15_pass_eta1_combinedw_arrr.csv', help='combined result file.')
    parser.add_argument('-sres', default='', help='shadow result file.')
    parser.add_argument('-fmt', default='eps', help='image format.')
    parser.add_argument('--t1', action='store_true', help='put dashed type-i error line.')
    # parser.add_argument('-xlabel', default='Dependence coefficient', help='x-label for plot.')
    # parser.add_argument('-out', default='', help='output image filename.')
    parser.add_argument('--all', action='store_true', help='generate plots for all out files.')
    parser.add_argument('--sb', action='store_true', help='generate stacked bar plot.')
    args = parser.parse_args()

    def draw_plot(result_file, shadow_file=''):
        results = pd.read_csv(result_file)
        columns = list(results.columns)

        xlabel = xmap[columns[0]]
        for k in ymap:
            if result_file.split('.')[0].endswith(k):
                error_type = k
                break
            # error_type = '_'.join(result_file.split('.')[0].split('_')[-2:])
        ylabel = ymap[error_type]

        out_file = 'plots/' + result_file.split('/')[1].split('.')[0] + '.' + args.fmt
        labels = list(map(lambda x: lmap[x.split('_')[0]], columns[1:]))

        shadow_result = None
        if shadow_file != '':
            shadow_result = pd.read_csv(shadow_file)
            if result_file.split('.')[0].split('_')[-1] == 'times':
                labels = []
                for c in columns[1:]:
                    labels.append(lmap[c.split('_')[0]] + '-Case 2')
                    labels.append(lmap[c.split('_')[0]] + '-Case 0')

            if result_file.split('.')[0].split('_')[-1] == 'ii':
                # shadow_result = pd.read_csv(result_file.replace('_ii.', '_i.'))
                labels = []
                for c in columns[1:]:
                    labels.append(lmap[c.split('_')[0]] + '-Type II')
                    labels.append(lmap[c.split('_')[0]] + '-Type I')
                ylabel = 'Type-I/II Error'


        draw_multi_y_column(results, results.shape[1]-1, labels, xlabel, ylabel, out_file, fmt=args.fmt, shadow_df=shadow_result)


    if args.sb:
        plot_stacked_bar(args.res)
    elif args.all:
        if not os.path.isdir(args.res):
            print("ERROR: -res is not a directory!")
            sys.exit(1)
        for path, _, files in os.walk(args.res):
            for file in files:
                result_file = os.path.join(path, file)
                if result_file.split('.')[-1] != 'csv':
                    continue
                # if 'type' not in result_file:   # TODO: handle times plot
                #     continue
                draw_plot(result_file)
            break
    else:
        if not os.path.isfile(args.res):
            print("ERROR: -res is not a file!")
            sys.exit(1)
        draw_plot(args.res, args.sres)


if __name__ == "__main__":
    main()