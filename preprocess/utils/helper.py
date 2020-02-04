import numpy as np
import matplotlib.pyplot as plt

class Helper(object):
    """
      File Utilities for read and write file
    """

    @staticmethod
    def count_ucategory(np_array):
        """
            efficient frequency counts for unique values in an array
        """

        try:
            unique, counts = np.unique(np_array, return_counts=True)
            # print np.asarray((unique, counts)).T
            return unique, counts
        except IOError as error:
            print(error)

    @staticmethod
    def rename_df_columnname(df, new_column_names):
        """ Rename dataframe column name
        """
        colunms = df.columns
        _new_columns = {}
        for _col_name in colunms:
            if _col_name in new_column_names:
                _new_columns[_col_name] = new_column_names[_col_name]

        # Rename column names
        df.rename(columns=_new_columns, inplace=True)

        # Return Dataframe with new columns'name
        return df


    @staticmethod
    def plot_hist(X_value, Y_value):
        """ Plot Histogram
        """

        np.random.seed(100)
        np_hist = np.random.normal(loc=0, scale=1, size=1000)
        
        hist,bin_edges = np.histogram(np_hist)

        plt.figure(figsize=[10,8])

        plt.bar(bin_edges[:-1], hist, width = 0.5, color='#0504aa',alpha=0.7)
        plt.xlim(min(bin_edges), max(bin_edges))
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value',fontsize=15)
        plt.ylabel('Frequency',fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel('Frequency',fontsize=15)
        plt.title('Normal Distribution Histogram',fontsize=15)
        plt.show()

        # # the histogram of the data
        # n, bins, patches = plt.hist(X_value, 50, density=True, facecolor='g', alpha=0.75)

        # plt.xlabel('Smarts')
        # plt.ylabel('Probability')
        # plt.title('Histogram of IQ')
        # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        # plt.xlim(40, 160)
        # plt.ylim(Y_value)
        # plt.grid(True)
        plt.show()

    @staticmethod
    def plot_histogram(X_value, Y_value, rotation=None):
        import pandas as pd
        df = pd.DataFrame({'lab':X_value, 'val':Y_value})
        # ax = df.plot.bar(x='lab', y='val', rot=0, color=['r','b', 'g'])
        ax = df.plot(kind='line', x='lab', y='val', color=['r'])
        #ax = df.plot(kind='bar', color=['r','b'])
        
        # Rotate label by 80 downward
        if not rotation:
            rotation = rotation
        else:
            rotation = -45
            

        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)

        # Annotate the bar
        for p in ax.patches:
            #ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
            ax.annotate(np.round(p.get_height(),decimals=2), \
                (p.get_x()+p.get_width()/2., p.get_height()), \
                    ha='center', va='center', xytext=(0, 10), \
                        textcoords='offset points')

        plt.show()

    @staticmethod
    def plot_graphline(X_value, Y_value, rotation=None):
        import pandas as pd
        df = pd.DataFrame({'Number of Users':X_value, 'Number of feedback':Y_value})
        # ax = df.plot.bar(x='lab', y='val', rot=0, color=['r','b', 'g'])
        _title = 'Number of Feedback per User'
        # ax = df.plot(kind='line', x='Number of users' , y='Number of feedback', color=['r'], title=_title)
        ax = df.plot(kind='line', x='Number of Users' , y='Number of feedback', color=['r'], title=_title)
        
        # Rotate label by 80 downward
        if not rotation:
            rotation = rotation
        else:
            rotation = -45
            

        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)

        # zip joins x and y coordinates in pairs
        for x,y in zip(X_value, Y_value):

            label = "{:}".format(y)

            ax.annotate(label, # this is the text
                        (x,y), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='center') # horizontal alignment can be left, right or center
        plt.show()

    @staticmethod
    def plot_heatmap(x, y, y_labels):
        
        import seaborn as sns; sns.set()

        # y_axis_labels = y_labels
        # ax = sns.heatmap([x, y], yticklabels=y_axis_labels, linewidths=.5,  annot=True, square=True,cmap='RdBu_r')
        # ax = sns.heatmap([x, y], linewidths=.5,  annot=True, square=True,cmap='RdBu_r')
        ax = sns.heatmap([x, y], linewidths=.5,  annot=True, cmap='RdYlBu')
        ax.set_title("User's Feedback Mos VS Predicted Mos: YouTube General Service", fontsize=12, fontdict={})
        #plt.yticks(np.arange(7)+0.5,y_labels, rotation=45, fontsize="10", va="center")
        plt.yticks(np.arange(1.5) + 0.5,y_labels, rotation=45, fontsize="10", va="center")

        plt.show()

        # flights = sns.load_dataset("flights")
        # flights = flights.pivot("month", "year", "passengers")
        # print(flights)
        # ax = sns.heatmap(flights)
    
    @staticmethod
    def plot_heatmap2(df):
        """
        """
        import seaborn as sns; sns.set()
        a4_dims = (12, 9)
        
        # values = "bad_status"
        # index = np.arange(1,6)
        # columns = np.arrang(1, 6)
        # vmax = 0.10
        # cellsize_vmax = 10000
        # g_ratio = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
        # g_size = df.pivot_table(index=index, columns=columns, values=values, aggfunc="size")
        # annot = np.vectorize(lambda x: "" if np.isnan(x) else "{:.1f}%".format(x * 100))(g_ratio)
    
        # # adjust visual balance
        # figsize = (g_ratio.shape[1]  * 0.8, g_ratio.shape[0]  * 0.8)
        # cbar_width = 0.05 * 6.0 / figsize[0] 
    
        # f, ax = plt.subplots(1, 1, figsize=figsize)
        # cbar_ax = f.add_axes([.91, 0.1, cbar_width, 0.8])
        # heatmap2(g_ratio, ax=ax, cbar_ax=cbar_ax, 
        #      vmax=vmax, cmap="PuRd", annot=annot, fmt="s", annot_kws={"fontsize":"small"},
        #      cellsize=g_size, cellsize_vmax=cellsize_vmax,
        #      square=True, ax_kws={"title": "{} x {}".format(index, columns)})

        #fig, ax = plt.subplots(figsize=a4_dims)
        #sns.heatmap(df, linewidths=.2,  annot=True, cmap='RdYlBu', ax=ax)
        #sns.heatmap(df, linewidths=1,  annot=True, cmap='Blues', ax=ax, square=True)
        #grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
        #f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
        #ax = sns.heatmap(df, ax=ax, cbar_ax=cbar_ax, cmap='Blues', cbar_kws={"orientation": "vertical"}, annot=True, vmin=0, vmax=1.0)
        ax = sns.heatmap(df, cmap='Blues', cbar_kws={"orientation": "vertical"}, annot=True, vmin=0, vmax=1.0, center= 0, fmt='.1g')

        ax.set_xlabel('')
        ax.set_ylabel('')

        # for text in ax.texts:
        #     text.set_size(12)
        #     if text.get_text() == '118':
        #         text.set_size(18)
        #         text.set_weight('bold')
        #         text.set_style('italic')

        # sns.set(rc={'figure.figsize':(20.7,15.27)})
        plt.show()

    @staticmethod
    def plot3(df, index, columns):

        import seaborn as sns; sns.set()
        values='Val %'
        vmax = 0.10
        cellsize_vmax = 10000
        g_ratio = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
        g_size = df.pivot_table(index=index, columns=columns, values=values, aggfunc="size")
        annot = np.vectorize(lambda x: "" if np.isnan(x) else "{:.1f}%".format(x * 100))(g_ratio)
        
        # adjust visual balance
        figsize = (g_ratio.shape[1]  * 0.8, g_ratio.shape[0]  * 0.8)
        cbar_width = 0.05 * 6.0 / figsize[0] 
        
        f, ax = plt.subplots(1, 1, figsize=figsize)
        cbar_ax = f.add_axes([.91, 0.1, cbar_width, 0.8])
        ax = sns.heatmap(g_ratio, ax=ax, cbar_ax=cbar_ax, 
                vmax=vmax, cmap="PuRd", annot=annot, fmt="s", annot_kws={"fontsize":"small"},
                cellsize=g_size, cellsize_vmax=cellsize_vmax,
                square=True, ax_kws={"title": "{} x {}".format(index, columns)})
        plt.show()

    @staticmethod
    def split(word):
        """ # Python3 program to Split string into characters 
        """

        return [char for char in word]

    @staticmethod
    def is_substring(subsstr, word):
        _sstr_arr = Helper.split(subsstr)
        _str_arr =  Helper.split(word)

        for i, char in enumerate(_sstr_arr):
            found_sub = False
            try:
                _idx = -1
                _idx = _str_arr.index(char)

                if (_idx == 0 and i==0):
                    found_sub = True
                elif (_idx >= 0):
                    if (i==0):
                        found_sub = False
                        # stop finding
                        break
                    else: found_sub = True
                else: found_sub = False

            except ValueError:
                found_sub = False
                # print("Char %s is not found in %s", char, word)

        return found_sub





