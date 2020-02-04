"""
    Preprocessing
"""

import time, math

import pandas as pd
import numpy as np

from model.user_feedback import UserFeedback
from algorithms.decisiontree.decisiontree_youtubeauto import DecisionTreeYoutubeAuto

from utils.file_helper import FileHelper
from utils.helper import Helper

class UserFbViewModel:
    """
        UserFbViewModel
    """

    def __init__(self, **kwargs):
        self.config = kwargs

        # Read User Feedback from csv
        self.user_feedback = UserFeedback(**self.config)

    def __to_nan(self, X):
        """ Check the value in X 
            if it match the errors, then replace it by NaN of Pandas.

            Parameters
            ----------
                X : value of any data type
            
            Returns
            ----------
                - NaN if it is an error value
                - X otherwise
        """
        # Error values to be identifier
        error_values = ['nan', '+Inf','#NULL!', 'inf', 'Inf']
        try:     
            if len(str(X))==0:
                return np.nan
            elif str(X) in error_values:
                return np.nan
            else: return X
        except ValueError:
            return np.nan

    def dataset_propre(self):
        """ This fuction create new dataset from 
        the original dataset
        """

        # Read all feedbacks from table user's feedback
        criteria = {}
        df_userfb = self.user_feedback.get_user_feedback(criteria)

        # Columns to map from original dataset to new dataset
        cols_name = ['FEEDBACK_APP', 'FEEDBACK_VALUE', 'USER_ID', 'RTT', \
                    'DOWNLOAD_LOSS_RATE', 'UPLOAD_LOSS_RATE', 'DOWNLOAD_JITTER', \
                    'UPLOAD_JITTER', 'UDP_DOWNLOAD_THROUGHPUT', 'UDP_UPLOAD_THROUGHPUT', \
                    'SKYPE___VOICE_CALL___MOS', 'YOUTUBE___VIDEO_PLAYBACK___P_720___MOS']

        # iterating the columns
        new_dataset = pd.DataFrame(columns=cols_name)
        for col in df_userfb.columns:
            new_dataset[col] = df_userfb.loc[:, col].apply(lambda x: self.__to_nan(x))
        
        # Drop NaN
        new_dataset = new_dataset.dropna()

        # Cols to remove
        del_cols = set(df_userfb.columns) - set(cols_name)
        new_dataset = new_dataset.drop(list(del_cols), axis=1)

        # Create empty DataFrame
        #new_dataset = pd.DataFrame(columns=cols_name)
        #for colname in cols_name:
            #new_dataset[colname] = df_userfb.loc[:, colname].apply(lambda x: self.__to_nan(x))

        # Save Dataframe to csv file
        filename = self.config['OUT_DIR'] + '2K_dataset.csv'
        # _groupby_df.transpose().to_csv(filename)
        new_dataset.to_csv(filename)

        return new_dataset

    def get_2k_userfeedback(self):
        # Read all feedbacks from table user's feedback
        criteria = {}
        df_2k_userfb = self.user_feedback.get_2k_userfeedback(criteria)

        return df_2k_userfb

    
    def count_userfb_bygroup(self):
        """
            Count user feedback by Group of 1 to 5
            Feedback MoS: [1..5]
            Estimated MoS: [1..5]
        """

        ### Read all feedbacks from table user's feedback
        criteria = {}
        df_userfb = self.user_feedback.get_user_feedback(criteria)

        # Filter by Feedback_App name; Feedback_App= YOUTUBE_GENERAL
        try:
            # Filter by YOUTUBE___GENERAL___MOS
            mask = df_userfb[self.config['COL']['FB_APP']] == self.config['COL']['YU_G']
            app_yu_g = df_userfb[mask]

            # Select 2 Columns: FB_APP and YU_VP_720
            # app_yu_g = app_yu_g[[self.config['COL']['FB_DATE'], self.config['COL']['FB_VAL'], self.config['COL']['YU_VP_720']]]
            app_yu_g = app_yu_g[[self.config['COL']['FB_VAL'], self.config['COL']['YU_VP_720']]]

            ## Remove Nan or Zero from records
            app_yu_g[self.config['COL']['YU_VP_720']] = app_yu_g[self.config['COL']['YU_VP_720']].apply(lambda x: 0 if len(str(x))==0 or str(x)=='nan' else x)
            
            ## Recording the inconsistent instances index 
            dropIx = app_yu_g[app_yu_g[self.config['COL']['YU_VP_720']]==0].index
            ## Dropping these instances from the dataset:
            app_yu_g.drop(dropIx, inplace=True)

            #df = pd.pivot_table(app_yu_g, index=self.config['COL']['FB_VAL'], columns=self.config['COL']['YU_VP_720'], values=self.config['COL']['YU_VP_720'], aggfunc=np.sum)
            # df = pd.pivot_table(app_yu_g, index=self.config['COL']['FB_VAL'], columns='idx', values=self.config['COL']['YU_VP_720'], aggfunc=lambda x: len(x.unique()))

            # Group Records by estimated MoS: (1, 5)
            # grouped_fbs_by_user = df_userfbs.groupby([self.config['GROUP_BY']['USER_ID']]).count().reset_index()
            df = app_yu_g.groupby([self.config['COL']['FB_VAL']]).count().reset_index()
            df['idx'] = df[self.config['COL']['FB_VAL']]

            # Save for testing purpose
            filename = self.config['OUT_DIR'] + 'groupby_user_cat_3.csv'
            # _groupby_df.transpose().to_csv(filename)
            df.to_csv(filename)

            # Get List of MoS from Config
            mos_list = self.config['CONST']['MOS']

            # Create dataframe of columns: [1, 2, 3, 4, 5]
            
            # Dataframe
            df = pd.DataFrame()
            # iterate over MOS list one by one from 1 to 5
            for mos_val in mos_list:
                # Filter by YOUTUBE___GENERAL___MOS
                _mask = app_yu_g[self.config['COL']['FB_VAL']] == mos_val
                _df = app_yu_g[_mask]

                # Group Records by estimated MoS: (1, 5)
                # grouped_fbs_by_user = df_userfbs.groupby([self.config['GROUP_BY']['USER_ID']]).count().reset_index()
                _groupby_df = _df.groupby([self.config['COL']['YU_VP_720']]).count().reset_index()

                # Reshape row to column 
                _sum = (_groupby_df[self.config['COL']['FB_VAL']]).values.reshape(1,-1)
                _sum = np.around(_sum/np.sum(_sum), decimals=2)
                _df = pd.DataFrame(_sum, index=[mos_val], columns = _groupby_df[self.config['COL']['YU_VP_720']])

                #
                if df.empty:
                    df = pd.DataFrame(_df)
                else:
                    frames = [df, _df]
                    df = pd.concat(frames)
                    
            # Save for testing purpose
            filename = self.config['OUT_DIR'] + 'fb_user_cat_2.csv'
            # _groupby_df.transpose().to_csv(filename)
            app_yu_g.to_csv(filename)
            # Create the pandas DataFrame

            # import seaborn as sns; sns.set()
            # flights_long = sns.load_dataset("flights")
            # # Save for testing purpose
            # filename2 = self.config['OUT_DIR'] + 'flight_user_cat.csv'
            # # _groupby_df.transpose().to_csv(filename)
            # flights_long.to_csv(filename2)

            # Draw heatmap
            #Helper.plot_heatmap2(df)

        except IOError as error:
            print(error)

    def __to_number(self, X):
        """ Convert string to float
        """

        exclude_values = ['nan', '+Inf', '#NULL!', 'inf', 'Inf']
        
        try:
            if len(str(X))==0 or (str(X) in exclude_values):
                return -1
            else: return float(X)
        except ValueError:
            return 0

    def get_all_userfeedbacks(self):
        """
            Get all users'feedbacks from table user's feedback
        """

        # Read User Feedback from csv
        user_feedback = UserFeedback(**self.config)
        ### Read all feedbacks from table user's feedback
        criteria = {}
        df_userfb = user_feedback.get_user_feedback(criteria)
        df_userfb = df_userfb.dropna()

        # Filter by Feedback_App name; Feedback_App= YOUTUBE_GENERAL
        try:

            ## Remove from column RTT: the values Blank space, NaN, Zero, #NULL!, +Inf
            df_userfb[self.config['COL']['RTT']] = df_userfb[self.config['COL']['RTT']].apply(lambda x: self.__to_number(x) )
            
            ## Recording the inconsistent instances index 
            dropIx = df_userfb[df_userfb[self.config['COL']['RTT']]==-1].index
            ## Dropping these instances from the dataset:
            df_userfb.drop(dropIx, inplace=True)
            df_userfb.reset_index(inplace=True)

            # Rename columns to match the columns name in model
            df_userfb = Helper.rename_df_columnname(df_userfb, self.config['COL'])

            # Save for testing purpose
            filename = self.config['OUT_DIR'] + 'all_ufb_except_unknown.csv'
            df_userfb.to_csv(filename)

        except IOError as error:
            print(error)

        return df_userfb
    
    def normalize_string(self, datafarame, feature_names):
        """ Replace decimal separator , anglophone decimal separator .
        """
        for name in feature_names:
            datafarame[name] = datafarame[name].apply(lambda x: float(x.split()[0].replace(',', '.')))
        return datafarame

    def dataset_youtubeauto(self):
        """ Generate dataset for Youtube Auto decision tree model
        """
   
        # Features for decision tree model
        #feature_names = ['RTT','UTH','DTH','DL','UL','DJ','UJ']
        # Columns to map from original dataset to new dataset
        feature_names = ['RTT', 'DOWNLOAD_LOSS_RATE', 'UPLOAD_LOSS_RATE', 'DOWNLOAD_JITTER', \
                    'UPLOAD_JITTER', 'UDP_DOWNLOAD_THROUGHPUT', 'UDP_UPLOAD_THROUGHPUT']

        # Load users'feedbacks from CSV
        df_userfbs = self.get_2k_userfeedback()

        # Test set
        X_test = []

        # Get all column names
        colunms = list(df_userfbs.columns.values)
        if all(_fname in colunms for _fname in feature_names):
            try:
                for idx, row in df_userfbs.iterrows():
                    # Convert metric units from ACQUA to YoutubeAuto
                    # # Nanosec to Millisec
                    # rtt_auto = (row['RTT']) / 1000000
                    # # Microsec to Millisec
                    # dj_auto = (row['DJ']) / 1000000000
                    # uj_auto = (row['UJ']) / 1000000000
                    # # bps to Kbps
                    # dth_auto = (row['DTH']) / 1024 
                    # uth_auto = (row['UTH']) / 1024
                    # # Nanosec to Millisec
                    rtt_auto = (row['RTT'])
                    # Microsec to Millisec
                    dj_auto = (row['DOWNLOAD_JITTER'])
                    uj_auto = (row['UPLOAD_JITTER'])
                    # bps to Kbps
                    dth_auto = (row['UDP_DOWNLOAD_THROUGHPUT'])
                    uth_auto = (row['UDP_UPLOAD_THROUGHPUT'])

                    # Loss
                    ul_auto = (row['UPLOAD_LOSS_RATE'])
                    dl_auto = (row['DOWNLOAD_LOSS_RATE'])

                    # Check list of keys in dict
                    # access data using column names, cast to float 
                    # (bydefault its strings and change ',' to '. for all records')
                    feature ={'idx':idx, 'DTH':dth_auto,'RTT':rtt_auto , 'DJ':dj_auto, \
                    'DL': dl_auto, 'UJ':uj_auto, 'UL':ul_auto , 'UTH':uth_auto }

                    # Add record to list
                    X_test.append(feature)

            except EOFError as error:
                return error

        # Save for testing purpose
        filename = self.config['OUT_DIR'] + 'DATASET_EST_QOE_BYYOUTUBEAUTOMODEL.csv'
        df_userfbs.to_csv(filename)

        return X_test, df_userfbs

    def predict_youtubeauto(self):
        """
            Make predictiion for youtube auto
        """

        # Initialize decision tree
        dt = DecisionTreeYoutubeAuto(**self.config)

        # load json from file (feature data)
        filename = "youtube.dash.json"
        file_uri = FileHelper.dataset_path(self.config, filename)
        json_model = FileHelper.load_model_json(file_uri)

        # Generate X_test dataset
        X_test, df_userfbs = self.dataset_youtubeauto()

        # # Test with Mockup data
        # import random

        # dj_max = 124326180.9
        # uj_max = 128804961.5
        # ul_max = 1
        # dl_max = 1
        # uth_max = 29335149.8038362
        # dth_max = 18033919.2661197
        # rtt_max = 1000 #14226774236.96

        # X_test = []
        # number_X_test = 5
        # for i in np.arange(number_X_test):
        #     features_1 = None
        #     features_1 = {'idx': i,'RTT': random.uniform(0,rtt_max), 'DJ': random.uniform(0,dj_max), 'UJ':random.uniform(0, uj_max), 'DL': random.uniform(0,dl_max), 'UL': random.uniform(0, ul_max), 'DTH': random.uniform(0,dth_max), 'UTH': random.uniform(0,uth_max)}
            
        #     for k, v in features_1.items():
        #         if not k == 'idx':
        #             features_1[k] = round(v, 2)

        #     features_1['Userfeedback'] = 0
        #     features_1['Youtube_720P'] = 0

        #     # features_2 = {'DTH': randint(30000,4000000) ,'RTT': randint(500,300000), 'DJ': randint(0,1000), 'DL': randint(0,1000), 'UJ':randint(0,1000), 'UL': rand(), 'UTH': randint(1000,200000)}
        #     # features_3 = {'DTH': randint(1000000,4000000) ,'RTT': randint(1000,240000), 'DJ': randint(0,1000), 'DL': randint(0,1000), 'UJ':randint(0,1000), 'UL': rand(), 'UTH': randint(1000000,4000000)}
        #     X_test.append(features_1)

        # features_1 = {'idx': number_X_test,'RTT': 172211718, 'DL': 0.25, 'UL': 0.0, 'DJ': 1462940.373, 'UJ':967358.4, \
        #     'DTH': 6904033.241, 'UTH': 6688152.991, 'Userfeedback': 1, 'Youtube_720P': 1}
        # features_2 = {'idx': number_X_test+1,'RTT': 329342189.4, 'DL': 0.0, 'UL': 0.0, 'DJ': 2217979.68, 'UJ':4026196.84, \
        #     'DTH': 206766.1582, 'UTH':1152765.337, 'Userfeedback': 1, 'Youtube_720P': 1}
        # features_3 = {'idx': number_X_test+2,'RTT': 14226774237, 'DL': 0.0, 'UL': 0.0, 'DJ': 78439389.08, 'UJ':124548859.2, \
        # 'DTH': 41369.68321, 'UTH':34202.56526, 'Userfeedback': 1, 'Youtube_720P': 1}

        # features_4 = {'idx': number_X_test+3,'RTT': 14226774237, 'DL': 0.0, 'UL': 0.0, 'DJ': 78439389.08, 'UJ':124548859.2, \
        # 'DTH': 41369.68321, 'UTH':34202.56526, 'Userfeedback': 1, 'Youtube_720P': 1}

        # features_5 = {'idx': number_X_test+4, 'RTT': 3303547790, 'DL': 0.0, 'UL': 0.0, 'DJ': 11294340.32, 'UJ': 27797712.96, \
        # 'DTH': 110344.9385, 'UTH': 32443.88377, 'Userfeedback': 2, 'Youtube_720P': 2}

        # X_test.append(features_1)
        # X_test.append(features_2)
        # X_test.append(features_3)
        # X_test.append(features_4)
        # X_test.append(features_5)
        
        # Transform from List to Dataframe
        df_userfbs = pd.DataFrame(X_test)

        # Start our prediciton
        estimated_mos = dt.predict(json_model, X_test)

        # Create dataframe from estimated_mos
        # Save for testing purpose
        df_estimated_qoe = pd.DataFrame(estimated_mos)
        filename = self.config['OUT_DIR'] + 'Estimated_MOS_YoutubeAuto_New.csv'
        df_estimated_qoe.to_csv(filename)
        
        # Merge both X_test and estimated_QoE 
        # Create dataframe from estimated_mos
        # Save for testing purpose
        df_merged = pd.merge(df_userfbs, df_estimated_qoe, left_index=True, right_index=True)
        df_merged = df_merged.drop(['idx_x' , 'idx_y'] , axis='columns')
        filename = self.config['OUT_DIR'] + 'DATASET_MOSUSERFEEDBACK_QOE_YOUTUBEAUTO.csv'
        df_merged.to_csv(filename)

        # Print the results
        dt.print(estimated_mos)
        return estimated_mos

    def count_userfb_bygroup2(self):
        """
            Count user feedback by Group of 1 to 5
            Feedback MoS: [1..5]
            Estimated MoS: [1..5]
        """

        # Read User Feedback from csv
        user_feedback = UserFeedback(**self.config)
        ### Read admissions groupby date and
        criteria = {}
        df_userfb = user_feedback.get_user_feedback(criteria)

        # Filter by Feedback_App name; Feedback_App= YOUTUBE_GENERAL
        try:
            # Filter by YOUTUBE___GENERAL___MOS
            mask = df_userfb[self.config['COL']['FB_APP']] == self.config['COL']['YU_G']
            app_yu_g = df_userfb[mask]

            # Select 2 Columns: FB_APP and YU_VP_720
            # app_yu_g = app_yu_g[[self.config['COL']['FB_DATE'], self.config['COL']['FB_VAL'], self.config['COL']['YU_VP_720']]]
            ## Remove Nan or Zero from records
            app_yu_g[self.config['COL']['YU_VP_720']] = app_yu_g[self.config['COL']['YU_VP_720']].apply(lambda x: 0 if len(str(x))==0 or str(x)=='nan' else x)
            
            ## Recording the inconsistent instances index 
            dropIx = app_yu_g[app_yu_g[self.config['COL']['YU_VP_720']]==0].index
            ## Dropping these instances from the dataset:
            app_yu_g.drop(dropIx, inplace=True)

            # Assign data type as Int to column
            # app_yu_g[self.config['COL']['YU_VP_720']] = app_yu_g.astype({self.config['COL']['YU_VP_720']: 'int32'}).dtypes

            # Get List of MoS from Config
            mos_list = self.config['CONST']['MOS']

            # Create dataframe of columns: [1, 2, 3, 4, 5]
            
            # Dataframe
            df = pd.DataFrame()
            # iterate over MOS list one by one from 1 to 5
            for mos_val in mos_list:
                # Filter by YOUTUBE___GENERAL___MOS
                _mask = app_yu_g[self.config['COL']['FB_VAL']] == mos_val
                _df = app_yu_g[_mask]

                # Group Records by estimated MoS: (1, 5)
                # grouped_fbs_by_user = df_userfbs.groupby([self.config['GROUP_BY']['USER_ID']]).count().reset_index()
                _groupby_df = _df.groupby([self.config['COL']['YU_VP_720']]).count().reset_index()

                # Reshape row to column 
                #_sum = (_groupby_df[self.config['COL']['FB_VAL']]).values.reshape(1,-1)
                _sum = _groupby_df[self.config['COL']['FB_VAL']]
                # _sum = np.around(_sum/np.sum(_sum), decimals=2)
                _groupby_df['Val %'] = np.around(_sum/np.sum(_sum), decimals=2)
                _groupby_df['Feedback_MoS'] = ['m_' + str(mos_val) for i in range(5)]

                # Save for testing purpose
                filename = self.config['OUT_DIR'] + 'fb_user_cat_2.csv'
                # _groupby_df.transpose().to_csv(filename)
                _groupby_df.to_csv(filename)
                
                #_df = pd.DataFrame(_sum, index=[mos_val], columns = _groupby_df[self.config['COL']['YU_VP_720']])
                # _df = pd.DataFrame(_sum, index=[mos_val for i in range(5)], columns = _groupby_df[self.config['COL']['YU_VP_720']])
                #_df = pd.DataFrame(_sum, index=[mos_val for i in range(5)], columns = ['A'])
                #
                if df.empty:
                    df = pd.DataFrame(_groupby_df)
                else:
                    frames = [df, _groupby_df]
                    df = pd.concat(frames)

            # Save for testing purpose
            filename = self.config['OUT_DIR'] + 'fb_user_cat_1.csv'
            # _groupby_df.transpose().to_csv(filename)
            df.to_csv(filename)
            # Create the pandas DataFrame
            index='Feedback_MoS'
            columns='YOUTUBE___VIDEO_PLAYBACK___P_720___MOS'
            values='Val %'
            Helper.plot3(df, index, columns)

        except IOError as error:
            print(error)
        
    def get_feedbackmos_youtubeautomos(self):
        """
            Generate users'feedback MOS vs YoutubeAuto MOS
            Given user's feedback MOS of a specific class, associate to MOS classes predicted by Youtube Auto
            For instance given user's feebdback MOS class = 1, count the number of MOS classes predicted 
            by Youtube Auto model.
            
            Feedback MoS: {user_fb_mos: 1, estimated_yuauto_mos: {cl-1: x1, cl-2: x2, cl-3: x3, cl-4: x4, cl-5:x5}
            Estimated MoS using Youtube Auto: [1..5]
        """

        # Read User Feedback from csv
        user_feedback = UserFeedback(**self.config)
        ### Read all feedbacks from table user's feedback
        criteria = {}
        df_userfb = user_feedback.get_user_feedback(criteria)

        # Filter by Feedback_App name; Feedback_App= YOUTUBE_GENERAL
        try:
            # Filter by YOUTUBE___GENERAL___MOS
            mask = df_userfb[self.config['COL']['FB_APP']] == self.config['COL']['YU_G']
            app_yu_g = df_userfb[mask]

            # Select 2 Columns: FB_APP and YU_VP_720
            # app_yu_g = app_yu_g[[self.config['COL']['FB_DATE'], self.config['COL']['FB_VAL'], self.config['COL']['YU_VP_720']]]
            app_yu_g = app_yu_g[[self.config['COL']['FB_VAL'], self.config['COL']['YU_VP_720']]]

            ## Remove Nan or Zero from records
            app_yu_g[self.config['COL']['YU_VP_720']] = app_yu_g[self.config['COL']['YU_VP_720']].apply(lambda x: 0 if len(str(x))==0 or str(x)=='nan' else x)
            
            ## Recording the inconsistent instances index 
            dropIx = app_yu_g[app_yu_g[self.config['COL']['YU_VP_720']]==0].index
            ## Dropping these instances from the dataset:
            app_yu_g.drop(dropIx, inplace=True)

            #df = pd.pivot_table(app_yu_g, index=self.config['COL']['FB_VAL'], columns=self.config['COL']['YU_VP_720'], values=self.config['COL']['YU_VP_720'], aggfunc=np.sum)
            # df = pd.pivot_table(app_yu_g, index=self.config['COL']['FB_VAL'], columns='idx', values=self.config['COL']['YU_VP_720'], aggfunc=lambda x: len(x.unique()))

            # Group Records by estimated MoS: (1, 5)
            # grouped_fbs_by_user = df_userfbs.groupby([self.config['GROUP_BY']['USER_ID']]).count().reset_index()
            df = app_yu_g.groupby([self.config['COL']['FB_VAL']]).count().reset_index()
            df['idx'] = df[self.config['COL']['FB_VAL']]

            # Save for testing purpose
            filename = self.config['OUT_DIR'] + 'groupby_user_cat_3.csv'
            # _groupby_df.transpose().to_csv(filename)
            df.to_csv(filename)

            # Get List of MoS from Config
            mos_list = self.config['CONST']['MOS']

            # Create dataframe of columns: [1, 2, 3, 4, 5]
            
            # Dataframe
            df = pd.DataFrame()
            # iterate over MOS list one by one from 1 to 5
            for mos_val in mos_list:
                # Filter by YOUTUBE___GENERAL___MOS
                _mask = app_yu_g[self.config['COL']['FB_VAL']] == mos_val
                _df = app_yu_g[_mask]

                # Group Records by estimated MoS: (1, 5)
                # grouped_fbs_by_user = df_userfbs.groupby([self.config['GROUP_BY']['USER_ID']]).count().reset_index()
                _groupby_df = _df.groupby([self.config['COL']['YU_VP_720']]).count().reset_index()

                # Reshape row to column 
                _sum = (_groupby_df[self.config['COL']['FB_VAL']]).values.reshape(1,-1)
                _sum = np.around(_sum/np.sum(_sum), decimals=2)
                _df = pd.DataFrame(_sum, index=[mos_val], columns = _groupby_df[self.config['COL']['YU_VP_720']])

                #
                if df.empty:
                    df = pd.DataFrame(_df)
                else:
                    frames = [df, _df]
                    df = pd.concat(frames)
                    
            # Save for testing purpose
            filename = self.config['OUT_DIR'] + 'fb_user_cat_2.csv'
            # _groupby_df.transpose().to_csv(filename)
            app_yu_g.to_csv(filename)
            # Create the pandas DataFrame

            # import seaborn as sns; sns.set()
            # flights_long = sns.load_dataset("flights")
            # # Save for testing purpose
            # filename2 = self.config['OUT_DIR'] + 'flight_user_cat.csv'
            # # _groupby_df.transpose().to_csv(filename)
            # flights_long.to_csv(filename2)

            # Draw heatmap
            #Helper.plot_heatmap2(df)

        except IOError as error:
            print(error)




