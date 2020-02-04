"""
    Preprocessing
"""

import time, math

import pandas as pd
import numpy as np

from model.mos_userfeedback import MosUserFeedback

from utils.file_helper import FileHelper
from utils.helper import Helper

class MosUserFbViewModel:
    """
        MosUserFbViewModel
    """

    def __init__(self, **kwargs):
        self.config = kwargs
        # Initial object Mos User Feedback
        self.mos_userfeedback = MosUserFeedback(**self.config)

    # def generate_mos_by_criteria(self, app_feeback):
    #     """
    #     """

    def generate_mos_userfb_yuauto(self):
        """
            Generate MOS User Feedback and MOS Youtbe Auto
            Feedback MoS: [1..5]
            Youtube Auto MoS: [1..5]
        """

        ### Read all feedbacks from table MOS user feedback - youtube auto
        criteria = {}
        df_mos_userfb = self.mos_userfeedback.get_mos_userfeedback(criteria)
        ## Remove Nan or Zero from records
        df_mos_userfb.dropna()

        df_pivot = None
        # Filter by Feedback_App name; Feedback_App= YOUTUBE_GENERAL
        try:
            # Filter column APP_FEEDBACK by value YOUTUBE___GENERAL___MOS
            mask = df_mos_userfb[self.config['COL']['FB_APP']] == self.config['COL']['YU_G']
            mos_feedback_yuauto = df_mos_userfb[mask]

            # Select 2 Columns: FEEDBACK_VALUE and YOUTUBE_AUTO_MOS
            # app_yu_g = app_yu_g[[self.config['COL']['FB_DATE'], self.config['COL']['FB_VAL'], self.config['COL']['YU_AUTO_MOS']]]
            mos_feedback_yuauto = mos_feedback_yuauto[[self.config['COL']['FB_VAL'], self.config['COL']['YU_AUTO_MOS']]]

            # Save for testing purpose
            filename = self.config['OUT_DIR'] + 'MOS_USERFEEDBACK_YOUTUBEAUTO.csv'
            mos_feedback_yuauto.to_csv(filename)

            # Get List of MoS from Config
            # Create dataframe of columns: [1, 2, 3, 4, 5]
            mos_list = self.config['CONST']['MOS']
            # Dataframe
            df = pd.DataFrame()
            # iterate over MOS list one by one from 1 to 5

            # x_grouby_youtube_g=mos_feedback_yuauto.groupby('FEEDBACK_VALUE')
            # for name,group in x_grouby_youtube_g:
            #     print(name,np.sum(group[group['YOUTUBE_AUTO_MOS']==1]['YOUTUBE_AUTO_MOS'])/np.sum(group['YOUTUBE_AUTO_MOS']),
            #     np.sum(group[group['YOUTUBE_AUTO_MOS']==2]['YOUTUBE_AUTO_MOS'])/np.sum(group['YOUTUBE_AUTO_MOS']),
            #     np.sum(group[group['YOUTUBE_AUTO_MOS']==3]['YOUTUBE_AUTO_MOS'])/np.sum(group['YOUTUBE_AUTO_MOS']),
            #     np.sum(group[group['YOUTUBE_AUTO_MOS']==4]['YOUTUBE_AUTO_MOS'])/np.sum(group['YOUTUBE_AUTO_MOS']),
            #     np.sum(group[group['YOUTUBE_AUTO_MOS']==5]['YOUTUBE_AUTO_MOS'])/np.sum(group['YOUTUBE_AUTO_MOS']))

            for mos_val in mos_list:
                # Filter by MOS class range(1,6)
                _mask = mos_feedback_yuauto[self.config['COL']['FB_VAL']] == mos_val
                _df = mos_feedback_yuauto[_mask]

                # Group Records by estimated MoS: (1, 5)
                # grouped_fbs_by_user = df_userfbs.groupby([self.config['GROUP_BY']['USER_ID']]).count().reset_index()
                _groupby_df = _df.groupby([self.config['COL']['YU_AUTO_MOS']]).count().reset_index()

                class_yuauto =  list(_groupby_df[self.config['COL']['YU_AUTO_MOS']])
                row_idx = -1
                for class_no in mos_list:
                    df_temp = None
                    if not class_no in class_yuauto:
                        if (row_idx != -1):
                            df_temp = pd.DataFrame(np.array([[0,class_no,class_no]]), columns = [self.config['COL']['FB_VAL'], self.config['COL']['YU_AUTO_MOS'], 'idx'])
                            _groupby_df = pd.concat([_groupby_df, df_temp], sort=False)
                            print(class_no)
                    else:
                        row_idx = class_no

                # Reshape row to column 
                _groupby_df = _groupby_df.sort_values(by=[self.config['COL']['YU_AUTO_MOS']]).reset_index()
                _sum = (_groupby_df[self.config['COL']['FB_VAL']])
                _groupby_df['Val_%'] = np.around(_sum/np.sum(_sum), decimals=2)
                _groupby_df['Feedback_MoS'] = ['m_' + str(mos_val) for i in range(5)]

                #
                if df.empty:
                    df = pd.DataFrame(_groupby_df)
                else:
                    frames = [df, _groupby_df]
                    df = pd.concat(frames)

            # Drop columns: index and idx
            filename = self.config['OUT_DIR'] + 'MOS_USERFEEDBACK_YOUTUBEAUTO_GROUPBY-BEFOREPIVOT.csv'
            df.to_csv(filename)
            df = df.drop(['idx' , 'FEEDBACK_VALUE', 'index'] , axis='columns')
            
            # Create the pandas DataFrame
            # import seaborn as sns; sns.set()
            # flights_long = sns.load_dataset("flights")
            # # Save for testing purpose
            # filename2 = self.config['OUT_DIR'] + 'flight_user_cat.csv'
            # # _groupby_df.transpose().to_csv(filename)
            # flights_long.to_csv(filename2)
            df_pivot = df.pivot("YOUTUBE_AUTO_MOS", "Feedback_MoS", "Val_%")
            print(df_pivot)

            # Save for testing purpose
            filename = self.config['OUT_DIR'] + 'MOS_USERFEEDBACK_YOUTUBEAUTO_GROUPBY.csv'
            df_pivot.to_csv(filename)
            # Draw heatmap
            #Helper.plot_heatmap2(df)

        except IOError as error:
            print(error)
        
        return df_pivot





