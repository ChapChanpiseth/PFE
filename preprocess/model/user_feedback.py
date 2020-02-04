# -*- coding: utf-8 -*-

import pandas as pd

from model.base import Base


class UserFeedback(Base):
    """
        Table: UserFeedback
    """

    def __init__(self, **kwargs):
        Base.__init__(self, **kwargs)

    def read_csv(self, criteria=None, filename=None):
        """
            Read data from table Feedback
        """

        # User Feedback file name
        if filename is None:
            filename = self.config['FILE_DIR'] + self.config['IN_FNAME']['CSV_USER_FEEDBACK']
        else: filename = filename
            
        # Retreive columns header from excel file
        usecols = self.header_cols(filename)

        # usecols = ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID',	'ITEMID', 'CHARTTIME', 'STORETIME',\
        # 	'CGID', 'VALUE', 'VALUENUM', 'VALUEUOM', 'WARNING', 'ERROR', 'RESULTSTATUS', 'STOPPED']

        # Set column dtype=str: Avoid ambiguity of Python interpreter
        # col_dtype = { 
        #     'ROW_ID': str, 'SUBJECT_ID': str, 'HADM_ID':str, 'ICUSTAY_ID':str,	'ITEMID':str,\
        #         'CHARTTIME':str, 'STORETIME': str, 'CGID': str, 'VALUE': str, 'VALUENUM': str,\
        #             'VALUEUOM':str, 'WARNING':str, 'ERROR':str, 'RESULTSTATUS':str, 'STOPPED':str}

        # Read from csv file
        #df_chartevs = pd.read_csv(filename, encoding='latin1', usecols=usecols, dtype=col_dtype)
        # Read from csv file
        if not criteria:
            # df_userfbs = pd.read_csv(filename, encoding='latin1', usecols=usecols, dtype=col_dtype)
            df_userfbs = pd.read_csv(filename, encoding='latin1', usecols=usecols)
        elif self.config['CONST']['N_ROWS'] in criteria:
            df_userfbs = pd.read_csv(filename, encoding='latin1', usecols=usecols,\
                nrows=criteria[self.config['CONST']['N_ROWS']])
                # nrows=criteria[self.config['CONST']['N_ROWS']], dtype=col_dtype)
        else: 
            #df_userfbs = pd.read_csv(filename, encoding='latin1', usecols=usecols, dtype=col_dtype)
            df_userfbs = pd.read_csv(filename, encoding='latin1', usecols=usecols)

        return df_userfbs

    def get_user_feedback(self, criteria=None):
        """ Retrieve CHARTEVENTS matching the give hospital admission
        """

        # Add prefix to column's name
        df_userfbs = self.read_csv(criteria)
        #df_userfbs = df_userfbs.add_prefix(self.config['PREFIX_CHEV'])

        # Filter Dataframe by SUBJECT_ID, HADM_ID and ICUSTAY_ID
        # mask = (df_chartevs[self.config['PREFIX_CHEV'] + 'SUBJECT_ID'].isin(\
        #     criteria[self.config['PREFIX_CHEV'] + 'SUBJECT_ID'].tolist()))\
        #     & (df_chartevs[self.config['PREFIX_CHEV'] + 'HADM_ID'].isin(\
        #         criteria[self.config['PREFIX_CHEV'] + 'HADM_ID'].tolist()))
        
        # df_chartevs = df_chartevs[mask]
        return df_userfbs
    
    def get_2k_userfeedback(self, criteria=None):
        """
            Get 2K dataset
        """
        filename = self.config['FILE_DIR'] + self.config['IN_FNAME']['CSV_2K_DATASET']
        # Add prefix to column's name
        df_userfbs = self.read_csv(criteria, filename)

        return df_userfbs

    def writte_csv(self, categories, counts):
        """
            Count unique FEEDBACK_APP and Occurrences corresponding to each application
        """

        try:
            # User Feedback file name
            filename = self.config['OUT_DIR'] + self.config['OUT_FNAME']['CSV_FEEDBACK_FOR_APP']

            df = pd.DataFrame({'FEEDBACK_APP':categories, 'OCURRENCE':counts})
            df.to_csv(filename)
        except IOError as error:
            print(error)

    
    def get_feedback_by_topuser(self, criteria=None):
        """ Retrieve CHARTEVENTS matching the give hospital admission
        """

        # Add prefix to column's name
        df_userfbs = self.read_csv(criteria)
        #df_userfbs = df_userfbs.add_prefix(self.config['PREFIX_CHEV'])

        # Filter Dataframe by SUBJECT_ID, HADM_ID and ICUSTAY_ID
        grouped_fbs_by_user = df_userfbs.groupby([self.config['GROUP_BY']['USER_ID']]).count().reset_index()

        # Pandas: Find maximum values & position in columns or rows of a Dataframe
        # Get maximum value of a single column 'y'
        # max_value = grouped_fbs_by_user[self.config['COL']['FB_APP']].max()
        idx_max_val = grouped_fbs_by_user[self.config['COL']['FB_APP']].idxmax()

        # Get USER ID reportig the most
        user_id = grouped_fbs_by_user.iloc[idx_max_val, :][self.config['COL']['UID']]

        # Filter Dataframe by SUBJECT_ID, HADM_ID and ICUSTAY_ID
        mask = df_userfbs[self.config['COL']['UID']] == user_id
        fb_top_user = df_userfbs[mask]

        print("Maximum value in column 'y': " , user_id)

        # df_chartevs = df_chartevs[mask]
        return fb_top_user

    # def get_chartevents_by_phadmicu(self, criteria=None):
    #     """ Retrieve CHARTEVENTS matching the give hospital admission
    #     """

    #     # Add prefix to column's name
    #     df_chartevs = self.read_csv(criteria)
    #     df_chartevs = df_chartevs.add_prefix(self.config['PREFIX_CHEV'])

    #     # Filter Dataframe by SUBJECT_ID, HADM_ID and ICUSTAY_ID
    #     mask = (df_chartevs[self.config['PREFIX_CHEV'] + 'SUBJECT_ID'].isin(\
    #         criteria[self.config['PREFIX_CHEV'] + 'SUBJECT_ID'].tolist()))\
    #         & (df_chartevs[self.config['PREFIX_CHEV'] + 'HADM_ID'].isin(\
    #             criteria[self.config['PREFIX_CHEV'] + 'HADM_ID'].tolist()))
        
    #     df_chartevs = df_chartevs[mask]

    #     return df_chartevs
