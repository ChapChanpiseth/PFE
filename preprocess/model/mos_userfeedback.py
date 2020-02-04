# -*- coding: utf-8 -*-

import pandas as pd

from model.base import Base


class MosUserFeedback(Base):
    """
        Table: Dataset MOSUserFeedback MOSYoutubeAuto
    """

    def __init__(self, **kwargs):
        Base.__init__(self, **kwargs)

    def read_csv(self, criteria=None):
        """
            Read data from table Feedback
        """

        # User Feedback file name
        filename = self.config['FILE_DIR'] + self.config['IN_FNAME']['CSV_MOS_USERFEEDBACK']

        # Retreive columns header from excel file
        usecols = self.header_cols(filename)

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

    def get_mos_userfeedback(self, criteria=None):
        """ Retrieve MOSUserFeedback MOSYoutubeAuto
        """

        # Add prefix to column's name
        df_mos_userfbs = self.read_csv(criteria)
        #df_userfbs = df_userfbs.add_prefix(self.config['PREFIX_CHEV'])

        # Filter Dataframe by SUBJECT_ID, HADM_ID and ICUSTAY_ID
        # mask = (df_chartevs[self.config['PREFIX_CHEV'] + 'SUBJECT_ID'].isin(\
        #     criteria[self.config['PREFIX_CHEV'] + 'SUBJECT_ID'].tolist()))\
        #     & (df_chartevs[self.config['PREFIX_CHEV'] + 'HADM_ID'].isin(\
        #         criteria[self.config['PREFIX_CHEV'] + 'HADM_ID'].tolist()))

        return df_mos_userfbs

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
