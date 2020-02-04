"""
    Preprocessing
"""

import time, math

import pandas as pd
import numpy as np

from model.user_feedback import UserFeedback
from model.mos_userfeedback import MosUserFeedback
from viewmodel.user_fd_viewmodel import UserFbViewModel
from viewmodel.mos_userfb_viewmodel import MosUserFbViewModel

from model.step2.data_compilation import DataCompilation

from utils.file_helper import FileHelper
from utils.helper import Helper

class PreproAcqua:
    """
        Preprocessing
    """

    def __init__(self, **kwargs):
        self.config = kwargs
        self.userfd_vm = UserFbViewModel(**self.config)
        self.mos_userfd_vm = MosUserFbViewModel(**self.config)

    def start_process(self):
        """ Read data from tables: UserFeedback
        """

        ### Read data from admission based on PAINTENT ID (SUBJECT_ID)
        criteria = {}
        #criteria[self.config['CONST']['N_ROWS']] = 10
        ### Read admissions groupby date and
        # self.userfd_vm.count_userfb_bygroup2()

        # Group by User
        #*************************************************************#
        #* Data Cleaning and Dataset mapping *#
        #*************************************************************#
        # Retrieve all users'feedbacks and Keep only numeric RTT
        #new_dataset = self.userfd_vm.dataset_propre()
        # Save data to csv file

        #*************************************************************#
        #* Make Prediction of users'feedbacks using YoutubeAuto Dash *#
        #*************************************************************#
        # Retrieve all users'feedbacks and Keep only numeric RTT
        est_qoe_byyoutubeauto = self.userfd_vm.predict_youtubeauto()

        #****************************************************************#
        #* Draw heatmap and correlation                                 *#
        # MOS of users'feedbacks vs MOS estimated using YoutubeAuto Dash *#
        #****************************************************************#
        # Retrieve all users'feedbacks and Keep only numeric RTT
        #** self.mos_userfd_vm.generate_mos_userfb_yuauto()

    def __get_user_feedback(self, criteria=None):
        """ Read User Feedback
        """
         ### criteria = {'nrows':10}
        user_feedback = UserFeedback(**self.config)

        ### Read admissions groupby date and
        ### Choose admissions of the year during which contains biggest number of admissions
        df_userfb = user_feedback.get_user_feedback(criteria)
        
        ### Limit number of patients based on condition LIMIT_NUM_PATIENT
        #df_adms = self.__shape_num_patient_by_limit(df_adms)
        #filename = self.config['OUT_DIR_S1'] + self.config['OUT_FNAME']['ADMISSIONS']
        #FileHelper.save_to_csv(df_adms, filename)
        fd_for_app, counts = Helper.count_ucategory(df_userfb.iloc[:, 0])

        #fd_for_app = np.frompyfunc(lambda x: np.char.replace(x, "___MOS", "", count=0), 1, 1)(fd_for_app)

        #u_categories, counts = None, None

        # Helper.plot_hist(u_categories, counts)

        # fd_for_app = Helper.map_word_to_abbr(fd_for_app, self.config['WORD_ABBR']['FEEDBACK_APP_ATT'])

        # user_feedback.writte_csv(fd_for_app, counts)

        # print out first colms
        # print()
        # print(df_userfb)
        ### Draw histogram of Occurence corresponding to Application
        Helper.plot_histogram(fd_for_app, counts)

        # Draw histogram: In percentage the occurence corresponding to Application
        total_fbs = np.sum(counts)
        Helper.plot_histogram(fd_for_app, counts/total_fbs)

        return df_userfb

    def get_feedack_per_user(self, criteria):
        """ Calculate feedback per user
        """

         ### criteria = {'nrows':10}
        user_feedback = UserFeedback(**self.config)

        ### Read admissions groupby date and
        ### Choose admissions of the year during which contains biggest number of admissions
        df_userfb = user_feedback.get_user_feedback(criteria)

        grouped_fbs_by_user = df_userfb.groupby([self.config['GROUP_BY']['USER_ID']]).count().reset_index()

        _users = grouped_fbs_by_user['USER_ID']
        _index = np.arange(1, _users.shape[0]+1)
        _counts = grouped_fbs_by_user['FEEDBACK_APP']

        filename = self.config['OUT_DIR'] + 'USER_FEEDBACK_21NOV19.csv'
        grouped_fbs_by_user.to_csv(filename)

        ### Draw histogram of Occurence corresponding to Application
        Helper.plot_graphline(_index, _counts, 0)

    def get_feedback_top_user(self, criteria):
        """ Get the feedback of top user
        """

        user_feedback = UserFeedback(**self.config)
        ### Read admissions groupby date and
        df_userfb = user_feedback.get_feedback_by_topuser(criteria)

        try:
            # Filter by YOUTUBE___GENERAL___MOS
            mask = df_userfb[self.config['COL']['FB_APP']] == self.config['COL']['YU_G']
            app_yu_g = df_userfb[mask]

            # SELECT ONLY 2 COLUMNS
            app_yu_g = app_yu_g[[self.config['COL']['FB_DATE'], self.config['COL']['FB_VAL'], self.config['COL']['YU_VP_720']]]
            
            app_yu_g[self.config['COL']['YU_VP_720']] = app_yu_g[self.config['COL']['YU_VP_720']].apply(lambda x: 0 if len(str(x))==0 or str(x)=='nan' else x)
            
            ## Recording the inconsistent instances index 
            dropIx = app_yu_g[app_yu_g[self.config['COL']['YU_VP_720']]==0].index
            ## Dropping these instances from the dataset:
            app_yu_g.drop(dropIx, inplace=True)

            #app_youtube_general = app_youtube_general.drop(app_youtube_general[self.config['COL']['YU_VP_720']]==0, inplace=True)
            #app_yu_g = app_yu_g.head(5)
            # y_labels = (self.config['COL']['FB_VAL'], self.config['COL']['YU_VP_720'])
            y_labels = ('YU_VDO_PLAYB_P720', self.config['COL']['FB_VAL'])
            #Helper.plot_heatmap(app_yu_g[[self.config['COL']['FB_DATE'], self.config['COL']['FB_VAL'], self.config['COL']['YU_VP_720']]])
            Helper.plot_heatmap(app_yu_g[self.config['COL']['YU_VP_720']].head(100), \
                app_yu_g[self.config['COL']['FB_VAL']].head(100), \
                y_labels)
            # User Feedback file name
            # filename = self.config['OUT_DIR'] + self.config['OUT_FNAME']['FEEDBACK_TOP_USER.csv']
            # filename = self.config['OUT_DIR'] + 'FEEDBACK_TOP_USER.csv'
            #filename = self.config['OUT_DIR'] + 'FEEDBACK_TOP_USER_YU_G_2COLS.csv'
            #app_yu_g.to_csv(filename)
        except IOError as error:
            print(error)

    def execute(self):
        """ Execute the preprocessing here
        """

        start = time.time()
        print('\n=====================================================================')
        print('\n*** Step 1: Manipulate Admission, ICUStay, OutputEvent and Chartevent\n')
        self.start_process()
        end = time.time()

        print("\n*** Execution time of %d patient(s) is %f ********\n" %\
            (self.config['PARAM']['LIMIT_NUM_PATIENT'], end - start))

if __name__ == "__main__":

    ### Input filename configuration
    F_INPUT = {
        'CSV_USER_FEEDBACK': 'CSV_USER_FEEDBACK.csv',
        'CSV_MOS_USERFEEDBACK': 'DATASET_MOSUSERFEEDBACK_MOSYOUTUBEAUTO.csv',
        'CSV_2K_DATASET': '2K_DATASET.csv'
    }

    ### Output filename configuration
    F_OUTPUT = {
        'CHEV_BY_DATE_INTERVAL': 'OUT_CHEV_BY_DATE_INTERVAL.csv',
        'PT_ADM_ICUS_CHAREVS': 'OUT_PT_ADM_ICUS_CHAREVS.csv',
        'OUTEVENT_CHAREVS': 'OUT-OUTEVENT_CHAREVS.csv',
        'TEMP_DF': 'OUT_TEMP_DF.csv',
        'OUT_NUM_EVENTS_WINSIZE_24H':'OUT_NUM_EVENTS_WINSIZE_24H.csv',
        'OUT_LIMIT_NUM_EVENTS_WINSIZE_24H':'OUT_LIMIT_NUM_EVENTS_WINSIZE_24H.csv',
        'CSV_FEEDBACK_FOR_APP': 'FEEDBACK_FOR_APP.csv'  
    }

    ### CONST: THEY ARE USED IN STEP 2
    CONST = {
        'SUBJECT_ID': 'subject_id',
        'HADM_ID': 'hadm_id',
        'ICUSTAY_ID': 'icustay_id',
        'HUNIT': 'unit',
        'HUNIT_ICU': 'ICU',
        'HUNIT_CHAREV': 'CHAREV',
        'HUNIT_CHARTEVENT': 'CHEVENT',
        'PROCEDURE': 'procedure',
        'K_YES': 'YES',
        'K_NO': 'NO',
        'N_ROWS': 'N_ROWS',
        'MOS': [1, 2, 3, 4, 5]
    }

    ### PARAM: Parameters to tune so as to sharp the number of ouput records
    ### DOB is the date of birth of the given patient. Patients who are older than 89 years old 
    # at any time in the database have had their date of birth shifted to obscure 
    # their age and comply with HIPAA
    ### When Setting READ_ALL_RECORDS=YES, then LIMIT_NUM_PATIENT has no effect
    # - LIMIT_NUM_CHARTEVENTS: there are 330 million records in this CHARTEVENTS, so it is good
    # to limit to 10 million records for less time consuming
    PARAM = {
        'READ_ALL_RECORDS': 'YES',
        'LIMIT_NUM_PATIENT': 0,
        'LIMIT_NUM_CHARTEVENTS': 10000
    }

    ### ABRREVATION
    ABBR = {
        'FEEDBACK_APP_ATT': ('EM_G', 'FB_G', 'HO_G', 'IG_G', 'MSN_G', 'NFX_G', 'RSS_G', \
            	'SK_AUD', 'SK_IMG', 'SK_VDO', 'SK_CHA', 'SK_G', 'SK_VDO_CAL', 'SK_VOI_CAL', \
                'SC_G',	'SFY_G', 'TG_G', 'TT_G', 'VB_G', 'W3_G', 'WA_G', 'YU_COM', 'YU_G', \
                'YU_SCH', 'YU_VP_P360', 'YU_VP_480', 'YU_VP_720', 'YU_VP_AUTO', 'YU_VP_S1440' \
                'YU_VP_S144', 'YU_VP_S360')   
    }
																											
    WORD_ABBR = {
        'FEEDBACK_APP_ATT': {'EMAIL___GENERAL___MOS': 'EM_G', 'FACEBOOK___GENERAL___MOS':'FB_G', \
            'HANGOUTS___GENERAL___MOS':'HO_G', 'INSTAGRAM___GENERAL___MOS':'IG_G', 'MESSENGER___GENERAL___MOS':'MSN_G', \
            'NETFLIX___GENERAL___MOS':'NFX_G', 'RSS___GENERAL___MOS':'RSS_G', 'SKYPE___CHAT_AUDIO_MESSAGE___MOS':'SK_AUD',\
            'SKYPE___CHAT_IMAGE_MESSAGE___MOS':'SK_IMG', 'SKYPE___CHAT_VIDEO_MESSAGE___MOS':'SK_VDO', 'SKYPE___CHAT___MOS':'SK_CHA',\
            'SKYPE___GENERAL___MOS':'SK_G', 'SKYPE___VIDEO_CALL___MOS':'SK_VDO_CAL', 'SKYPE___VOICE_CALL___MOS':'SK_VOI_CAL', \
            'SNAPCHAT___GENERAL___MOS':'SC_G',	'SPOTIFY___GENERAL___MOS':'SFY_G', 'TELEGRAM___GENERAL___MOS':'TG_G',\
            'TWITTER___GENERAL___MOS':'TT_G', 'VIBER___GENERAL___MOS':'VB_G', 'WEB___GENERAL___MOS':'W3_G', 'WHATSAPP___GENERAL___MOS':'WA_G',\
            'YOUTUBE___COMMENT___MOS':'YU_COM', 'YOUTUBE___GENERAL___MOS':'YU_G', \
            'YOUTUBE___SEARCH___MOS':'YU_SCH', 'YOUTUBE___VIDEO_PLAYBACK___P_360___MOS':'YU_VP_P360', 'YOUTUBE___VIDEO_PLAYBACK___P_480___MOS':'YU_VP_480',\
            'YOUTUBE___VIDEO_PLAYBACK___P_720___MOS':'YU_VP_720', 'YOUTUBE___VIDEO_PLAYBACK___P_AUTO___MOS':'YU_VP_AUTO', 'YOUTUBE___VIDEO_PLAYBACK___S_1440___MOS':'YU_VP_S1440',\
            'YOUTUBE___VIDEO_PLAYBACK___S_144___MOS':'YU_VP_S144', 'YOUTUBE___VIDEO_PLAYBACK___S_360___MOS':'YU_VP_S360', 'RTT':'RTT'}
    }

    GROUP_BY = {'USER_ID': 'USER_ID'}

    COLUMNS = {'FB_APP':'FEEDBACK_APP', 'UID': 'USER_ID', 'YU_G':'YOUTUBE___GENERAL___MOS', 'YU_AUTO_MOS':'YOUTUBE_AUTO_MOS',\
        'YU_VP_720':'YOUTUBE___VIDEO_PLAYBACK___P_720___MOS', 'FB_VAL':'FEEDBACK_VALUE', 'USER_ID': 'USER_ID', \
            'FB_DATE':'FEEDBACK_DATE', 'RTT': 'RTT', 'DOWNLOAD_LOSS_RATE':'DL', 'UPLOAD_LOSS_RATE': 'UL', \
            'DOWNLOAD_JITTER': 'DJ', 'UPLOAD_JITTER':'UJ', 'UDP_DOWNLOAD_THROUGHPUT':'DTH', \
                'UDP_UPLOAD_THROUGHPUT': 'UTH'}

    CONFIG = {
        'FILE_DIR': '/Volumes/D/py-workspace/PFE/dataset/Input/',
        'MODEL_JSON_DIR_': 'dataset/model/json',
        'FILE_DIR_S2': '/Volumes/DATASSD/Mimic/Data/Input/Step2/',
        'OUT_DIR': '/Volumes/D/py-workspace/PFE/dataset/output/',
        'OUT_DIR_S1': '/Volumes/DATASSD/Mimic/Data/Output/Step1/',
        'OUT_DIR_S2': '/Volumes/DATASSD/Mimic/Data/Output/Step2/',
        'IN_FNAME': F_INPUT,
        'OUT_FNAME': F_OUTPUT,
        'CONST': CONST,
        'PARAM': PARAM,
        'ABBR': ABBR,
        'WORD_ABBR':WORD_ABBR,
        'GROUP_BY': GROUP_BY,
        'PREFIX_HADM': 'HADM_',
        'PREFIX_ICU': 'ICU_',
        'PREFIX_CHEV': 'CHEV_',
        'PREFIX_OUEV': 'OUEV_',
        'PREFIX_DITEM': 'DITEM_',
        'COL': COLUMNS
    }

    ###
    prepro = PreproAcqua(**CONFIG)
    prepro.execute()
