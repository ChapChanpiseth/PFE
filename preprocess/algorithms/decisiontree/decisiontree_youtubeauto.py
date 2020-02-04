"""
    This Decision Tree Youtube Auto
"""

import sys, os
import argparse
import time
import numpy as np
from copy import deepcopy

syspath = 'PFE'
sys.path.append(os.path.abspath(os.path.join('..', syspath)))

# from ml_factory import MLFactory
from utils.file_helper  import FileHelper

class DecisionTreeYoutubeAuto:
    """ This class is built for YoutubeAuto Dash Decision Tree Model 
    """

    def __init__(self, **kwargs):
        self.config = kwargs
        self.axis = 'axis'
        self.comp_ope = 'comparisonOperator'
        self.r_operand = 'rightOperand'
        self.qoe_class = 'qoeClass'

    def decide_branch(self, comp_operand, val, f_name, features):
        """
            Decide which branch to select: trueBranch or falseBranch
            Parameters
            ----------
            comp_operand : Comperation Operand "lte"
            val          : value given by model tree
            f_name       : name of the feature
            features     : {dictionary-like}

            Returns
            -------
            trueBranch/falseBranch : String
        """

        if comp_operand == 'lte':
            # if features[f_name] >= val:
            # print(type(features[f_name]), features[f_name])
            if features[f_name] <= val:
                return 'trueBranch'
            else: return 'falseBranch'
        else:
            return None

    def predict(self, json_model, X_test):
        """
            Predict QoE given decision tree model in json format

            Parameters
            ----------
            json_model  : {json format}
            X_test      : {list of dictionary of features}

            Returns
            -------
            MoS         : {List of estimated MoS value}
        """

        # We predict MoS record by record
        estimated_mos = []
        for features in X_test:
            
            _block_info = {}
            found_qoe = False
            depth = 0
            branch = None
            detail_node = []

            _model = deepcopy(json_model)
            while (not found_qoe):
                # Iterate through block information, and keep it in dictionary
                # such as: axis, comparisonOperator, rightOperand, trueBranch, falseBranch
                for key in _model:

                    # Asap, we reach the leaf of the Tree at the moment, 
                    # Retrieve QoE class and leave the loop
                    if key == self.qoe_class:
                        #print('QoE class is %s' % (_model[key]))
                        found_qoe = True
                        estimated_mos.append({'idx': features['idx'], 'YOUTUBE_AUTO_MOS': _model[key], 'Depth': depth, 'Detail_node': detail_node})
                        # Leave while loop
                        break
                    else:
                        _block_info[key] = _model[key]

                # Once we have information about the block, we are able to calculate cost function
                # To decide whether go to trueBranch or falseBranch
                if (not found_qoe) and bool(_block_info):

                    # Parameters for cost function
                    comp_operand = _block_info[self.comp_ope]
                    val = _block_info[self.r_operand]
                    f_name = _block_info[self.axis]

                    # Increment the deep by one
                    depth += 1

                    # Get Branch
                    branch = self.decide_branch(comp_operand, val, f_name, features)

                    # If there exists one branch, we continue. Otherwise, we break
                    if branch:
                        # We decide to continue with one branch of the tree,
                        # Select subtree of the tree
                        _model = deepcopy(_block_info[branch])
                        _block_info = {}
                        #print('----------------------')
                        my_string = "D {0} -> F_name: {1}  (F_Val: {2}, M_Val: {3}) Branch: {4}"
                        _node_info = my_string.format(depth, f_name, features[f_name], val, branch)
                        detail_node.append(_node_info)
                        #print('D %s -> F_name: %s  (F_Val: %s, M_Val: %s) Branch: %s' % (depth, f_name, features[f_name], val, branch))
                        #print('----------------------')
                    else:
                        break
        # 
        return estimated_mos

    def load_model_json(self, file):
        """
        Read decision tree model from json file
        """
        #from rapidjson import loads, Encoder
        import json

        try:
            with open(file, 'rb') as handle:
                #model = loads(Encoder(ensure_ascii=True)(handle))
                model = json.load(handle)
        except ValueError as error:
            raise Exception(error)
        else:
            return model

    def print(self, estimated_mos):
        # Estimated MoS
        print("\n\r****************************************\n\r")
        for _record in estimated_mos:
            print("*** Row_index %s Predicted QoE class %s | Depth %s" % (_record['idx'], _record['YOUTUBE_AUTO_MOS'], _record['Depth']))
            for _info in _record['Detail_node']:
                print(_info)
            print("--------------------------------")

    def __test__(self):

        #load json from file (feature data)
        filename = "youtube.dash.json"
        # file_uri = FileHelper.dataset_path(config, filename)
        # json_model = FileHelper.load_model_json(file_uri)
        json_model = self.load_model_json(filename)

        # Create test set
        features_1 = {'DTH': 187500, 'RTT': 234500000, 'DJ': 300000, 'DL': 0.005, 'UJ': 59000000, 'UL': 0.277, 'UTH': 6000000}
        features_2 = {'DTH': 187500, 'RTT': 234500000, 'DJ': 300000, 'DL': 0.005, 'UJ': 206000000, 'UL': 0.277, 'UTH': 6000000}
        # features_3 = {'DTH': 187400, 'RTT': 234500000, 'DJ': 1500000, 'DL': 0.005, 'UJ': 206000000, 'UL': 0.277, 'UTH': 6000000}
        features_3 = {'DTH': 4000000, 'RTT': 1000, 'DJ': 100, 'DL': 0.0, 'UJ': 1000, 'UL': 0.0, 'UTH': 2000000}
        features_4 = {'DTH':9113961.102 , 'RTT': 5040786.26, 'DJ': 556120.26, 'DL':0.0, 'UJ':816236.06 , 'UL':0.0, 'UTH': 16534836.79}
        
        X_test = [features_1, features_2, features_3, features_4]

        # Estimated MoS
        print("\n\r****************************************\n\r")
        estimated_mos = self.predict(json_model, X_test)
        for _record in estimated_mos:
            print("*** Predicted QoE class %s | Depth %s" % (_record['YOUTUBE_AUTO_MOS'], _record['Depth']))
            for _info in _record['Detail_node']:
                print(_info)
            print("--------------------------------")

if __name__ == "__main__":
    whole_st = time.time()

    config = {
        'MODEL_JSON_DIR_': 'dataset/model/json',
    }

    dt = DecisionTreeYoutubeAuto(**config)
    dt.__test__()
    prepro_time = time.time() - whole_st
