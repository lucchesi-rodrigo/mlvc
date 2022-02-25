"""

author: Rodrigo Lucchesi
date: February 2022
"""
# library doc string
import json
import os
from datetime import datetime,date
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from typing import Dict,List,Tuple

from loguru import logger
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report



class MlVersion:
    

    def __init__(self, name):
        self.__name__ = name