"""

author: Rodrigo Lucchesi
date: February 2022
"""
# library doc string
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from loguru import logger
from typing import Dict, List, Tuple
import json
import os
from datetime import datetime, date
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class MlController:

    def __init__(self, name):
        self.__name__ = name
