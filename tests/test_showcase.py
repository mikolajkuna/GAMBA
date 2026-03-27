import pytest
import numpy as np
import sys
sys.path.insert(0, '.')

from src.config import RAW_DATA, RANDOM_SEED
from src.dataset import load_raw_salary
from src.features import preprocess, make_xy


def test_data_loads():
    df = load_raw_salary(RAW_DATA)
    assert len(df) == 2000
    assert 'income' in df.columns
    assert 'gender' in df.columns


def test_preprocessing():
    df = load_raw_salary(RAW_DATA)
    df_clean = preprocess(df)
    assert len(df_clean) == 2000
    assert df_clean['gender'].isin([0, 1]).all()
    assert (df_clean['income'] >= 1276).all()


def test_income_range():
    df = load_raw_salary(RAW_DATA)
    df_clean = preprocess(df)
    X, y = make_xy(df_clean)
    assert y.min() >= 1276
    assert y.max() < 100000
    assert X.shape == (2000, 8)


def test_gender_distribution():
    df = load_raw_salary(RAW_DATA)
    df_clean = preprocess(df)
    counts = df_clean['gender'].value_counts()
    assert set(counts.index) == {0, 1}
    assert counts.min() > 100  
