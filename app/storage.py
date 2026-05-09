import pandas as pd

DATA = pd.DataFrame()


def save_data(df: pd.DataFrame):
    global DATA
    DATA = pd.concat([DATA, df], ignore_index=True)


def get_data():
    return DATA.copy()


def clear_data():
    global DATA
    rows = len(DATA)
    DATA = pd.DataFrame()
    return rows