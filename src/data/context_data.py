import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from src.utils import get_sampler

### fill NaN of Age using location mean
def fill_age_na_with_loc_mean(user_):
    u_age, u_city, u_state, u_country = user_.iloc[1:5]
    
    if np.isnan(u_age):
        if u_state != "n/a":
            u_age = state_age_info[u_state] 
        elif u_country != "n/a":
            u_age = country_age_info[u_country] 

    user_["age"] = u_age    
    return user_ 

### modi from origin for zero mapping
def age_map(x) -> int:
    # x = int(x)
    if np.isnan(x):
        return 0
    elif x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6
    
def age_map_cat(x) -> int:
      x = int(x)
      if x < 20:
          return 1
      elif x >= 20 and x < 30:
          return 2
      elif x >= 30 and x < 40:
          return 3
      elif x >= 40 and x < 50:
          return 4
      elif x >= 50 and x < 60:
          return 5
      else:
          return 6
          

def process_context_data(users, books, ratings1, ratings2, process_cat, process_age, process_loc):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    process_cat : str
        books 데이터에서 선택할 카테고리 (원본, 상위)
    process_age : str
        Age 데이터의 결측치를 처리할 방법 (global_mean, zero_cat, loc_mean, rand_norm)
    ----------
    """

    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[-1])
    users = users.drop(['location'], axis=1)

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    if process_cat == 'basic': # 원본 카테고리
        print("+++++++++++++++++++ processing cat : BASIC +++++++++++++++++")
        context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
        train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
        test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category', 'publisher', 'language', 'book_author']], on='isbn', how='left')
    elif process_cat == 'high': # 상위 카테고리
        print("+++++++++++++++++++ processing cat : HIGH +++++++++++++++++")
        context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category_high', 'publisher', 'language', 'book_author']], on='isbn', how='left')
        train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category_high', 'publisher', 'language', 'book_author']], on='isbn', how='left')
        test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category_high', 'publisher', 'language', 'book_author']], on='isbn', how='left')


    # Location 처리
    print(f"+++++++++++++++++++ processing Loc : {process_loc} +++++++++++++++++")
    loc_city2idx = None
    loc_state2idx = None
    loc_country2idx = None
    if 'city' in process_loc:
        loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
        train_df['location_city'] = train_df['location_city'].map(loc_city2idx) 
        test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    else:
        train_df.drop(columns='location_city', inplace=True)
        test_df.drop(columns='location_city', inplace=True)
        
    if 'state' in process_loc:
        loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
        train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
        test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    else:
        train_df.drop(columns='location_state', inplace=True)
        test_df.drop(columns='location_state', inplace=True)
    
    if 'country' in process_loc:
        loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}
        train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
        test_df['location_country'] = test_df['location_country'].map(loc_country2idx)
    else:
        train_df.drop(columns='location_country', inplace=True)
        test_df.drop(columns='location_country', inplace=True)
    
    
    # Age 결측치 처리
    if process_age == 'global_mean': # fill NaN with global mean
        print("+++++++++++++++++++ processing Age : global_mean +++++++++++++++++")
        train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
        test_df['age']  = test_df['age'].fillna(int(test_df['age'].mean()))
    elif process_age == 'zero_cat': # fill NaN with zero_cat
        print("+++++++++++++++++++ processing Age : zero_cat +++++++++++++++++")
    elif process_age == 'loc_mean': # fill NaN with location_mean
        print("+++++++++++++++++++ processing Age : loc_mean +++++++++++++++++")
        state_age_info = dict()        
        for state in users['location_state'].unique():
            state_age_info[state] = users[users["location_state"] == state].age.mean()
            
        country_age_info = dict()
        for country in users['location_country'].unique():
            country_age_info[country] = users[users["location_country"] == country].age.mean()
        
        train_df = train_df.apply(fill_age_na_with_loc_mean, axis=1)
        train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
        test_df = test_df.apply(fill_age_na_with_loc_mean, axis=1)
        test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    elif process_age == 'rand_norm': # fill NaN with random numb from normal Dist
        print("+++++++++++++++++++ processing Age : rand_norm +++++++++++++++++")
        age_mean =  np.mean(train_df['age']) 
        age_std =  np.std(train_df['age']) 
        for idx in np.where(np.isnan(train_df['age']))[0]:
            train_df.loc[idx,'age'] = int(np.random.normal(age_mean, age_std,1))
            if train_df.loc[idx,'age'] < 0:
                train_df.loc[idx,'age'] *= -1
        
        age_mean =  np.mean(test_df['age']) 
        age_std =  np.std(test_df['age']) 
        for idx in np.where(np.isnan(test_df['age']))[0]:
            test_df.loc[idx,'age'] = int(np.random.normal(age_mean, age_std,1))
            if test_df.loc[idx,'age'] < 0:
                test_df.loc[idx,'age'] *= -1

    elif process_age == 'stratified': # fill NaN to have same distribution with origin
        print("+++++++++++++++++++ processing Age : stratified +++++++++++++++++")
        train_na_cnt = sum(np.isnan(train_df.age))
        train_age_sample = pd.DataFrame(train_df['age'].dropna().apply(age_map_cat))
        train_impute_list = train_age_sample.apply(lambda x : x.sample(n = train_na_cnt, replace = True)).reset_index(drop = True)
        for i,idx in enumerate(np.where(np.isnan(train_df['age']))[0]):
            train_df.loc[idx,'age'] = train_impute_list.age[i]

        test_na_cnt = sum(np.isnan(test_df.age))
        test_age_sample = pd.DataFrame(test_df['age'].dropna().apply(age_map_cat))
        test_impute_list = test_age_sample.apply(lambda x : x.sample(n = test_na_cnt, replace = True)).reset_index(drop = True)
        for i,idx in enumerate(np.where(np.isnan(test_df['age']))[0]):
            test_df.loc[idx,'age'] = test_impute_list.age[i]

    
    if process_age != 'stratified':
        train_df['age'] = train_df['age'].apply(age_map)
        test_df['age'] = test_df['age'].apply(age_map)
    
    
    # book 파트 인덱싱
    if process_cat == 'basic': # 원본 카테고리
        category2idx = {v:k for k,v in enumerate(context_df['category'].unique())}
        train_df['category'] = train_df['category'].map(category2idx)
        test_df['category'] = test_df['category'].map(category2idx)
    elif process_cat == 'high': # 상위 카테고리
        category2idx = {v:k for k,v in enumerate(context_df['category_high'].unique())}
        train_df['category_high'] = train_df['category_high'].map(category2idx)
        test_df['category_high'] = test_df['category_high'].map(category2idx)

    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}

    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['language'] = train_df['language'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['language'] = test_df['language'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "category2idx":category2idx,
        "publisher2idx":publisher2idx,
        "language2idx":language2idx,
        "author2idx":author2idx,
    }

    return idx, train_df, test_df


def context_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
        process_cat : str
            books 데이터에서 선택할 카테고리 (원본, 상위)
    ----------
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.data_path + 'users.csv')
    books = pd.read_csv(args.data_path + 'books.csv')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_test = process_context_data(users, books, train, test, args.process_cat, args.process_age, args.process_loc)
    
    ########## Build context data
    size = [len(user2idx), len(isbn2idx), 6]
    for xx2idx in ['loc_city2idx', 'loc_state2idx', 'loc_country2idx', 'category2idx', 'publisher2idx', 'language2idx', 'author2idx']:
        if xx2idx in idx and idx[xx2idx]:
            size.append(len(idx[xx2idx]))
    
    field_dims = np.array(size, dtype=np.uint32)

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data


def context_data_split(args, data):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """

    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def context_data_loader(args, data):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, sampler=get_sampler(args, data['y_train'].values))
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
