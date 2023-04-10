import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm


class Image_Dataset(Dataset):
    def __init__(self, user_isbn_vector, img_vector, label):
        """
        Parameters
        ----------
        user_isbn_vector : np.ndarray
            벡터화된 유저와 책 데이터를 입렵합니다.
        img_vector : np.ndarray
            벡터화된 이미지 데이터를 입력합니다.
        label : np.ndarray
            정답 데이터를 입력합니다.
        ----------
        """
        self.user_isbn_vector = user_isbn_vector
        self.img_vector = img_vector
        self.label = label
    def __len__(self):
        return self.user_isbn_vector.shape[0]
    def __getitem__(self, i):
        return {
                'user_isbn_vector' : torch.tensor(self.user_isbn_vector[i], dtype=torch.long),
                'img_vector' : torch.tensor(self.img_vector[i], dtype=torch.float32),
                'label' : torch.tensor(self.label[i], dtype=torch.float32),
                }


def image_vector(path):
    """
    Parameters
    ----------
    path : str
        이미지가 존재하는 경로를 입력합니다.
    ----------
    """
    img = Image.open(path)
    scale = transforms.Resize((32, 32))
    tensor = transforms.ToTensor()
    img_fe = Variable(tensor(scale(img)))
    return img_fe


def process_img_data(df, books, user2idx, isbn2idx, train=False):
    """
    Parameters
    ----------
    df : pd.DataFrame
        기준이 되는 데이터 프레임을 입력합니다.
    books : pd.DataFrame
        책 정보에 대한 데이터 프레임을 입력합니다.
    user2idx : Dict
        각 유저에 대한 index 정보가 있는 사전을 입력합니다.
    isbn2idx : Dict
        각 책에 대한 index 정보가 있는 사전을 입력합니다.
    ----------
    """
    books_ = books.copy()
    books_['isbn'] = books_['isbn'].map(isbn2idx)

    if train == True:
        df_ = df.copy()
    else:
        df_ = df.copy()
        df_['user_id'] = df_['user_id'].map(user2idx)
        df_['isbn'] = df_['isbn'].map(isbn2idx)

    df_ = pd.merge(df_, books_[['isbn', 'img_path']], on='isbn', how='left')
    df_['img_path'] = df_['img_path'].apply(lambda x: 'data/'+x)
    img_vector_df = df_[['img_path']].drop_duplicates().reset_index(drop=True).copy()
    data_box = []
    for idx, path in tqdm(enumerate(sorted(img_vector_df['img_path']))):
        data = image_vector(path)
        if data.size()[0] == 3:
            data_box.append(np.array(data))
        else:
            data_box.append(np.array(data.expand(3, data.size()[1], data.size()[2])))
    img_vector_df['img_vector'] = data_box
    df_ = pd.merge(df_, img_vector_df, on='img_path', how='left')
    return df_


def image_data_load(args):
    """
    Parameters
    ----------
    Args : argparse.ArgumentParser
        data_path : str
            데이터가 존재하는 경로를 입력합니다.
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            seed 값을 입력합니다.
        batch_size : int
            Batch size를 입력합니다.
    ----------
    """
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

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)

    img_train = process_img_data(train, books, user2idx, isbn2idx, train=True)
    img_test = process_img_data(test, books, user2idx, isbn2idx, train=False)

    data = {
            'train':train,
            'test':test,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            'img_train':img_train,
            'img_test':img_test,
            }
    return data


def image_data_split(args, data):
    """
    Parameters
    ----------
    Args : argparse.ArgumentParser
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            seed 값을 입력합니다.
    data : Dict
        image_data_load로 부터 전처리가 끝난 데이터가 담긴 사전 형식의 데이터를 입력합니다.
    ----------
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['img_train'][['user_id', 'isbn', 'img_vector']],
                                                        data['img_train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data


def image_data_loader(args, data):
    """
    Parameters
    ----------
    Args : argparse.ArgumentParser
        batch_size : int
            Batch size를 입력합니다.
    data : Dict
        image_data_split로 부터 학습/평가/실험 데이터가 담긴 사전 형식의 데이터를 입력합니다.
    ----------
    """
    train_dataset = Image_Dataset(
                                data['X_train'][['user_id', 'isbn']].values,
                                data['X_train']['img_vector'].values,
                                data['y_train'].values
                                )
    valid_dataset = Image_Dataset(
                                data['X_valid'][['user_id', 'isbn']].values,
                                data['X_valid']['img_vector'].values,
                                data['y_valid'].values
                                )
    test_dataset = Image_Dataset(
                                data['img_test'][['user_id', 'isbn']].values,
                                data['img_test']['img_vector'].values,
                                data['img_test']['rating'].values
                                )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader
    return data
