import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from .image_data import image_data_load
from .text_data import text_data_load
from torch.utils.data import DataLoader, Dataset
from src.utils import get_sampler


class Image_Text_Dataset(Dataset):
    def __init__(self, user_isbn_vector, user_summary_merge_vector, item_summary_vector, img_vector, label):
        """
        Parameters
        ----------
        user_isbn_vector : np.ndarray
            벡터화된 유저와 책 데이터를 입렵합니다.
        user_summary_merge_vector : np.ndarray
            벡터화된 유저에 대한 병합한 요약 정보 데이터 입력합니다.
        item_summary_vector : np.ndarray
            벡터화된 책에 대한 요약 정보 데이터 입력합니다.
        img_vector : np.ndarray
            벡터화된 이미지 데이터를 입력합니다.
        label : np.ndarray
            정답 데이터를 입력합니다.
        ----------
        """
        self.user_isbn_vector = user_isbn_vector
        self.user_summary_merge_vector = user_summary_merge_vector
        self.item_summary_vector = item_summary_vector
        self.img_vector = img_vector
        self.label = label
    def __len__(self):
        return self.user_isbn_vector.shape[0]
    def __getitem__(self, i):
        return {
                'user_isbn_vector' : torch.tensor(self.user_isbn_vector[i], dtype=torch.long),
                'user_summary_merge_vector' : torch.tensor(self.user_summary_merge_vector[i].reshape(-1, 1), dtype=torch.float32),
                'item_summary_vector' : torch.tensor(self.item_summary_vector[i].reshape(-1, 1), dtype=torch.float32),
                'img_vector' : torch.tensor(self.img_vector[i], dtype=torch.float32),
                'label' : torch.tensor(self.label[i], dtype=torch.float32),
                }
        

def image_text_data_load(args):
    
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
        device : str
            학습에 사용할 Device를 입력합니다.
        vector_create : bool
            사전에 텍스트 데이터 벡터화에 대한 여부를 입력합니다.
    ----------
    """
    text_data = text_data_load(args)
    image_data = image_data_load(args)
    users = text_data['users']
    books = text_data['books']
    sub = text_data['sub']
    idx2user = text_data['idx2user']
    idx2isbn = text_data['idx2isbn']
    user2idx = text_data['user2idx']
    isbn2idx = text_data['isbn2idx']
    text_train = text_data['text_train']
    text_test = text_data['text_test']
    img_train = image_data['img_train']
    img_test = image_data['img_test']
    
    train = pd.merge(
        text_train[['isbn', 'user_id', 'user_summary_merge_vector', 'item_summary_vector', 'rating']], 
        img_train[['isbn', 'user_id', 'img_vector']], 
        on=['isbn', 'user_id'], how='inner')
    test = pd.merge(
        text_test[['isbn', 'user_id', 'user_summary_merge_vector', 'item_summary_vector', 'rating']], 
        img_test[['isbn', 'user_id', 'img_vector']], 
        on=['isbn', 'user_id'], how='inner')
    
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
            }

    
    return data
    
def image_text_data_split(args, data):
    """
    Parameters
    ----------
    Args : argparse.ArgumentParser
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            seed 값을 입력합니다.
    data : Dict
        image_text_data_load로 부터 전처리가 끝난 데이터가 담긴 사전 형식의 데이터를 입력합니다.
    ----------
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'][['user_id', 'isbn', 'user_summary_merge_vector', 'item_summary_vector', 'img_vector']],
                                                        data['train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data
    
def image_text_data_loader(args, data):
    """
    Parameters
    ----------
    Args : argparse.ArgumentParser
        batch_size : int
            Batch size를 입력합니다.
    data : Dict
        image_text_data_split로 부터 학습/평가/실험 데이터가 담긴 사전 형식의 데이터를 입력합니다.
    ----------
    """
    whole_dataset = Image_Text_Dataset(
                                data['train'][['user_id', 'isbn']].values,
                                data['train']['user_summary_merge_vector'].values,
                                data['train']['item_summary_vector'].values,
                                data['train']['img_vector'].values,
                                data['train']['rating'].values
                                )
    train_dataset = Image_Text_Dataset(
                                data['X_train'][['user_id', 'isbn']].values,
                                data['X_train']['user_summary_merge_vector'].values,
                                data['X_train']['item_summary_vector'].values,
                                data['X_train']['img_vector'].values,
                                data['y_train'].values
                                )
    valid_dataset = Image_Text_Dataset(
                                data['X_valid'][['user_id', 'isbn']].values,
                                data['X_valid']['user_summary_merge_vector'].values,
                                data['X_valid']['item_summary_vector'].values,
                                data['X_valid']['img_vector'].values,
                                data['y_valid'].values
                                )
    test_dataset = Image_Text_Dataset(
                                data['test'][['user_id', 'isbn']].values,
                                data['test']['user_summary_merge_vector'].values,
                                data['test']['item_summary_vector'].values,
                                data['test']['img_vector'].values,
                                data['test']['rating'].values
                                )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, sampler=get_sampler(args, train_dataset, data['y_train'].values))
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    data['whole_dataset'], data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = whole_dataset, train_dataloader, valid_dataloader, test_dataloader
    return data