import pandas as pd
from sklearn.model_selection import train_test_split
from .image_data import image_data_load, Image_Dataset
from .context_data import context_data_load
from torch.utils.data import DataLoader

def image_context_data_load(args):
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
        data_path : str
            데이터 경로
        process_cat : str
            books 데이터에서 선택할 카테고리 (원본, 상위)
    ----------
    """
    context_data = context_data_load(args)
    image_data = image_data_load(args)
    
    
    train = pd.merge(context_data['train'], image_data['img_train'][['isbn', 'user_id', 'img_vector']], on=['isbn', 'user_id'], how='left')
    test = pd.merge(context_data['test'], image_data['img_test'][['isbn', 'user_id', 'img_vector', 'rating']], on=['isbn', 'user_id'], how='left')
    users = context_data['users']
    books = context_data['books']
    sub = context_data['sub']
    idx2user = context_data['idx2user']
    idx2isbn = context_data['idx2isbn']
    user2idx = context_data['user2idx']
    isbn2idx = context_data['isbn2idx']
    field_dims = context_data['field_dims']
    
    
    data = {
            'train':train,
            'test':test,
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
    
def image_context_data_split(args, data):
    """
    Parameters
    ----------
    Args : argparse.ArgumentParser
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            seed 값을 입력합니다.
    data : Dict
        image_context_data_load로 부터 전처리가 끝난 데이터가 담긴 사전 형식의 데이터를 입력합니다.
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
    
def image_context_data_loader(args, data):
    """
    Parameters
    ----------
    Args : argparse.ArgumentParser
        batch_size : int
            Batch size를 입력합니다.
    data : Dict
        image_context_data_split로 부터 학습/평가/실험 데이터가 담긴 사전 형식의 데이터를 입력합니다.
    ----------
    """
    train_dataset = Image_Dataset(
                                data['X_train'].drop(['img_vector'], axis=1).values,
                                data['X_train']['img_vector'].values,
                                data['y_train'].values
                                )
    valid_dataset = Image_Dataset(
                                data['X_valid'].drop(['img_vector'], axis=1).values,
                                data['X_valid']['img_vector'].values,
                                data['y_valid'].values
                                )
    test_dataset = Image_Dataset(
                                data['test'].drop(['img_vector', 'rating'], axis=1).values,
                                data['test']['img_vector'].values,
                                data['test']['rating'].values
                                )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader
    return data