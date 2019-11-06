import os, pickle, json
import logging
# from os.path import dirname, basename


def save_file(data, file_path, data_name='data', mode='pickle'):
    logging.info("saving %s to \'%s\'..." % (data_name, file_path))
    if mode == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(obj=data,
                        file=f)
    elif mode == 'json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(obj=data,
                      fp=f)
    else:
        raise(ValueError, 'Function save_file does not have mode %s' % (mode))


def load_file(file_path, data_name='data', mode='pickle'):
    logging.info("trying to load %s from \'%s\'..." % (data_name, file_path))
    data = None
    if os.path.isfile(file_path):
        if mode == 'pickle':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        elif mode == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise (ValueError, 'Function save_file does not have mode %s' % (mode))

    else:
        logging.info('Have not found the file')
    return data


