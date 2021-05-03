import sys
import argparse
import os
from torch import load


short_consts   = ['dataset_name', 'session_id', 'enable_mixmatch', 'n_labeled']
verbose_consts = ['epochs', 'train_iteration', 'batch_size', 'lr', 'lambda_u', 'T', 'alpha']

short_vars = ['epoch', 'acc', 'best_acc']
verbose_vars = []

LABEL_LENGTH = 30


parser = argparse.ArgumentParser(description='Session information tool')

parser.add_argument('--dataset-name', default='animals10', type=str, metavar='NAME',
                    help='name of the dataset')
parser.add_argument('--session-id', default=-1, type=int, metavar='ID',
                    help='the id of the session whose information will be printed')
parser.add_argument('--verbose', '-v', action="store_true",
                    help='show more information')

args = parser.parse_args()

if args.session_id == -1:
    max_id = -1
    sessions_dir = os.path.join('sessions', args.dataset_name)
    for fname in os.listdir(sessions_dir):
        try:
            _id = int(fname)

            if _id >= max_id:
                max_id = _id
        except ValueError:
            pass

    if max_id == -1:
        raise Exception("Could not find any sessions using the dataset " + \
            args.dataset_name + ".")
    _id = max_id
else:
    session_dir = os.path.join('sessions', args.dataset_name, str(args.session_id))
    if not os.path.exists(session_dir):
        raise Exception("Could not find session " + str(args.session_id) + \
            " using the dataset " + args.dataset_name + ".")
    _id = args.session_id

const_path = os.path.join('sessions', args.dataset_name, str(_id), 'const.pth.tar')
if not os.path.exists(const_path):
    raise Exception("Session exists but is not properly initialized: missing constants")
    
checkpoint_path = os.path.join('sessions', args.dataset_name, str(_id), 'training', 'checkpoint.pth.tar')
if not os.path.exists(checkpoint_path):
    raise Exception("Session exists but hasn't saved a checkpoint yet")


def print_list(list_name, short_list, verbose_list, dictionary):
    print(' ' + (list_name + '\'S LABEL').ljust(LABEL_LENGTH) + list_name + '\'S VALUE')
    print('-' * (LABEL_LENGTH * 2))

    def print_value(name, active):
        if not active:
            return

        value = dictionary[name]
        if not isinstance(value, float):
            str_value = str(value)
        else:
            str_value = '%.3f' % value

        line = ' ' + name.ljust(LABEL_LENGTH) + str_value
        print(line)

    for name in short_list:
        print_value(name, True)

    for name in verbose_list:
        print_value(name, args.verbose)

constants = load(const_path)
constants['session_id'] = _id
print_list('CONSTANT', short_consts, verbose_consts, constants)

variables = load(checkpoint_path)
print(''); print('')
print_list('VARIABLE', short_vars, verbose_vars, variables)