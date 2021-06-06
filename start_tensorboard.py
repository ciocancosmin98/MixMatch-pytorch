from tensorboard import program
import sys
import argparse
import os

parser = argparse.ArgumentParser(description='Custom Tensorboard launcher')

parser.add_argument('--dataset-name', default='animals10', type=str, metavar='NAME',
                    help='name of the dataset')
parser.add_argument('--session-id', default=-1, type=int, metavar='ID',
                    help='the id of the session to be visualized')
parser.add_argument('--session-path', default=None, type=str)

args = parser.parse_args()

if args.session_path is None:
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
    session_path = os.path.join('sessions', args.dataset_name, str(_id))
else:
    session_path = args.session_path

tb = program.TensorBoard()
logdir = os.path.join(session_path, 'tensorboard')
tb.configure(argv=[None, '--logdir', logdir])
url = tb.launch()

print('Open in browser:', url)

exit_list = ['exit', 'quit', 'q']

print('Type one of', exit_list, 'to quit')

for line in sys.stdin:
    if line.rstrip() in exit_list:
        break
    
