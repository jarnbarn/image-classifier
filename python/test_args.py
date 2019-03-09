# defines arguments to be used in test.py program

import argparse

def get_args():

    parser = argparse.ArgumentParser(description = 'enter args to test trained network')


    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--vgg', action = 'count',
                      help = 'repeat arg up to 4 times to increase hidden layers')
    group.add_argument('-a', '--alexnet', action = 'store_true')
    group.add_argument('-d', '--densenet', action = 'count',
                      help = 'repeat arg up to 4 times to increase hidden layers')

    parser.add_argument('-dev', '--device', action='store', choices=['cpu', 'cuda'], type=str, default='cpu',
                       help='sets device argument to either cpu or cuda')

    parser.parse_args()
    return parser
