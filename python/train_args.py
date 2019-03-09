import argparse

def get_args():
    
    parser = argparse.ArgumentParser(description = 'choose type of architecture, choose learning rate, number of hidden units, and epochs')
    
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--vgg', action ='count',
                      help ='repeat arg up to 4 times to increase hidden layers')
    group.add_argument('-a', '--alexnet', action ='store_true')
    group.add_argument('-d', '--densenet', action ='count',
                      help ='repeat arg up to 4 times to increase hidden layers')

    parser.add_argument('-l', '--learning_rate', action='store', type=float, default=0.01)
    parser.add_argument('-e', '--epochs', action='store', type=int, default=4)
    parser.add_argument('-dev', '--device', action='store', choices=['cpu', 'cuda'], type=str, default='cpu', 
                       help='sets device argument to either cpu or cuda')
    
    
    parser.parse_args()
    return parser


