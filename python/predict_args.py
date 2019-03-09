import argparse

def get_args():
    
    parser = argparse.ArgumentParser(description = 'submit an image, choose the top k choices to be shown, choose to use gpu or cpu')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--vgg', action = 'count',
                      help = 'enter the model architecture you trained')
    group.add_argument('-a', '--alexnet', action = 'store_true',
                      help = 'enter the model architecture you trained')
    group.add_argument('-d', '--densenet', action = 'count',
                      help = 'enter the model architecture you trained')
    parser.add_argument('-m', '--trained_model', action='store', type=str, default='checkpoint.pth')
    parser.add_argument('-i', '--image_path', action='store', type=str, default='flowers/test/1/image_06743.jpg')
    parser.add_argument('-tk', '--topk', action='store', type=int, default=3)
    parser.add_argument('-out', '--output', action='store', choices=['top', 'topk'], default='top',
                        help='choose between top and topk classes to be printed')
    parser.add_argument('-n', '--category_names', action='store', type=str, default ='cat_to_name.json')
    parser.add_argument('-dev', '--device', action='store', choices=['cpu', 'cuda'], type=str, default='cpu', 
                       help='sets device argument to either cpu or cuda')
    parser.parse_args()
    return parser