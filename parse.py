import argparse

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--result_dir', type=str, default='result', help='result directory')
    parser.add_argument('--model_dir', type=str, default='saved_models', help='result directory')
    parser.add_argument('--test_dir', type=str, default='test_result')
    # data location
    data_args = parser.add_argument_group('Dataset options')

    data_args.add_argument('--dataset_name', type=str, default='dataset', help='a TextData object')

    data_args.add_argument('--dataset', type=str, default='rotten')

    data_args.add_argument('--data_dir', type=str, default='data', help='dataset directory, save pkl here')
    data_args.add_argument('--embedding_file', type=str, default='glove.840B.300d.txt')
    data_args.add_argument('--vocab_size', type=int, default=-1, help='vocab size, use the most frequent words')

    # data_args.add_argument('--max_length', type=int, default=1000, help='max length of samples')
    data_args.add_argument('--ev_max_length', type=int, default=200, help='max length of evidence')
    data_args.add_argument('--data_size', type=int, default=1000, help='number of subdirs to include')

    # only valid when using rotten
    data_args.add_argument('--train_file', type=str, default='train.txt')
    data_args.add_argument('--val_file', type=str, default='val.txt')
    data_args.add_argument('--test_file', type=str, default='test.txt')

    # neural network options
    nn_args = parser.add_argument_group('Network options')
    nn_args.add_argument('--embedding_size', type=int, default=300)
    nn_args.add_argument('--hidden_size', type=int, default=200)
    nn_args.add_argument('--gen_layers', type=int, default=1)
    nn_args.add_argument('--gen_bidirectional', action='store_true', default=True)
    nn_args.add_argument('--naive', action='store_true', help='use a naive model')

    nn_args.add_argument('--max_steps', type=int, default=1000, help='number of steps in RNN')
    nn_args.add_argument('--n_classes', type=int, default=2)
    nn_args.add_argument('--dependent', action='store_true',
                         help='two kinds of rationales, only independent is supported at the moment')
    nn_args.add_argument('--r_unit', type=str, default='lstm', choices=['lstm', 'rcnn'],
                         help='only support lstm at the moment')

    # training options
    trainingArgs = parser.add_argument_group('Training options')
    trainingArgs.add_argument('--rl', action='store_true', help='whether or not to use REINFORCE algorithm')
    trainingArgs.add_argument('--pre_embedding', action='store_true', default=True)
    trainingArgs.add_argument('--elmo', action='store_true')
    trainingArgs.add_argument('--train_elmo', action='store_true')
    trainingArgs.add_argument('--drop_out', type=float, default=1.0, help='dropout rate for RNN (keep prob)')
    trainingArgs.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    trainingArgs.add_argument('--weight_decay', type=float, default=0., help='weight_decay rate')

    # using different batch_size for training and evaluation
    trainingArgs.add_argument('--batch_size', type=int, default=32, help='batch size')
    trainingArgs.add_argument('--test_batch_size', type=int, default=4, help='test batch size')

    trainingArgs.add_argument('--epochs', type=int, default=100, help='most training epochs')
    trainingArgs.add_argument('--device', type=int, default=0, help='most training epochs')
    trainingArgs.add_argument('--load_model', action='store_true', help='whether or not to use old models')
    trainingArgs.add_argument('--theta', type=float, default=0.1, help='for #choices')
    trainingArgs.add_argument('--gamma', type=float, default=0.1, help='for continuity')
    trainingArgs.add_argument('--temperature', type=float, default=1., help='gumbel softmax temperature')
    trainingArgs.add_argument('--threshold', type=float, default=0.5, help='threshold for producing hard mask')
    trainingArgs.add_argument('--test_model', action='store_true')

    trainingArgs.add_argument('--train_bert', action='store_true')
    trainingArgs.add_argument('--train_probing', action='store_true')
    trainingArgs.add_argument('--evaluate_probing', action='store_true')
    trainingArgs.add_argument('--generate_probing', action='store_true')
    trainingArgs.add_argument('--evaluate_attention', action='store_true')
    trainingArgs.add_argument('--evaluate_gradient', action='store_true')
    trainingArgs.add_argument('--evaluate_true', action='store_true')
    trainingArgs.add_argument('--evaluate_lime', action='store_true')
    trainingArgs.add_argument('--train_stack', action='store_true')

    data_args.add_argument('--model_path', type=str, default='none', help='model directory, save pkl here')
    parser.add_argument('--evaluate_method', type=str, default='gradient', help='explaining method')
    trainingArgs.add_argument('--max_length', type=int, default=400, help='most training epochs')
    trainingArgs.add_argument('--seed', type=int, default=100, help='most training epochs')
    trainingArgs.add_argument('--sample', type=int, default=100, help='most training epochs')
    trainingArgs.add_argument('--drop_sample', type=int, default=0, help='most training epochs')

    return parser.parse_args(args)
