import argparse
from torchvision import datasets
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Download MNIST data')
    parser.add_argument('--data-dir', type=str, default='.', metavar='dir',
                        help='Data directory that stores the MNIST dataset')
    args = parser.parse_args()

    MNIST_DATA = args.data_dir
   
#     transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#         ])

    MNIST_DATA = args.data_dir
    datasets.MNIST(MNIST_DATA, download=True)

#     train_dset = datasets.MNIST(MNIST_DATA, train=True, download=True,
#                        transform=transform)
#     test_dset = datasets.MNIST(MNIST_DATA, train=False,
#                        transform=transform)
    
if __name__ == '__main__':
    main()
