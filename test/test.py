import time
import argparse


def main(a):

    print('Hello World!')
    print(a)
    time.sleep(10)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-a', type=str, help='Path to save the output')

    args = parser.parse_args()
    main(args.a)
