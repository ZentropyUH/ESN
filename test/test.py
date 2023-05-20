import time
import argparse
import json


def main(path:str, input:str):

    print('Hello World!')
    print('Path: ', path)
    print('Input: ', input)

    time.sleep(10)

    with open(path + '/output.json', 'w') as outfile:
        json.dump({x: x**2 for x in range(10)}, outfile)

    print('JSON saved!')


def m():
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-p', '--path', type=str, help='Path to save the output')
    parser.add_argument('-i', '--input', type=int, help='Path to input file')

    args = parser.parse_args()
    main(args.path, args.input)
