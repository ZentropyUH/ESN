import time
import argparse



def main(path:str):
    print('Starting...')
    time.sleep(60)
    print('Finished!')


    import json


    a = {x: x**2 for x in range(10)}


    with open(path + '/output.json', 'w') as outfile:
        json.dump(a, outfile)

    print('JSON saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-p', '--path', type=str, help='Path to save the output')
    args = parser.parse_args()
    main(args.path)

