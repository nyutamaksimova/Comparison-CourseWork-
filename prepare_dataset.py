import json
import argparse
from pathlib import Path


def main(args):
    data = dict()

    for file in Path(args.work_programs).glob('**/*.json'):
        print(f'process file: {file}')

        with open(file, errors='ignore', encoding='utf-8') as fp:
            program = json.load(fp)
            text = ''

            for key, value in program.items():
                text += value['title']
                text += value['text']

            data['text'] = list(text.split())

        with open("data/prepared_data/" + fp.name[19:-5] + "_new.json", 'w', encoding='utf-8') as fp:
            json.dump(data, fp, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare Dataset')

    parser.add_argument('--work-programs', type=str, default='data/work_programs')
    parser.add_argument('--dataset', type=str, default='data/dataset.json')

    args = parser.parse_args()

    main(args)