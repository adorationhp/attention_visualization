#!/usr/bin/env python3
import argparse
from pathlib import Path
from sys import stdin
from typing import Generator, List, NamedTuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=argparse.FileType('r'), default=stdin,
                        help="Sockeye's translation output of type 'translation_with_alignment_matrix'")
    parser.add_argument('--annotate', action='store_true')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory.')
    return parser


TranslationAttention = NamedTuple('TranslationAttention', [('sentence_id', int),
                                                           ('source', str),
                                                           ('target', str),  # TODO: needed?
                                                           ('attention_matrix', np.ndarray)
                                                           ])


def translation_attentions(translation_with_attention_text: List[str]) -> Generator[TranslationAttention, None, None]:
    is_header_line = True
    for line in translation_with_attention_text:
        if is_header_line:
            header = line.split('|||')

            sentence_id = int(header[0])
            source = header[3]
            target = header[1]
            source_length = int(header[4])
            target_length = int(header[5])

            rows = ''
            is_header_line = False
        elif line != '':
            rows = ' '.join([rows, line])
        else:
            # encountered blank line - end of attention matrix
            is_header_line = True  # reset header flag
            attention_matrix = np.fromstring(rows, dtype=float, sep=' ').reshape(source_length, target_length)
            yield TranslationAttention(sentence_id=sentence_id,
                                       source=source,
                                       target=target,
                                       attention_matrix=attention_matrix)


if __name__ == '__main__':
    args = create_parser().parse_args()

    with args.input as f:
        trans_att_text = f.read().splitlines()

    output_directory = Path(args.output)
    output_directory.mkdir(parents=True, exist_ok=True)

    i = 0
    for trans_att in translation_attentions(trans_att_text):
        i += 1
        f, ax = plt.subplots(figsize=trans_att.attention_matrix.shape)

        sns.heatmap(trans_att.attention_matrix.transpose(),
                    annot=args.annotate,
                    fmt=".2f",
                    linewidths=0.5,
                    cmap='Blues',
                    square=True,
                    xticklabels=trans_att.source.split(),
                    yticklabels=trans_att.target.split(),
                    ax=ax)

        ax.xaxis.set_tick_params(labelbottom=False, labeltop=True, bottom=False, labelrotation=40)
        ax.yaxis.set_tick_params(left=False)
        ax.xaxis.set_label_position('top')
        ax.set(xlabel='Source', ylabel='Target')
        f.suptitle('Sentence-ID: {}'.format(trans_att.sentence_id))

        output_file = str(output_directory) + '/sentence-{}.svg'.format(trans_att.sentence_id)
        f.savefig(output_file, bbox_inches='tight', dpi=300)

        plt.close(f)
