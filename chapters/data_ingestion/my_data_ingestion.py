#!/use/bin/env python3
# encoding: utf-8

"""My module to convert the csv to tfrecords."""
import csv
from typing import Dict, List, Any, Callable, Iterator

import tensorflow as tf

original_data_file: str = '../../data/consumer_complaints_with_narrative.csv'
tfrecords_filename: str = 'consumer-complaints.tfrecords'


def convert_row(raw_row: Dict[str, Any]) -> bytes:
    """Convert each row to an example."""

    def clean_rows(inner_row: Dict[str, Any]) -> Dict[str, Any]:
        """Add the sentinel value 99999 if no zip code exists in the row."""
        if not inner_row["zip_code"]:
            inner_row["zip_code"] = "99999"
        return inner_row

    def byte_feature_eng(value: str) -> tf.train.Feature:
        """Convert a string to byte feature."""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value.encode()]))

    def numeric_feature_eng(value: str) -> tf.train.Feature:
        """Convert a value to numeric feature, only for zipcode."""

        def convert_zipcode_to_int(zipcode: str) -> int:
            """Convert zipcode"""
            if isinstance(zipcode, str) and "XX" in zipcode:
                zipcode = zipcode.replace("XX", "00")
            return int(zipcode)

        return tf.train.Feature(int64_list=tf.train.Int64List(
            value=[convert_zipcode_to_int(value)]))

    bytes_features: List[str] = ['product', 'sub_product', 'issue',
                                 'sub_issue',
                                 'state', 'company', 'company_response',
                                 'timely_response',
                                 'consumer_disputed']
    numeric_features: List[str] = ['zip_code']

    callable_mapper: Dict[str, Callable] = {key: byte_feature_eng for key in
                                            bytes_features}
    callable_mapper.update(
        {key: numeric_feature_eng for key in numeric_features})
    processed_row: Dict[str, Any] = clean_rows(raw_row)
    feature: Dict[str, tf.train.Feature] = {fea: mapper(processed_row[fea])
                                            for fea, mapper in
                                            callable_mapper.items()}
    return tf.train.Example(
        features=tf.train.Features(feature=feature)).SerializeToString()


if __name__ == '__main__':
    with (open(file=original_data_file) as csv_file,
          tf.io.TFRecordWriter(path=tfrecords_filename) as writer):
        reader: Iterator[Dict[str, Any]] = csv.DictReader(f=csv_file,
                                                          delimiter=',',
                                                          quotechar='"')
        _ = [*map(lambda item: writer.write(record=item),
                  map(convert_row, reader))]
