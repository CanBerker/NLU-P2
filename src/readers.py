import csv
import numpy as np


class CsvReader(object):
    def __init__(self, filename):
        self.filename = filename

    def read(self):
        """Reads the csv data input and returns a 2-dimensional numpy array representation of the csv file,
        excluding the header line. """
        with open(self.filename, 'r', encoding='utf-8') as f:
            csv_read = csv.reader(f)
            # Remove the csv header
            next(csv_read)
            return np.array([np.array(line) for line in csv_read])


class TrainReader(CsvReader):
    pass


class ValidationReader(CsvReader):
    pass


class TestReader(CsvReader):
    pass
