#!/usr/bin/env python3
import lu2d
import numpy as np


def dataset_from_text():
	"""Example of reading a dataset from plain-text files"""

	# Read dataset
	fmtstrs = {
		"real": "data/text/demo_Real_{t2}fs.txt",
		"imag": "data/text/demo_Imaginary_{t2}fs.txt"
	}
	t2 = [0, 100, 1000, 900000]
	dataset = lu2d.Dataset.from_text(fmtstrs, t2)

	# Signal
	signal = dataset.signal

	# Axes
	t1, t2, w3 = dataset.axes
	assert len(dataset.axes) == dataset.signal.ndim
	assert len(t1) == signal.shape[0]
	assert len(t2) == signal.shape[1]
	assert len(w3) == signal.shape[2]

	return dataset

def dataset_to_text():
	"""Example of writing a dataset to plain-text files"""

	# Example dataset
	dataset = dataset_from_text()

	# File specification
	fmtstrs = {
		"real": "data/text/write_Real_{t2}fs.txt",
		"imag": "data/text/write_Imaginary_{t2}fs.txt"
	}

	# Write dataset
	dataset.to_text(fmtstrs)


if __name__ == "__main__":
	dataset_from_text()
	dataset_to_text()
