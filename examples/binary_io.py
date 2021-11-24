#!/usr/bin/env python3
import lu2d
import numpy as np


def dataset_from_binary():
	"""Example of reading a dataset from binary files"""

	# Read dataset
	path = "data/binary/demo.ini"
	dataset = lu2d.Dataset.from_binary(path)

	# Signal
	signal = dataset.signal

	# Axes
	t1, t2, w3 = dataset.axes
	assert len(dataset.axes) == dataset.signal.ndim
	assert len(t1) == signal.shape[0]
	assert len(t2) == signal.shape[1]
	assert len(w3) == signal.shape[2]

	# Local ocillators
	los = dataset.los
	assert len(los) == len(t2)
	for lo in los:
		assert len(lo) == len(w3)

	# Timestamps
	timestamps = dataset.timestamps
	assert len(timestamps) == len(t2)

	# Excitation energy
	excitation_energy = dataset.excitation_energy

	# Repetition rate
	repetition_rate = dataset.repetition_rate

	# Polarizations
	polarizations = dataset.polarizations
	assert len(polarizations) == 4

	# FS thickness
	fs_thickness = dataset.fs_thickness

	# All metadata
	scans_per_point = dataset.metadata["Header"]["ScansPerPoint"]
	experiment_2_path = dataset.metadata["Experiment 2"]["File2DSignal"]

	return dataset

def dataset_to_binary():
	"""Example of writing a dataset to binary files"""

	# Instantiate dataset
	dataset = dataset_from_binary()
	dataset.signal = dataset.signal[2:-2,:-1,:]
	dataset.axes[0] = dataset.axes[0][2:-2]
	dataset.axes[1] = dataset.axes[1][:-1]
	dataset.los.pop()
	dataset.timestamps.pop()

	# Write dataset
	path = "data/binary/write.ini"
	dataset.to_binary(path)

if __name__ == "__main__":
	dataset_from_binary()
	dataset_to_binary()
