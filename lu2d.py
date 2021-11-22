"""Analysis tools for 2D spectroscopy at Chemical Physics, Lund University"""
# stdlib imports
import configparser
import os.path

# External imports
import numpy as np


class Dataset:
	"""Multi-dimensional coherent spectroscopy dataset"""

	signal = None
	"""Dataset signal array

	:type: ``numpy.ndarray``

	An ``ndarray`` holding the dataset signal (dependent variable)

	"""

	axes = None
	"""Dataset axes array

	:type: sequence of ``numpy.ndarray``s

	Each ``ndarray`` holds the dataset axis (independent variable) for the
	corresponding signal array dimension. As such, ``len(axis) == signal.ndim``
	and ``len(axis[i]) == signal.shape[i]``.

	"""

	los = None
	"""Local oscillator arrays

	:type: sequence of ``numpy.ndarray``s

	Each ``ndarray`` holds the local oscillator intensity  (dependent variable)
	for the measurement at the corresponding population time — i.e. ``lo[i]`` is
	associsated with ``dataset.axes[1][i]``. As such, ``len(los) ==
	len(dataset.axes[1]) == signal.shape[1]``.

	The independent variable is the detection wavelenth, ``dataset.axis[2]``. As
	such, ``len(los[i]) == len(dataset.axis[2])``

	"""

	metadata = None
	"""Metadata

	:type: mapping

	**Valid for ``Datasets`` instantiated from binary files only**

	A nested mapping containing the sections, options and values of the (INI
	formatted) experimental metadata file.

	"""

	def __init__(
		self,
		signal=None,
		axes=None,
		los=None,
		metadata=None
	):
		self.signal = signal
		self.axes = axes
		self.los = los
		self.metadata = metadata

	@classmethod
	def from_text(cls, fmtstrs, t2):
		"""Instantiate dataset from plain-text files

		:param fmtstrs: Format strings for the real and imaginary filepaths
		:type fmtstrs:  dict
		:param t2: Population time array
		:type t2: sequence of ints

		Instantiate dataset from multiple plain-text encoded files, as output by
		2D analysis software.

		``fmtstrs`` is dict with 'real' and 'imag' keys containing format
		strings for the real and imaginary filepaths. Each format string should
		contain a 't2' key designating the population time.

		e.g.::

			>>> fmtstrs = {
				'real': '2D_Rephasing_Real_{t2}fs.txt',
				'imag': '2D_Rephasing_Imaginary_{t2}fs.txt'
			}
			>>> t2 = [t2 for t2 in range(0, 525, 25)]
			>>> dataset = Dataset.from_txt(fmtstrs, t2)

		"""

		# Read first file
		buffer = np.loadtxt(
			fmtstrs["real"].format(t2=t2[0]),
			dtype=np.float_
		)

		# Allocate arrays
		w1 = buffer[0, 1:]
		t2 = np.array(t2, dtype=np.float_)
		w3 = buffer[1:, 0]
		axes = [w1, t2, w3]
		signal = np.empty(
			(
				buffer.shape[1] - 1,
				len(t2),
				buffer.shape[0] - 1
			),
			np.complex_
		)

		# Populate arrays from files
		for i_t2, val_t2 in enumerate(t2):
			real = np.loadtxt(
				fmtstrs["real"].format(t2=int(val_t2)),
				dtype=np.float_
			)[1:, 1:].T
			imag = 1j * np.loadtxt(
				fmtstrs["imag"].format(t2=int(val_t2)),
				dtype=np.float_
			)[1:, 1:].T
			signal[:, i_t2, :] = (real + imag)
		return cls(signal=signal, axes=axes)

	def to_text(self, fmtstrs):
		"""Write dataset to plain-text files

		:param fmtstrs: Format strings for the real and imaginary filepaths
		:type fmtstrs:  dict

		Write data to multiple plain-text encoded files, as output by 2D
		analysis software.

		``fmtstrs`` is dict with 'real' and 'imag' keys containing format
		strings for the real and imaginary filepaths. Each format string should
		contain a 't2' key designating the population time.

		e.g.::

			>>> fmtstrs = {
				'real': '2D_Rephasing_Real_{t2}fs.txt',
				'imag': '2D_Rephasing_Imaginary_{t2}fs.txt'
			}
			>>> Dataset.write_txt(fmtstrs)

		"""

		for i_t2, t2 in enumerate(self.axes[1]):
			for part in ("real", "imag"):
				path = fmtstrs[part].format(t2=t2)
				row_0 = np.concatenate(
					(
						(0,),						# row 0, col 0
						self.axes[0]
					),
					axis=0
				)[np.newaxis, :]
				col_0 = self.axes[2][:, np.newaxis]
				signal = getattr(self.signal, part)[:, i_t2, :].T
				array = np.concatenate(
					(
						row_0,
						np.concatenate((col_0, signal), axis=1)
					),
					axis=0
				)
				np.savetxt(
					path,
					array,
					fmt="%1.4E",
					delimiter="\t",
					newline="\r\n"
				)

	@classmethod
	def from_binary(cls, path):
		"""Instatiate dataset from binary files

		:param path: Path to metadata file
		:type path: str

		Instantiate dataset from binary encoded files, as output by 2D
		acquisition software.

		``path`` is the path to the experimental metadata file — the primary
		plain-text file output by the 2D acquisition software. This contains all
		experimental metadata, including the locations of the binary data files
		(``*.bin``).

		e.g.::

			>>> path = "GSBRC_2D_1kHz_4nJ_(0,0,0,0)_01.ini"
			>>> dataset = Dataset.from_binary(path)

		"""

		# Read metadata file
		metadata = _BinaryMetadata(path)
		t2 = []
		los = []
		signals = []
		for experiment in metadata.experiments:
			t2.append(experiment.getfloat("PopulationTime"))
			with open(experiment.getpath("File2DSignal"), "rb") as file:
				data = _BinaryData(file)
				los.append(data.lo.squeeze())
				signals.append(data.signal[:,np.newaxis,:])
		t1 = np.linspace(
			metadata.header.getfloat("CoherenceTimeBegin"),
			metadata.header.getfloat("CoherenceTimeEnd"),
			signals[0].shape[0]
		)
		t2 = np.array(t2)
		w3 = np.loadtxt(metadata.header.getpath("FileCalibration"))
		axes = [ t1, t2, w3 ]
		signal = np.concatenate(signals, axis=1)
		return cls(
			signal=signal,
			axes=axes,
			los=los,
			metadata=metadata
		)

	@property
	def spectral_sensitivity(self):
		return NotImplementedError

	@property
	def timestamps(self):
		"""Local oscillator arrays

		:rtype: sequence of ``str`s

		Each element is the (plain-text) timestamp of the measurement at the
		corresponding population time — i.e. ``timestamp[i]`` is associsated
		with ``dataset.axes[1][i]``. As such, ``len(timestamps) ==
		len(dataset.axes[1]) == signal.shape[1]``.

		"""
		return [
			experiment.getstr("DateAndTime")
			for experiment in self.metadata.experiments
		]

	@property
	def excitation_energy(self):
		"""Excitation energy

		:rtype: float

		"""
		return (
			self.metadata.header.getfloat("ExcitationEnergy")
			if self.metadata
			else None
		)

	@property
	def repetition_rate(self):
		"""Excitation repetition rate

		:rtype: float

		"""
		return (
			self.metadata.header.getfloat("RepetitionRate")
			if self.metadata
			else None
		)

	@property
	def polarizations(self):
		"""Pulse polarizations

		:rtype: sequence of ``float``s

		A sequence whose elements correspond to the incident pulse polarizations
		— i.e. ``polarizations[0]`` is the polarization of the  first pulse. As
		such, ``len(polarizations) == 4``

		"""
		return (
			(
				self.metadata.header.getfloat("Polarizer1"),
				self.metadata.header.getfloat("Polarizer2"),
				self.metadata.header.getfloat("Polarizer3"),
				self.metadata.header.getfloat("Polarizer4")
			)
			if self.metadata
			else None
		)

	@property
	def fs_thickness(self):
		"""Fused silica thickness

		:rtype: float

		Thickness of fused silica between pulses 3 and 4

		"""
		return (
			self.metadata.header.getfloat("FSThicknessBetween3and4")
			if self.metadata
			else None
		)


class _BinaryMetadata(configparser.ConfigParser):

	dirname = None

	def __init__(self, path, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.read(path)
		self.dirname = os.path.dirname(path)

	def getstr(self, *args, **kwargs):
		return self.get(*args, **kwargs).strip("\"")

	def getpath(self, *args, **kwargs):
		return os.path.join(
			self.dirname,
			self.getstr(*args, **kwargs)
		)

	@property
	def header(self):
		return self["Header"]

	def experiment(self, n):
		return self[f"Experiment {n:d}"]

	@property
	def experiments(self):
		for name, section in self.items():
			if "Experiment" in name:
				yield section


class _BinaryData:

	file = None
	magic_number = b"\x00\x00\x00\rLO and Signal"

	lo = None
	signal = None

	def __init__(self, file):
		self.file = file
		self._validate()
		self.lo, self.signal = self._data()

	def _validate(self):
		"""Validate file

		Validations:

		* Magic number

		"""
		magic_number = self.file.read(len(self.magic_number))
		if magic_number != self.magic_number:
			raise ValueError(
				"Invalid magic number"
				f" ({magic_number} != {self.magic_number})"
			)

	def _read_shape(self):
		return tuple(
			np.fromfile(
				self.file,
				dtype=np.dtype(">i4"),
				count=2							# Shapes always length 2
			)[:]
		)

	def _read_data(self, shape):
		return np.fromfile(
			self.file,
			dtype=np.dtype(">f4"),
			count=np.prod(shape)
		).reshape(shape)

	def _data(self):
		"""Iterator over data in file"""
		while self.file.peek(1):
			shape = self._read_shape()
			data = self._read_data(shape)
			yield data
