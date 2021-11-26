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

	timestamps = None
	"""Timestamps array

	:type: sequence of ``str`s

	Each element is the (plain-text) timestamp of the measurement at the
	corresponding population time — i.e. ``timestamp[i]`` is associsated
	with ``dataset.axes[1][i]``. As such, ``len(timestamps) ==
	len(dataset.axes[1]) == signal.shape[1]``.

	"""

	metadata = None
	"""Metadata

	:type: mapping

	**Valid for ``Datasets`` instantiated from binary files only**

	A nested mapping containing the sections, options and values of the (INI
	formatted) experimental metadata file.

	"""

	# -- Public -------------------------------------------------------------- #

	# Special

	def __init__(
		self,
		signal=None,
		axes=None,
		los=None,
		timestamps=None,
		metadata=None
	):
		self.signal = signal
		if axes is None:
			axes = [[],[],[]]
		self.axes = axes
		if los is None:
			los = []
		self.los = los
		if timestamps is None:
			timestamps = []
		self.timestamps = timestamps
		self.metadata = metadata

	# I/O

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
		self = cls()
		self.metadata = _BinaryMetadata()
		self.metadata.read(path)
		self._read_data(path)
		self._read_t1()
		self._read_λ3(path)
		return self

	def to_binary(self, path):
		"""Write dataset to binary files
		
		:param path: Path to metadata file
		:type path: str

		Write dataset to binary encoded files, as output by 2D
		acquisition software.

		``path`` is the path to the experimental metadata file — the primary
		plain-text file output by the 2D acquisition software. This contains all
		experimental metadata, including the locations of the binary data files
		(``*.bin``).

		e.g.::

			>>> path = "GSBRC_2D_1kHz_4nJ_(0,0,0,0)_01.ini"
			>>> dataset.to_binary(path)

		
		"""
		self._assert()
		metadata = _BinaryMetadata.from_dataset(self)
		with open(path, "wt", newline="\r\n", encoding="utf-8") as file:
			metadata.write(file)
		self._write_data(path, metadata)
		self._write_λ3(path, metadata)

	# Properties

	@property
	def spectral_sensitivity(self):
		return NotImplementedError

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

	# -- Private ------------------------------------------------------------- #

	def _read_data(self, path):
		dirname = os.path.dirname(path)
		t2 = []
		signals = []
		for experiment in self.metadata.experiments:
			t2.append(experiment.getfloat("PopulationTime"))
			self.timestamps.append(experiment.getstr("DateAndTime"))
			with open(
				os.path.join(
					dirname,
					experiment.getstr("File2DSignal")
				),
				"rb"
			) as file:
				data = _BinaryData().read(file)
				self.los.append(data.lo)
				signals.append(data.signal[:,np.newaxis,:])
		self.axes[1] = np.array(t2)
		self.signal = np.concatenate(signals, axis=1)

	def _read_t1(self):
		self.axes[0] = np.linspace(
			self.metadata.header.getfloat("CoherenceTimeBegin"),
			self.metadata.header.getfloat("CoherenceTimeEnd"),
			self.signal.shape[0]
		)

	def _read_λ3(self, path):
		self.axes[2] = np.loadtxt(
			os.path.join(
				os.path.dirname(path),
				self.metadata.header.getstr("FileCalibration")
			)
		)

	def _write_data(self, path, metadata=None):
		if metadata is None:
			metadata = self.metadata
		dirname = os.path.dirname(path)
		os.mkdir(os.path.join(dirname, metadata.datadir(path)))
		for experiment, lo, signal in zip(
			metadata.experiments,
			self.los,
			self.signal.transpose((1,0,2))
		):
			with open(
				os.path.join(
					dirname,
					experiment.getstr("File2DSignal"),
				),
				"wb"
			) as file:
				_BinaryData(lo=lo, signal=signal).write(file)

	def _write_λ3(self, path, metadata=None):
		if metadata is None:
			metadata = self.metadata
		dirname = os.path.dirname(path)
		np.savetxt(
			os.path.join(dirname, metadata.header.getstr("FileCalibration")),
			self.axes[2][:,np.newaxis],
			fmt="%.3f",
			newline="\r\n",
			encoding="utf-8"
		)

	def _assert_axes(self):
		if (
			len(self.axes) != self.signal.ndim or
			any(
				len(axis) != self.signal.shape[i_axis]
				for i_axis, axis in enumerate(self.axes)
			)
		):
			raise ValueError("Axes assertion failed")

	def _assert_los(self):
		if self.los and (
			len(self.los) != self.signal.shape[1] or
			any(
				len(lo) != self.signal.shape[2]
				for lo in self.los
			)
		):
			raise ValueError("LO assertion failed")

	def _assert_timestamps(self):
		if self.timestamps and len(self.timestamps) != self.signal.shape[1]:
			raise ValueError("Timestamp assertion failed")

	def _assert(self):
		self._assert_axes()
		self._assert_los()
		self._assert_timestamps()


class _BinaryMetadata(configparser.ConfigParser):

	# -- Public -------------------------------------------------------------- #

	# I/O

	@classmethod
	def from_dataset(cls, dataset):
		self = cls()
		self["Header"] = dict(dataset.metadata["Header"])
		self.setfloat("Header", "CoherenceTimeBegin", dataset.axes[0][0])
		self.setfloat("Header", "CoherenceTimeEnd", dataset.axes[0][-1])
		self.experiments = (dataset.axes[1], dataset.timestamps)
		return self

	def write(self, file, *args, **kwargs):
		self._update_λ3_path(file)
		self._update_data_paths(file)
		super().write(file, *args, **kwargs)

	def optionxform(self, optionstr):
		return optionstr

	# Converters

	def getstr(self, *args, **kwargs):
		return self.get(*args, **kwargs).strip("\"")

	def setstr(self, section, option, value):
		return self.set(section, option, f"\"{value}\"")

	def setfloat(self, section, option, value):
		return self.set(section, option, f"{value:.3f}")

	# General

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

	@experiments.setter
	def experiments(self, value):
		for i_t2, (t2, timestamp) in enumerate(zip(*value)):
			key = f"Experiment {i_t2:d}"
			self[key] = {}
			self.setstr(key, "DateAndTime", timestamp)
			self.setfloat(key, "PopulationTime", t2)
			self.setstr(key, "File2DSignal", f"Signal and LO {i_t2:d}.bin")

	def datadir(self, path):
		return f"{os.path.splitext(os.path.basename(path))[0]} Data"

	# -- Private ------------------------------------------------------------- #

	def _update_λ3_path(self, file):
		self.setstr(
			"Header",
			"FileCalibration",
			f"{os.path.splitext(os.path.basename(file.name))[0]}.cal"
		)

	def _update_data_paths(self, file):
		datadir = self.datadir(file.name)
		for experiment in self.experiments:
			basename = os.path.basename(
				experiment.getstr("File2DSignal")
			)
			self.setstr(
				experiment.name,
				"File2DSignal",
				os.path.join(datadir, basename)
			)


class _BinaryData:

	lo = None
	signal = None

	_magic_number = b"\x00\x00\x00\rLO and Signal"

	_dtypes = {
		"shape": np.dtype(">i4"),
		"data": np.dtype(">f4")
	}

	# -- Public -------------------------------------------------------------- #

	# Special

	def __init__(self, lo=None, signal=None):
		self.lo = lo
		self.signal = signal

	# I/O

	def read(self, file):
		self._assert_magic_number(file)
		self.lo, self.signal = self._iter_read_array(file)
		self.lo = self.lo.squeeze()
		self._assert_shapes()
		return self

	def write(self, file):
		self._assert_shapes()
		self._write_magic_number(file)
		self._write_array(file, self.lo[np.newaxis,:])
		self._write_array(file, self.signal.squeeze())

	# -- Private ------------------------------------------------------------- #

	def _assert_magic_number(self, file):
		magic_number = self._read_magic_number(file)
		if magic_number != self._magic_number:
			raise ValueError(
				"Magic number assertion failed"
				f" ({magic_number} != {self._magic_number})"
			)

	def _assert_shapes(self):
		if (
			self.signal.ndim != 2 or
			len(self.lo) != self.signal.shape[1]
		):
			raise ValueError("Shape assertion failed")

	def _read_magic_number(self, file):
		return file.read(len(self._magic_number))

	def _read_shape(self, file):
		return tuple(
			np.fromfile(
				file,
				dtype=self._dtypes["shape"],
				count=2							# Shapes always length 2
			)[:]
		)

	def _read_data(self, file, shape):
		return np.fromfile(
			file,
			dtype=self._dtypes["data"],
			count=np.prod(shape)
		).reshape(shape)

	def _read_array(self, file):
		shape = self._read_shape(file)
		return self._read_data(file, shape)

	def _iter_read_array(self, file):
		"""Iterator over data in file"""
		while file.peek(1):
			yield self._read_array(file)

	def _write_magic_number(self, file):
		return file.write(self._magic_number)

	def _ndshape(self, data):
		return np.array(
			data.shape,
			dtype=self._dtypes["shape"]
		)

	def _write_shape(self, file, data):
		return file.write(self._ndshape(data).tobytes())

	def _write_data(self, file, data):
		if data.dtype != self._dtypes["data"]:
			data = data.astype(self._dtypes["data"])
		return file.write(data.tobytes())

	def _write_array(self, file, data):
		self._write_shape(file, data)
		self._write_data(file, data)