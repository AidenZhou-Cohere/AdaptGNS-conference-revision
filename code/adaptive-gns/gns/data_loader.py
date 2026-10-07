import torch
import numpy as np
import hashlib
import json
from pathlib import Path
import re
from collections.abc import Sequence


def resolve_dataset_path(path):
    """Allow existing train.npz call sites to use a published train.json manifest."""
    path = Path(path)
    if not path.exists() and path.suffix == ".npz" and path.with_suffix(".json").exists():
        return path.with_suffix(".json")
    return path


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_numeric_trajectory(positions, particle_types):
    if (positions.ndim != 3 or positions.shape[0] < 2 or positions.shape[1] < 1
            or positions.shape[2] not in (2, 3) or not np.issubdtype(positions.dtype, np.floating)):
        raise ValueError("Positions must be floating [time>=2, particles>=1, dim=2 or 3]")
    types = np.asarray(particle_types)
    if not np.issubdtype(types.dtype, np.integer):
        raise ValueError("Particle types must have integer dtype")
    if types.ndim > 1 or (types.ndim == 1 and len(types) != positions.shape[1]):
        raise ValueError("Particle types must be scalar or one per particle")


def _manifest_array_path(root, description):
    relative = Path(description["path"])
    if relative.is_absolute():
        raise ValueError("Manifest array paths must be relative")
    file_path = (root / relative).resolve()
    if not file_path.is_relative_to(root) or file_path.suffix != ".npy":
        raise ValueError("Manifest array path escapes dataset root or is not .npy")
    return file_path


def _manifest_array(root, description):
    array = np.load(_manifest_array_path(root, description), mmap_mode="r", allow_pickle=False)
    if list(array.shape) != description["shape"] or array.dtype != np.dtype(description["dtype"]):
        raise ValueError(f"Array shape/dtype differs from manifest: {description['path']}")
    return array


class ManifestTrajectories(Sequence):
    """Lazy indexable trajectories; no thousands-of-open-files requirement.

    Each indexed tuple owns its read-only memmaps until the caller releases it.
    Shape scans and random minibatches need only their currently used records.
    """
    def __init__(self, root, manifest):
        self.root = root
        self.manifest = manifest
        self.trajectory_ids = [record["id"] for record in manifest["records"]]

    def __len__(self):
        return len(self.manifest["records"])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        record = self.manifest["records"][index]
        return (_manifest_array(self.root, record["positions"]),
                _manifest_array(self.root, record["particle_types"]))


def load_manifest_data(path, verify_hashes=False):
    """Open safe read-only .npy memmaps without materializing all trajectories.

    Path/schema/shape/dtype checks always run. Set verify_hashes=True to stream
    every position/type file and metadata through SHA256 at startup or resume.
    Legacy pickle archives are never used on this path.
    """
    path = Path(path)
    root = path.parent.resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("format") != "gns-trajectory-manifest" or manifest.get("version") != 1:
        raise ValueError("Unsupported trajectory manifest schema")
    records = manifest.get("records", [])
    if not records or manifest.get("record_count") != len(records):
        raise ValueError("Manifest must contain its declared nonempty record count")
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate trajectory IDs in manifest")
    if verify_hashes and _file_sha256(root / "metadata.json") != manifest["metadata_sha256"]:
        raise ValueError("Dataset metadata SHA256 mismatch")
    def load_array(description):
        file_path = _manifest_array_path(root, description)
        if verify_hashes and _file_sha256(file_path) != description["sha256"]:
            raise ValueError(f"Array SHA256 mismatch: {description['path']}")
        return _manifest_array(root, description)
    for record in records:
        positions = load_array(record["positions"])
        particle_types = load_array(record["particle_types"])
        _validate_numeric_trajectory(positions, particle_types)
        del positions, particle_types
    return ManifestTrajectories(root, manifest)


def load_npz_data(path, verify_hashes=False):
    """Load data stored in npz format.

    The file format for Python 3.9 or less supports ragged arrays and Python 3.10
    requires a structured array. This function supports both formats.

    Args:
        path (str): Path to npz file.

    Returns:
        data (list): List of tuples of the form (positions, particle_type).
    """
    path = resolve_dataset_path(path)
    if path.suffix == ".json":
        return load_manifest_data(path, verify_hashes=verify_hashes)
    # The new numeric NPZ format needs no pickle and is useful for small subsets.
    with np.load(path, allow_pickle=False) as data_file:
        position_keys = [key for key in data_file.files if key.startswith("position_")]
        if position_keys:
            if any(re.fullmatch(r"position_\d+", key) is None for key in position_keys):
                raise ValueError("Invalid numeric trajectory key")
            position_keys.sort(key=lambda key: int(key.split("_")[-1]))
            expected_keys = {key for position in position_keys
                             for key in (position, position.replace("position_", "type_", 1))}
            if set(data_file.files) != expected_keys:
                raise ValueError("Numeric NPZ requires exactly paired position_N/type_N arrays")
            data = []
            for key in position_keys:
                positions = data_file[key]
                particle_types = data_file[key.replace("position_", "type_", 1)]
                _validate_numeric_trajectory(positions, particle_types)
                data.append((positions, particle_types))
            return data
    # Backwards compatibility for trusted legacy object archives only.
    with np.load(path, allow_pickle=True) as data_file:
        if 'gns_data' in data_file:
            data = data_file['gns_data']
        else:
            data = [item for _, item in data_file.items()]
    if not len(data):
        raise ValueError("Dataset is empty")
    return data


class SamplesDataset(torch.utils.data.Dataset):
    """Dataset of samples of trajectories.
    
    Each sample is a tuple of the form (positions, particle_type).
    positions is a numpy array of shape (sequence_length, n_particles, dimension).
    particle_type is an integer.

    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.

    Attributes:
        _data (list): List of tuples of the form (positions, particle_type).
        _dimension (int): Dimension of the data.
        _input_length_sequence (int): Length of input sequence.
        _data_lengths (list): List of lengths of trajectories in the dataset.
        _length (int): Total number of samples in the dataset.
        _precompute_cumlengths (np.array): Precomputed cumulative lengths of trajectories in the dataset.
    """

    def __init__(self, path, input_length_sequence, verify_hashes=False):
        super().__init__()
        # load dataset stored in npz format
        # data is loaded as dict of tuples
        # of the form (positions, particle_type)
        # convert to list of tuples
        # TODO: allow_pickle=True is potential security risk. See docs.
        self._data = load_npz_data(path, verify_hashes=verify_hashes)
        
        # length of each trajectory in the dataset
        # excluding the input_length_sequence
        # may (and likely is) variable between data
        self._dimension = self._data[0][0].shape[-1]
        self._input_length_sequence = input_length_sequence
        self._material_property_as_feature = True if len(self._data[0]) >= 3 else False
        if self._material_property_as_feature:  # if raw data includes material_property
            self._data_lengths = [x.shape[0] - self._input_length_sequence for x, _, _ in self._data]
        else:
            self._data_lengths = [x.shape[0] - self._input_length_sequence for x, _, in self._data]
        self._length = sum(self._data_lengths)

        # pre-compute cumulative lengths
        # to allow fast indexing in __getitem__
        self._precompute_cumlengths = [sum(self._data_lengths[:x]) for x in range(1, len(self._data_lengths) + 1)]
        self._precompute_cumlengths = np.array(self._precompute_cumlengths, dtype=int)

    def __len__(self):
        """Return length of dataset.
        
        Returns:
            int: Length of dataset.
        """
        return self._length

    def __getitem__(self, idx):
        """Returns a training example from the dataset.
        
        Args:
            idx (int): Index of training example.

        Returns:
            tuple: Tuple of the form ((positions, particle_type, n_particles_per_example), label).
        """
        # Select the trajectory immediately before
        # the one that exceeds the idx
        # (i.e., the one in which idx resides).
        trajectory_idx = np.searchsorted(self._precompute_cumlengths - 1, idx, side="left")

        # Compute index of pick along time-dimension of trajectory.
        start_of_selected_trajectory = self._precompute_cumlengths[trajectory_idx - 1] if trajectory_idx != 0 else 0
        time_idx = self._input_length_sequence + (idx - start_of_selected_trajectory)

        # Prepare training data.
        positions = self._data[trajectory_idx][0][time_idx - self._input_length_sequence:time_idx]
        positions = np.transpose(positions, (1, 0, 2))  # nparticles, input_sequence_length, dimension
        particle_type = np.full(positions.shape[0], self._data[trajectory_idx][1], dtype=int)
        n_particles_per_example = positions.shape[0]
        label = self._data[trajectory_idx][0][time_idx]

        if self._material_property_as_feature:  # if raw data includes material_property
            material_property = np.full(positions.shape[0], self._data[trajectory_idx][2], dtype=float)
            training_example = ((positions, particle_type, material_property, n_particles_per_example), label)
        else:
            training_example = ((positions, particle_type, n_particles_per_example), label)

        return training_example


def collate_fn(data):
    """Collate function for SamplesDataset.

    Args:
        data (list): List of tuples of the form ((positions, particle_type, n_particles_per_example), label).

    Returns:
        tuple: Tuple of the form ((positions, particle_type, n_particles_per_example), label).    
    """
    material_property_as_feature = True if len(data[0][0]) >= 4 else False
    position_list = []
    particle_type_list = []
    if material_property_as_feature:
        material_property_list = []
    n_particles_per_example_list = []
    label_list = []

    if material_property_as_feature:
        for ((positions, particle_type, material_property, n_particles_per_example), label) in data:
            position_list.append(positions)
            particle_type_list.append(particle_type)
            material_property_list.append(material_property)
            n_particles_per_example_list.append(n_particles_per_example)
            label_list.append(label)
    else:
        for ((positions, particle_type, n_particles_per_example), label) in data:
            position_list.append(positions)
            particle_type_list.append(particle_type)
            n_particles_per_example_list.append(n_particles_per_example)
            label_list.append(label)

    if material_property_as_feature:
        collated_data = (
            (
                torch.tensor(np.vstack(position_list)).to(torch.float32).contiguous(),
                torch.tensor(np.concatenate(particle_type_list)).contiguous(),
                torch.tensor(np.concatenate(material_property_list)).to(torch.float32).contiguous(),
                torch.tensor(n_particles_per_example_list).contiguous(),
            ),
            torch.tensor(np.vstack(label_list)).to(torch.float32).contiguous()
        )
    else:
        collated_data = (
            (
                torch.tensor(np.vstack(position_list)).to(torch.float32).contiguous(),
                torch.tensor(np.concatenate(particle_type_list)).contiguous(),
                torch.tensor(n_particles_per_example_list).contiguous(),
            ),
            torch.tensor(np.vstack(label_list)).to(torch.float32).contiguous()
        )

    return collated_data


class TrajectoriesDataset(torch.utils.data.Dataset):
    """Dataset of trajectories.

    Each trajectory is a tuple of the form (positions, particle_type).
    positions is a numpy array of shape (sequence_length, n_particles, dimension).
    """

    def __init__(self, path, verify_hashes=False):
        super().__init__()
        # load dataset stored in npz format
        # data is loaded as dict of tuples
        # of the form (positions, particle_type)
        # convert to list of tuples
        # TODO (jpv): allow_pickle=True is potential security risk. See docs.
        self._data = load_npz_data(path, verify_hashes=verify_hashes)
        self._dimension = self._data[0][0].shape[-1]
        self._length = len(self._data)
        self._material_property_as_feature = True if len(self._data[0]) >= 3 else False

    def __len__(self):
        """Return length of dataset.

        Returns:
            int: Length of dataset.
        """
        return self._length

    def __getitem__(self, idx):
        """Returns a training example from the dataset.

        Args:
            idx (int): Index of training example.

        Returns:
            tuple: Tuple named,
              trajectory = (positions, particle_type, material_property (optional), n_particles_per_example).
        """
        if self._material_property_as_feature:
            positions, _particle_type, _material_property = self._data[idx]
            positions = np.transpose(positions, (1, 0, 2))
            particle_type = np.full(positions.shape[0], _particle_type, dtype=int)
            material_property = np.full(positions.shape[0], _material_property, dtype=float)
            n_particles_per_example = positions.shape[0]

            trajectory = (
                torch.tensor(positions).to(torch.float32).contiguous(),
                torch.tensor(particle_type).contiguous(),
                torch.tensor(material_property).to(torch.float32).contiguous(),
                n_particles_per_example
            )
        else:
            positions, _particle_type = self._data[idx]
            positions = np.transpose(positions, (1, 0, 2))
            particle_type = np.full(positions.shape[0], _particle_type, dtype=int)
            n_particles_per_example = positions.shape[0]

            trajectory = (
                torch.tensor(positions).to(torch.float32).contiguous(),
                torch.tensor(particle_type).contiguous(),
                n_particles_per_example
            )

        return trajectory


def get_data_loader_by_samples(path, input_length_sequence, batch_size, shuffle=True,
                               verify_hashes=False):
    """Returns a data loader for the dataset.

    Args:
        path (str): Path to dataset.
        input_length_sequence (int): Length of input sequence.
        batch_size (int): Batch size.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.

    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = SamplesDataset(path, input_length_sequence, verify_hashes=verify_hashes)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       pin_memory=True, collate_fn=collate_fn)


def get_data_loader_by_trajectories(path, verify_hashes=False):
    """Returns a data loader for the dataset.

    Args:
        path (str): Path to dataset.

    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset.
    """
    dataset = TrajectoriesDataset(path, verify_hashes=verify_hashes)
    return torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False,
                                       pin_memory=True)
