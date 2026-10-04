import collections
import json
import os
import pickle
import glob
import re
import sys
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from absl import flags
from absl import app

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gns import learned_simulator
from gns import noise_utils
from gns import reading_utils
from gns import data_loader
from gns import distribute
from gns.losses import acceleration_loss
from gns.model_io import build_simulator, load_for_evaluation

flags.DEFINE_enum(
    'mode', 'train', ['train', 'valid', 'rollout'],
    help='Train model, validation or rollout evaluation.')
flags.DEFINE_integer('batch_size', 2, help='The batch size.')
flags.DEFINE_float('noise_std', 6.7e-4, help='The std deviation of the noise.')
flags.DEFINE_enum('loss', 'nll', ['nll', 'legacy_nll', 'mse', 'faithful'],
                  help='Correct NLL, released legacy objective, MSE, or faithful heteroscedastic regression.')
flags.DEFINE_float('variance_floor', 1e-6, help='Floor in normalized acceleration variance units.')
flags.DEFINE_integer('seed', 0, help='Training initialization and sampling seed.')
flags.DEFINE_enum('radius_backend', 'pyg', ['pyg', 'scipy'],
                  help='PyG production backend or explicit SciPy CPU reference backend.')
flags.DEFINE_float('connectivity_radius', None, help='Override dataset connectivity radius.')
flags.DEFINE_integer('nmessage_passing_steps', 10, help='Processor depth; zero is a particle MLP.')
flags.DEFINE_string('data_path', None, help='The dataset directory.')
flags.DEFINE_string('model_path', 'models/', help=('The path for saving checkpoints of the model.'))
flags.DEFINE_string('output_path', 'rollouts/', help='The path for saving outputs (e.g. rollouts).')
flags.DEFINE_string('output_filename', 'rollout', help='Base name for saving the rollout')
flags.DEFINE_string('model_file', None, help=('Model filename (.pt) to resume from. Can also use "latest" to default to newest file.'))
flags.DEFINE_string('train_state_file', 'train_state.pt', help=('Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.'))

flags.DEFINE_integer('ntraining_steps', int(2E7), help='Number of training steps.')
flags.DEFINE_integer('validation_interval', None, help='Validation interval. Set `None` if validation loss is not needed')
flags.DEFINE_integer('nsave_steps', int(5000), help='Number of steps at which to save the model.')

# Learning rate parameters
flags.DEFINE_float('lr_init', 1e-4, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(5e6), help='Learning rate decay steps.')

flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")
flags.DEFINE_integer("n_gpus", 1, help="The number of GPUs to utilize for training.")

# Graph augmentation parameters (for closing the train/inference covariate shift
# introduced by the adaptive rollout). Defaults of 0.0 keep behavior unchanged.
flags.DEFINE_float('augment_radius_prob', 0.0,
                   help='Per-particle probability of receiving expanded connectivity '
                        'during training. 0.0 disables augmentation (default).')
flags.DEFINE_float('augment_radius_factor', 1.267,
                   help='Multiplier on connectivity_radius for augmented particles.')

FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

def rollout(
        simulator: learned_simulator.LearnedSimulator,
        position: torch.tensor,
        particle_types: torch.tensor,
        material_property: torch.tensor,
        n_particles_per_example: torch.tensor,
        nsteps: int,
        device: torch.device):
  """
  Rolls out a trajectory by applying the model in sequence.

  Args:
    simulator: Learned simulator.
    position: Positions of particles (timesteps, nparticles, ndims)
    particle_types: Particles types with shape (nparticles)
    material_property: Friction angle normalized by tan() with shape (nparticles)
    n_particles_per_example
    nsteps: Number of steps.
    device: torch device.
  """

  initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]
  ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]

  current_positions = initial_positions
  predictions = []

  for step in tqdm(range(nsteps), total=nsteps):
    # Get next position with shape (nnodes, dim)
    next_position = simulator.predict_positions(
        current_positions,
        nparticles_per_example=[n_particles_per_example],
        particle_types=particle_types,
        material_property=material_property
    )

    # Update kinematic particles from prescribed trajectory.
    kinematic_mask = (particle_types == KINEMATIC_PARTICLE_ID).clone().detach().to(device)
    next_position_ground_truth = ground_truth_positions[:, step]
    kinematic_mask = kinematic_mask.bool()[:, None].expand(-1, current_positions.shape[-1])
    next_position = torch.where(
        kinematic_mask, next_position_ground_truth, next_position)
    predictions.append(next_position)

    # Shift `current_positions`, removing the oldest position in the sequence
    # and appending the next position at the end.
    current_positions = torch.cat(
        [current_positions[:, 1:], next_position[:, None, :]], dim=1)

  # Predictions with shape (time, nnodes, dim)
  predictions = torch.stack(predictions)
  ground_truth_positions = ground_truth_positions.permute(1, 0, 2)

  loss = (predictions - ground_truth_positions) ** 2

  output_dict = {
      'initial_positions': initial_positions.permute(1, 0, 2).cpu().numpy(),
      'predicted_rollout': predictions.cpu().numpy(),
      'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
      'particle_types': particle_types.cpu().numpy(),
      'material_property': material_property.cpu().numpy() if material_property is not None else None
  }

  return output_dict, loss


def predict(device: str):
  """Predict rollouts.

  Args:
    simulator: Trained simulator if not will undergo training.

  """
  # Read metadata
  metadata = reading_utils.read_metadata(FLAGS.data_path, "rollout")
  checkpoint = os.path.join(FLAGS.model_path, FLAGS.model_file)
  simulator, provenance = load_for_evaluation(
      checkpoint, metadata, device,
      normalization_noise_std=(FLAGS.noise_std if FLAGS['noise_std'].present else None),
      connectivity_radius=FLAGS.connectivity_radius,
      nmessage_passing_steps=(FLAGS.nmessage_passing_steps
                             if FLAGS['nmessage_passing_steps'].present else None),
      allow_missing_variance_head=True,
      radius_backend=(FLAGS.radius_backend if FLAGS["radius_backend"].present else None))

  # Output path
  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)

  # Missing validation data must never silently redirect model selection to test.
  split = "test" if FLAGS.mode == "rollout" else "valid"

  # Get dataset
  ds = data_loader.get_data_loader_by_trajectories(path=f"{FLAGS.data_path}{split}.npz")
  # See if our dataset has material property as feature
  if len(ds.dataset._data[0]) == 3:  # `ds` has (positions, particle_type, material_property)
    material_property_as_feature = True
  elif len(ds.dataset._data[0]) == 2:  # `ds` only has (positions, particle_type)
    material_property_as_feature = False
  else:
    raise NotImplementedError

  eval_loss = []
  with torch.no_grad():
    for example_i, features in enumerate(ds):
      print(f"processing example number {example_i}")
      positions = features[0].to(device)
      nsteps = positions.shape[1] - INPUT_SEQUENCE_LENGTH
      particle_type = features[1].to(device)
      if material_property_as_feature:
        material_property = features[2].to(device)
        n_particles_per_example = torch.tensor([int(features[3])], dtype=torch.int32).to(device)
      else:
        material_property = None
        n_particles_per_example = torch.tensor([int(features[2])], dtype=torch.int32).to(device)

      # Predict example rollout
      example_rollout, loss = rollout(simulator,
                                      positions,
                                      particle_type,
                                      material_property,
                                      n_particles_per_example,
                                      nsteps,
                                      device)

      example_rollout['metadata'] = metadata
      example_rollout['evaluation_provenance'] = provenance
      example_rollout['split'] = split
      dynamic_loss = loss[:, particle_type != KINEMATIC_PARTICLE_ID]
      if not dynamic_loss.numel():
        raise ValueError("Rollout MSE requires a dynamic particle")
      print("Predicting example {} loss: {}".format(example_i, dynamic_loss.mean()))
      eval_loss.append(torch.flatten(dynamic_loss))

      # Save rollout in testing
      if FLAGS.mode == 'rollout':
        example_rollout['metadata'] = metadata
        example_rollout['loss'] = dynamic_loss.mean()
        filename = f'{FLAGS.output_filename}_ex{example_i}.pkl'
        filename = os.path.join(FLAGS.output_path, filename)
        with open(filename, 'wb') as f:
          pickle.dump(example_rollout, f)

  print("Mean loss on rollout prediction: {}".format(
      torch.mean(torch.cat(eval_loss))))


def optimizer_to(optim, device):
  for param in optim.state.values():
    # Not sure there are any global tensors in the state dict
    if isinstance(param, torch.Tensor):
      param.data = param.data.to(device)
      if param._grad is not None:
        param._grad.data = param._grad.data.to(device)
    elif isinstance(param, dict):
      for subparam in param.values():
        if isinstance(subparam, torch.Tensor):
          subparam.data = subparam.data.to(device)
          if subparam._grad is not None:
            subparam._grad.data = subparam._grad.data.to(device)

def save_model_and_train_state(rank, device, simulator, flags, step, epoch, optimizer,
                                train_loss, valid_loss, train_loss_hist, valid_loss_hist):
  """Save model state
  
  Args:
    rank: local rank
    device: torch device type
    simulator: Trained simulator if not will undergo training.
    flags: flags
    step: step
    epoch: epoch
    optimizer: optimizer
    train_loss: training loss at current step
    valid_loss: validation loss at current step
    train_loss_hist: training loss history at each epoch
    valid_loss_hist: validation loss history at each epoch
  """
  if rank == 0 or device == torch.device("cpu"):
      if device == torch.device("cpu"):
          simulator.save(flags["model_path"] + 'model-' + str(step) + '.pt')
      else:
          simulator.module.save(flags["model_path"] + 'model-' + str(step) + '.pt')

      train_state = dict(optimizer_state=optimizer.state_dict(),
                          global_train_state={
                            "step": step, 
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "valid_loss": valid_loss
                            },
                          loss_history={"train": train_loss_hist, "valid": valid_loss_hist}
                          )
      torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')
      
def train(rank, flags, world_size, device):
  """Train the model.

  Args:
    rank: local rank
    world_size: total number of ranks
    device: torch device type
  """
  if device == torch.device("cuda"):
    distribute.setup(rank, world_size, device)
    device_id = rank
  else:
    device_id = device

  seed_rank = 0 if rank is None else rank
  random.seed(flags["seed"] + seed_rank)
  np.random.seed(flags["seed"] + seed_rank)
  torch.manual_seed(flags["seed"] + seed_rank)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(flags["seed"] + seed_rank)

  # Read metadata
  metadata = reading_utils.read_metadata(flags["data_path"], "train")
  def make_simulator(target_device):
    result = build_simulator(
        metadata, flags["noise_std"], flags["noise_std"], target_device,
        connectivity_radius=flags["connectivity_radius"],
        nmessage_passing_steps=flags["nmessage_passing_steps"],
        uncertainty_parameterization=("legacy_std" if flags["loss"] == "legacy_nll" else "variance"),
        variance_floor=flags["variance_floor"],
        detach_variance_features=(flags["loss"] == "faithful"),
        radius_backend=flags["radius_backend"])
    result._training_config = dict(flags)
    return result

  # Get simulator and optimizer
  if device == torch.device("cuda"):
    serial_simulator = make_simulator(rank)
    simulator = DDP(serial_simulator.to(rank), device_ids=[rank], output_device=rank,
                    find_unused_parameters=(flags["loss"] == "mse"))
    optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"]*world_size)
  else:
    simulator = make_simulator(device)
    optimizer = torch.optim.Adam(simulator.parameters(), lr=flags["lr_init"] * world_size)

  # Initialize training state
  step = 0
  epoch = 0
  steps_per_epoch = 0

  valid_loss = None
  epoch_train_loss = 0
  epoch_valid_loss = None

  train_loss_hist = []
  valid_loss_hist = []

  # If model_path does exist and model_file and train_state_file exist continue training.
  if flags["model_file"] is not None:

    if flags["model_file"] == "latest" and flags["train_state_file"] == "latest":
      # find the latest model, assumes model and train_state files are in step.
      fnames = glob.glob(f'{flags["model_path"]}*model*pt')
      max_model_number = 0
      expr = re.compile(".*model-(\d+).pt")
      for fname in fnames:
        model_num = int(expr.search(fname).groups()[0])
        if model_num > max_model_number:
          max_model_number = model_num
      # reset names to point to the latest.
      flags["model_file"] = f"model-{max_model_number}.pt"
      flags["train_state_file"] = f"train_state-{max_model_number}.pt"

    if os.path.exists(flags["model_path"] + flags["model_file"]) and os.path.exists(flags["model_path"] + flags["train_state_file"]):
      # load model
      if device == torch.device("cuda"):
        simulator.module.load(flags["model_path"] + flags["model_file"])
      else:
        simulator.load(flags["model_path"] + flags["model_file"])

      loaded_simulator = simulator.module if hasattr(simulator, "module") else simulator
      saved_loss = loaded_simulator._training_config.get("loss")
      if saved_loss is not None and saved_loss != flags["loss"]:
        raise ValueError("Resume loss differs from checkpoint; start a new experiment")
      loaded_simulator._training_config = dict(flags)
      # load train state
      train_state = torch.load(flags["model_path"] + flags["train_state_file"])
      
      # set optimizer state
      optimizer = torch.optim.Adam(
        simulator.module.parameters() if device == torch.device("cuda") else simulator.parameters())
      optimizer.load_state_dict(train_state["optimizer_state"])
      optimizer_to(optimizer, device_id)
      
      # set global train state
      step = train_state["global_train_state"]["step"]
      epoch = train_state["global_train_state"]["epoch"]
      train_loss_hist = train_state["loss_history"]["train"]
      valid_loss_hist = train_state["loss_history"]["valid"]

    else:
      msg = f'Specified model_file {flags["model_path"] + flags["model_file"]} and train_state_file {flags["model_path"] + flags["train_state_file"]} not found.'
      raise FileNotFoundError(msg)

  simulator.train()
  simulator.to(device_id)

  # Get data loader
  get_data_loader = (
    distribute.get_data_distributed_dataloader_by_samples
    if device == torch.device("cuda")
    else data_loader.get_data_loader_by_samples
  )

  # Load training data
  dl = get_data_loader(
      path=f'{flags["data_path"]}train.npz',
      input_length_sequence=INPUT_SEQUENCE_LENGTH,
      batch_size=flags["batch_size"],
  )
  n_features = len(dl.dataset._data[0])

  # Load validation data
  if flags["validation_interval"] is not None:
      dl_valid = get_data_loader(
          path=f'{flags["data_path"]}valid.npz',
          input_length_sequence=INPUT_SEQUENCE_LENGTH,
          batch_size=flags["batch_size"],
      )
      if len(dl_valid.dataset._data[0]) != n_features:
          raise ValueError(
              f"`n_features` of `valid.npz` and `train.npz` should be the same"
          )

  print(f"rank = {rank}, cuda = {torch.cuda.is_available()}")

  try:
    while step < flags["ntraining_steps"]:
      if device == torch.device("cuda"):
        torch.distributed.barrier()

      if hasattr(dl.sampler, "set_epoch"):
        dl.sampler.set_epoch(epoch)
      for example in dl:  
        steps_per_epoch += 1
        # ((position, particle_type, material_property, n_particles_per_example), labels) are in dl
        position = example[0][0].to(device_id)
        particle_type = example[0][1].to(device_id)
        if n_features == 3:  # if dl includes material_property
          material_property = example[0][2].to(device_id)
          n_particles_per_example = example[0][3].to(device_id)
        elif n_features == 2:
          n_particles_per_example = example[0][2].to(device_id)
        else:
          raise NotImplementedError
        labels = example[1].to(device_id)

        n_particles_per_example.to(device_id)
        labels.to(device_id)

        # TODO (jpv): Move noise addition to data_loader
        # Sample the noise to add to the inputs to the model during training.
        sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(position, noise_std_last_step=flags["noise_std"]).to(device_id)
        non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(device_id)
        sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

        # Get the predictions and target accelerations
        device_or_rank = rank if device == torch.device("cuda") else device
        predict_fn = simulator
        pred_acc, pred_variance, target_acc = predict_fn(
            next_positions=labels.to(device_or_rank),
            position_sequence_noise=sampled_noise.to(device_or_rank),
            position_sequence=position.to(device_or_rank),
            nparticles_per_example=n_particles_per_example.to(device_or_rank),
            particle_types=particle_type.to(device_or_rank),
            material_property=material_property.to(device_or_rank) if n_features == 3 else None,
            augment_radius_prob=flags["augment_radius_prob"],
            augment_radius_factor=flags["augment_radius_factor"],
        )

        # Validation
        if flags["validation_interval"] is not None and step > 0 and step % flags["validation_interval"] == 0:
          sampled_valid_example = next(iter(dl_valid))
          valid_loss = validation(
              simulator, sampled_valid_example, n_features, flags, rank, device_id)
          print(f"Validation loss at {step}: {valid_loss.item()}")

        # Calculate the loss (heteroscedastic NLL) and mask out kinematic particles
        loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask,
                                 pred_variance=pred_variance, loss_type=flags["loss"],
                                 variance_floor=flags["variance_floor"])

        train_loss = loss.item()
        epoch_train_loss += train_loss

        # Computes the gradient of loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate
        lr_new = flags["lr_init"] * (flags["lr_decay"] ** (step/flags["lr_decay_steps"])) * world_size
        for param in optimizer.param_groups:
          param['lr'] = lr_new
     
        print(f'rank = {rank}, epoch = {epoch}, step = {step}/{flags["ntraining_steps"]}, loss = {train_loss}', flush=True)

        step += 1
        # Checkpoint step is the number of completed optimizer updates.
        # Save model state
        if rank == 0 or device == torch.device("cpu"):
          if step % flags["nsave_steps"] == 0:
            save_model_and_train_state(rank, device, simulator, flags, step, epoch, 
                                       optimizer, train_loss, valid_loss, train_loss_hist, valid_loss_hist)

        if step >= flags["ntraining_steps"]:
            break

      # Epoch level statistics
      # Training loss at epoch
      epoch_train_loss /= steps_per_epoch
      epoch_train_loss = torch.tensor([epoch_train_loss]).to(device_id)
      if device == torch.device("cuda"):
        torch.distributed.reduce(epoch_train_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
        epoch_train_loss /= world_size

      train_loss_hist.append((epoch, epoch_train_loss.item()))

      # Validation loss at epoch
      if flags["validation_interval"] is not None:
        sampled_valid_example = next(iter(dl_valid))
        epoch_valid_loss = validation(
                simulator, sampled_valid_example, n_features, flags, rank, device_id)
        if device == torch.device("cuda"):
          torch.distributed.reduce(epoch_valid_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
          epoch_valid_loss /= world_size

        valid_loss_hist.append((epoch, epoch_valid_loss.item()))

      # Print epoch statistics
      if rank == 0 or device == torch.device("cpu"):
        print(f'Epoch {epoch}, training loss: {epoch_train_loss.item()}')
        if flags["validation_interval"] is not None:
          print(f'Epoch {epoch}, validation loss: {epoch_valid_loss.item()}')
      
      # Reset epoch training loss
      epoch_train_loss = 0
      if steps_per_epoch >= len(dl):
        epoch += 1
      steps_per_epoch = 0
      
      if step >= flags["ntraining_steps"]:
        break 
      
  except KeyboardInterrupt:
    pass

  # Save model state on keyboard interrupt
  save_model_and_train_state(rank, device, simulator, flags, step, epoch, optimizer, train_loss, valid_loss, train_loss_hist, valid_loss_hist)

  if torch.cuda.is_available():
    distribute.cleanup()


def _get_simulator(
        metadata: json,
        acc_noise_std: float,
        vel_noise_std: float,
        device: torch.device) -> learned_simulator.LearnedSimulator:
  """Instantiates the simulator.

  Args:
    metadata: JSON object with metadata.
    acc_noise_std: Acceleration noise std deviation.
    vel_noise_std: Velocity noise std deviation.
    device: PyTorch device 'cpu' or 'cuda'.
  """

  return build_simulator(metadata, acc_noise_std, vel_noise_std, device)


def validation(
        simulator,
        example,
        n_features,
        flags,
        rank,
        device_id):

  position = example[0][0].to(device_id)
  particle_type = example[0][1].to(device_id)
  if n_features == 3:  # if dl includes material_property
    material_property = example[0][2].to(device_id)
    n_particles_per_example = example[0][3].to(device_id)
  elif n_features == 2:
    n_particles_per_example = example[0][2].to(device_id)
  else:
    raise NotImplementedError
  labels = example[1].to(device_id)

  # Sample the noise to add to the inputs.
  sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
    position, noise_std_last_step=flags["noise_std"]).to(device_id)
  non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID).clone().detach().to(device_id)
  sampled_noise *= non_kinematic_mask.view(-1, 1, 1)

  # Do evaluation for the validation data
  device_or_rank = rank if isinstance(device_id, int) else device_id
  # Select the appropriate prediction function
  predict_accelerations = simulator.module.predict_accelerations if isinstance(device_id, int) else simulator.predict_accelerations
  # Get the predictions and target accelerations
  with torch.no_grad():
      pred_acc, pred_variance, target_acc = predict_accelerations(
          next_positions=labels.to(device_or_rank),
          position_sequence_noise=sampled_noise.to(device_or_rank),
          position_sequence=position.to(device_or_rank),
          nparticles_per_example=n_particles_per_example.to(device_or_rank),
          particle_types=particle_type.to(device_or_rank),
          material_property=material_property.to(device_or_rank) if n_features == 3 else None
      )

  # Compute loss (heteroscedastic NLL)
  loss = acceleration_loss(pred_acc, target_acc, non_kinematic_mask,
                                 pred_variance=pred_variance, loss_type=flags["loss"],
                                 variance_floor=flags["variance_floor"])

  return loss


def main(_):
  """Train or evaluates the model.

  """
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device == torch.device('cuda'):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

  myflags = reading_utils.flags_to_dict(FLAGS)

  if FLAGS.mode == 'train':
    # If model_path does not exist create new directory.
    if not os.path.exists(FLAGS.model_path):
      os.makedirs(FLAGS.model_path)

    # Train on gpu 
    if device == torch.device('cuda'):
      available_gpus = torch.cuda.device_count()
      print(f"Available GPUs = {available_gpus}")

      # Set the number of GPUs based on availability and the specified number
      if FLAGS.n_gpus is None or FLAGS.n_gpus > available_gpus:
        world_size = available_gpus
        if FLAGS.n_gpus is not None:
          print(f"Warning: The number of GPUs specified ({FLAGS.n_gpus}) exceeds the available GPUs ({available_gpus})")
      else:
        world_size = FLAGS.n_gpus

      # Print the status of GPU usage
      print(f"Using {world_size}/{available_gpus} GPUs")

      # Spawn training to GPUs
      distribute.spawn_train(train, myflags, world_size, device)

    # Train on cpu  
    else:
      rank = None
      world_size = 1
      train(rank, myflags, world_size, device)

  elif FLAGS.mode in ['valid', 'rollout']:
    # Set device
    world_size = torch.cuda.device_count()
    if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
      device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')
    #test code
    print(f"device is {device} world size is {world_size}")
    predict(device)


if __name__ == '__main__':
  app.run(main)
