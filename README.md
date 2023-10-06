# DeepClean, take 3
Here's a restructure of DeepClean intended for at least a few purposes:
- Taking advantage of [lightning](https://lightning.ai/) during training for faster, more modular, and simpler training code
- Containerizing DeepClean projects for easier uptake by new users
- Serving as a proof of concept and sandbox for a `law`-based workflow with custom configs
- Serving as a proof of concept for use of a centralized [`toolbox`](./toolbox/) containing standard ML4GW libraries and code-styling configs
- Generalizing many of the components of DeepClean for easier experimentation with new architectures and for target new frequency bands with the [`couplings`](./deepclean/couplings) submodule of the pipeline library

I'll expand more on these when I have time, but for now I'll add a couple instructions to get started with training.

## Run instructions
### Environment setup
| **TODO: data authentication instructions**

#### Install
##### `pip` instructions: not recommended
You can install the local library via `pip`:

```bash
python -m pip install -e .
```

##### `poetry` instructions: recommended
However, I'd recommend using some sort of virtualization software. This repo is automatically compatible with [`poetry`](https://python-poetry.org/), my personal favorite. In that case, you would just need to do

```bash
poetry install
```

#### Directory setup
Set up a local directory to save container images that we'll export as `DEEPCLEAN_CONTAINER_ROOT`.
**IMPORTANT: This directory should _not_ live locally with the rest of this code. Apptainer has no syntax for excluding files from being added into containers at build time, so if you save your (often GB-sized) containers here, you'll make building the next container enormously more painful. This means that you'll need to build your images on every _filesystem_ (not node) that you intend to run on. In the future, we'll make these containers available on `/cvmfs` via the Open Science Grid, but for now you'll need to build them**.

```bash
# or wherever you want to save this
export DEEPCLEAN_CONTAINER_ROOT=~/images/deepclean
mkdir -p $DEEPCLEAN_CONTAINER_ROOT
```

Finally make a directory to save our data and run outputs

```bash
export DATA_DIR=~/deepclean/data
mkdir -p $DATA_DIR

export RESULTS_DIR=~/deepclean/results
mkdir -p $RESULTS_DIR
```

### Dataset generation
I don't have this built into a `law.Task` yet, so you'll have to run this one manually. Start by building the container

```bash
apptainer build $DEEPCLEAN_CONTAINER_ROOT/data.sif projects/data/apptainer.def
```

Then you can query segments containing usable data via

```bash
apptainer run $DEEPCLEAN_CONTAINER_ROOT/data.sif \
    python /opt/deepclean/projects/data/data --config /opt/deepclean/projects/data/config.yaml \
        query --output-file $DATA_DIR/segments.txt
```

Then select some segment times from the output text file and run (for example)

```bash
apptainer run $DEEPCLEAN_CONTAINER_ROOT/data.sif \
    python /opt/deepclean/projects/data/data --config /opt/deepclean/projects/data/config.yaml \
        fetch --output-directory $DATA_DIR --start 1250916945 --end 1250951947
```

#### Making changes to the code
Once these tools are moved into the `deepclean` pipeline this will be done automatically, but for now if you make any local changes to the code, be sure to add the `--bind .:/opt/deepclean` flag after `apptainer run` in the commands above so that you changes are reflected in the container.

### Training
Once you've generated your training data, you're ready to train! Start by building your training container image

```bash
apptainer build $DEEPCLEAN_CONTAINER_ROOT/train.sif projects/train/apptainer.def
```

Find a node with some decently-sized GPUs, ensure that the one you want isn't being used, and then run (assuming you built this library with `poetry`):

```bash
GPU_INDEX=0  # or whichever you want
poetry run law run deepclean.tasks.Train  \
    --image train.sif \
    --gpus $GPU_INDEX \
    --data-fname $DATA_DIR/deepclean-1250916945-35002.h5 \
    --output-dir $RESULTS_DIR/my-first-run
```
