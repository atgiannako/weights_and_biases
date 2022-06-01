# Weights and Biases Tutorial
Weights and biases `wandb` is a library for experiment tracking, metric logging and hyperoptimization tuning.
    
    pip install wandb

## Local installation
    # create a virtual environment
    source venv/bin/activate
    pip install -r requirements.txt

## Get the data

    wget www.di.ens.fr/~lelarge/MNIST.tar.gz
    tar -zxvf MNIST.tar.gz

## Run wand locally
    docker pull wandb/local
    docker stop wandb-local
    docker run --rm -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local

## License and API key
In order to run `wandb` locally with docker, you need to create a license and get an API key. Read more from the official [documentation](https://docs.wandb.ai/guides/self-hosted/local#login).

### Source Code
Example taken from [here](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb).
