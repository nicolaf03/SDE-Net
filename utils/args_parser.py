import argparse
import json

# Initialize the parser
parser = argparse.ArgumentParser()

# Add arguments to the parser
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--patience', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('--t-size', type=int, default=7)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--viz', action='store_true', help="Show plots while training")
parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
    "If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")
parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
    "Used for periodic function demo.")
parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")
parser.add_argument('-l', '--latents', type=int, default=10, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")
parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")
parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")
parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")
parser.add_argument('--linear-classif', action='store_true', help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

# Parse the arguments
args = parser.parse_args()

# Create a dictionary from the parsed arguments
args_dict = vars(args)

# Print the dictionary
print(args_dict)

# Specify the file path where you want to save the dictionary
file_path = "parameters/params_latentODE.json"

# Open the file in binary write mode
with open(file_path, "w") as file:
    # Serialize and write the dictionary to the file
    json.dump(args_dict, file)
