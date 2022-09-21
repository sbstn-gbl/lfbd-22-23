import matplotlib.pyplot as plt
import yaml


plt.style.use("seaborn-white")

plt.rcParams.update(
    {
        "figure.figsize": (9, 9),
        "axes.titlesize": 20,
        "axes.labelsize": 15,
        "legend.fontsize": 15,
        "axes.grid": True,
        "axes.axisbelow": True,
        "pcolor.shading": "auto",
    }
)


def read_yaml(f):
    with open(f, "r") as con:
        x = yaml.safe_load(con)
    return x