import yaml


my_variable = "hello world"

my_other_variable = "bandits are great"


def read_yaml(f):
    with open(f, "r") as con:
        x = yaml.safe_load(con)
    return x
