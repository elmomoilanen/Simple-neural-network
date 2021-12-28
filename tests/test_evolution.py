import numpy as np
from numpy.lib.arraysetops import isin
import pytest

from simple_neural_network.evolution import Evolution

# Evolution class uses these
params = ("hidden_nodes", "optimizer", "learning_rate", "lambda_", "activation1", "activation2")


def test_init_obj():
    generations, pop_size = 5, 10
    evo = Evolution(generations=generations, population_size=pop_size)

    assert evo.generations == generations
    assert evo.pop_size == pop_size

    top_popu = int(evo.top_percentage * pop_size)
    assert top_popu == evo._top_popu

    poor_popu = int(evo.poor_percentage * pop_size)
    assert poor_popu == evo._poor_popu

    other_popu = pop_size - top_popu - poor_popu
    assert other_popu == evo._other_popu


def test_init_obj_invalid_popu_partition():
    evo = Evolution(generations=2, population_size=3)

    old_top_per, old_poor_per = Evolution.top_percentage, Evolution.poor_percentage
    # modify percentages to be invalid
    Evolution.top_percentage = 0.9
    Evolution.poor_percentage = 0.9

    new_pop_size = 10
    top_popu = int(new_pop_size * Evolution.top_percentage)
    poor_popu = int(new_pop_size * Evolution.poor_percentage)
    top_and_poor = top_popu + poor_popu

    # reset population size which should raise error due to percentage changes

    with pytest.raises(
        ValueError,
        match=rf"Size {top_and_poor} of top and poor performers cannot be larger than total population size {new_pop_size}",
    ):
        evo.pop_size = new_pop_size

    # set percentages back
    Evolution.top_percentage = old_top_per
    Evolution.poor_percentage = old_poor_per


def test_init_population():
    evo = Evolution(generations=1, population_size=2)
    population = evo._init_population()

    for member in population:
        shared_keys = set(member.keys()).intersection(params)
        assert len(shared_keys) == len(params)


def test_select_by_fitness():
    pop_size = 10
    evo = Evolution(generations=2, population_size=pop_size)

    assert evo.pop_size == pop_size

    new_pop_size = evo._top_popu + evo._poor_popu
    assert new_pop_size < evo.pop_size

    population = [{"fitness": j} for j in range(pop_size)]
    result = evo._select_by_fitness(population)

    assert len(result) == new_pop_size


def test_mutation():
    evo = Evolution(generations=2, population_size=3)

    population = [{key: 1 for key in params} for j in range(evo.pop_size)]

    evo._mutate(population)

    # test that mutation didn't change key names in population

    for member in population:
        shared_keys = set(member.keys()).intersection(params)
        assert len(shared_keys) == len(params)


def test_reproducing():
    init_pop_size = 15
    evo = Evolution(generations=5, population_size=init_pop_size)

    # test that new members (offsprings) have all key names and the population size is correct
    # correct size here: evo.pop_size + offspring count
    top_size = int(Evolution.top_percentage * init_pop_size)
    poor_size = int(Evolution.poor_percentage * init_pop_size)
    offspring_size = init_pop_size - top_size - poor_size

    correct_new_size = init_pop_size + offspring_size

    population = [{key: 1 for key in params} for j in range(evo.pop_size)]
    assert len(population) == init_pop_size

    evo._reproduce(population)

    assert len(population) == correct_new_size

    for member in population:
        shared_keys = set(member.keys()).intersection(params)
        assert len(shared_keys) == len(params)


def test_splitting_data():
    evo = Evolution(generations=2, population_size=2)

    x = np.zeros((50, 3))
    y = np.zeros(50)

    train_tuple, test_tuple = evo.split_data_to_train_and_test(x, y)

    assert isinstance(train_tuple, tuple)
    assert isinstance(test_tuple, tuple)

    assert len(train_tuple) == 2
    assert len(test_tuple) == 2

    x_train, y_train = train_tuple
    x_test, y_test = test_tuple

    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]

    assert (x_train.shape[0] + x_test.shape[0]) == x.shape[0]
