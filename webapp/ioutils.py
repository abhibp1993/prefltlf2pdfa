# import abc
# import logic
import simplejson as json
from loguru import logger

import translate2


def to_json(fpath, obj_dict):
    """
    Save the given dictionary to JSON file.

    :param fpath:
    :param obj_dict:
    :return:
    """
    with open(fpath, "w") as file:
        json.dump(obj_dict, file, indent=2, default=custom_encoder, tuple_as_array=False)


def to_json_str(obj_dict):
    """
    Save the given dictionary to JSON file.

    :param obj_dict:
    :return:
    """
    return json.dumps(obj_dict, indent=2, default=custom_encoder, tuple_as_array=False)


def from_json(fpath):
    """
    Loads an object dictionary from JSON file.
    :param fpath:
    :return:
    """
    with open(fpath, "r") as file:
        obj_dict = json.load(file, object_hook=custom_decoder)
    return obj_dict


def custom_encoder(py_obj):
    if isinstance(py_obj, tuple):
        py_obj = {"__type__": "tuple", "__value__": list(py_obj)}
    elif isinstance(py_obj, set):
        py_obj = {"__type__": "set", "__value__": list(py_obj)}
    elif isinstance(py_obj, translate2.PrefAutomaton):
        py_obj = {"__type__": "PrefAutomaton", "__value__": py_obj.serialize()}
    else:
        raise TypeError(f"{py_obj} of type:{type(py_obj)} could not be encoded by JSON.")
    return py_obj


def custom_decoder(json_obj):
    if "__type__" in json_obj and json_obj["__type__"] == "tuple":
        return tuple(json_obj["__value__"])
    if "__type__" in json_obj and json_obj["__type__"] == "set":
        return set(json_obj["__value__"])
    if "__type__" in json_obj and json_obj["__type__"] == "PrefAutomaton":
        return translate2.PrefAutomaton.deserialize(json_obj["__value__"])
    return json_obj

