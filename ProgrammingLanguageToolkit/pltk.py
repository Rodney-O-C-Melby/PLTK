#  Copyright (c) 2020. Rodney Olav C. Melby, Royal Holloway University of London

import os
import math
"""
    Programming Language ToolKit (PLTK).
    Library to create, vectorize and write programming language data-sets using Sklearn/TensorFlow and machine learning.
    Tested with C although should be portable to any language by editing the vector the code (C=; Python=" ").
    Author: Rodney Olav C. Melby
    Date: 02/02/2020 - 28/02/2020
"""


def load_data(input_data=None):
    """
        Function to load data from a local file or folder.
        :param input_data: String of the directory or file to load data from.
        :return list: Strings of loaded data.
    """
    # check if user provided no params
    if input_data is None:  # auto load all file in input directory
        return get_folder_contents(None)

    # check if user provided directory
    if os.path.isdir(input_data):
        return get_folder_contents(input_data)

    # check if user provided file
    if os.path.isfile(input_data):
        check = input_data.find('/')  # check for slash in path (user provided directory)
        string_size = len(input_data)

        # slash at first character, so full file path file
        if check == 0:
            length = string_size
            base = 0
            break_point = 0
            while length > base:
                slash_positions = input_data.find('/', length)
                # break when first slash found in reverse count of string
                if slash_positions == -1:
                    break_point = length

                length = length - 1

            # print('path: ', input_data[0:break_point])
            # print('filename: ', input_data[break_point:string_size])
            return get_file_contents(input_data[break_point:string_size], input_data[0:break_point])

        # single file in current directory
        elif check == -1:
            return get_file_contents(input_data, None)

    # check if user provided url TODO


# read a file into list line by line
def get_file_contents(filename, directory=None):
    """
    Loads the provided file contents into a list of tokens.
    :param filename: The name of the file to load.
    :param directory: The directory to load the file from, default directory='/input'.
    :return: list of file contents as lines.
    """
    results = []
    if directory is None:
        directory = os.getcwd() + "/input"  # default is current working directory
    else:
        directory = directory  # user specified directory

    if not os.path.exists(directory) and not os.path.isfile(directory):  # creates directory if it doesn't exist
        try:
            os.makedirs(directory)
        except OSError:
            print("Cannot create directory, cannot recursively create directories, only one at a time!")

    path = directory + "/" + filename  # create file path
    try:
        with open(path) as file:  # open and close file
            lines = file.readlines()  # read all lines
        results.append(lines)  # add line to list
    except FileNotFoundError:
        print("No input samples in " + directory)

    return results  # return list


# read each file into a list
def get_folder_contents(directory):
    """
        Recursively adds files from provided directory to a list of lists, one list per file.
        :param directory: String Directory to recursively load files.
        :rtype List Matrix of file data, one list per file.
        :return List : returns a list of files.
    """
    if directory is None:
        directory = os.getcwd() + "/input"  # default is current working directory
    else:
        directory = directory  # user specified directory

    if not os.path.exists(directory) and not os.path.isdir(directory):  # creates directory if it doesn't exist
        try:
            os.makedirs(directory)
        except OSError:
            print("Cannot create directory, cannot recursively create directories, only one at a time!")

    folder_results = []

    for filename in os.listdir(directory):  # for each file in directory
        path = directory + "/" + filename  # create file path
        try:
            with open(path) as file:  # open and close file
                lines = file.readlines()  # read all lines
                folder_results.append(lines)  # add line to list
        except FileNotFoundError:
            print("No input samples in " + directory)

    return folder_results  # return list


def tokenize_file(file, split="lines", separator=")"):
    """
    Separates lists into tokens of either(chars, words, line, function, file).
    :param file: List to tokenize.
    :param split: type of tokenization ("char", "word", "line", "function", "file").
    :param separator: optional separator for function, only when using "function" type.
    :return: A new list of tokens, ready for vectorization or feature extraction.
    """
    new_file = []
    for line in file:  # for each file
        if split == "chars":  # separate into chars
            for character in line:
                new_file.append(character)
        elif split == "words":  # separate into words
            token = line.rpartition(" ")
            for word in token:
                if word != "":
                    new_file.append(word)
        elif split == "lines":  # separate into lines
            token = line.rpartition("/n")
            for word in token:
                if word != "":
                    new_file.append(word)
        elif split == "functions":  # separate into end of line/functions
            token = line.rpartition(separator)
            for word in token:
                if word != "":
                    new_file.append(word)

    # end of lines for loop
    if split == "files":  # separate into file
        temp = ""
        for word in file:
            if word != "":
                temp += word
        new_file.append(temp)

    return new_file


# tokenize a loaded folder of files
def tokenize_folder(the_list, split="lines", separator=")"):
    """
    Vectorize's a list.
    :param split:
    :param separator:
    :param the_list: list of strings to vectorize
    :return: new list of vectorized strings
    """
    files = []
    for sample in the_list:
        the_token = tokenize_file(sample, split, separator)
        files.append(the_token)
    return files


# tokenize data
def tokenize(data, split="lines", separator=")"):

    length = len(data)
    # tokenize file
    if length == 1:
        return tokenize_file(data[0], split, separator)
    # tokenize folder
    elif length > 1:
        return tokenize_folder(data, split, separator)


# converts string to unique number (using bytes and little endian - hint: needs to be unique and reversible)
def get_vector(string):
    """
    Converts string to unique number (using bytes and little endian - hint: needs to be unique and reversible).
    :param string: The string to vectorize.
    :return: Unique integer of input string.
    """
    return int.from_bytes(string.encode(), 'little')  # python3 requires bytes (python strings)


# converts unique number back to original string data - python3 requires bytes (python strings)
def get_string(number):
    """
    Converts unique number back to original string data - python3 requires bytes (python strings).
    :param number: the vector to convert back to a string.
    :return: string value of given integer.
    """
    return number.to_bytes(math.ceil(number.bit_length() / 8), 'little').decode()


# vectorize a list
def vectorize_file(the_list):
    """
    Vectorize's a list.
    :param the_list: list of strings to vectorize
    :return: new list of vectorized strings
    """
    vectors = []
    for token in the_list:
        vector = get_vector(token)
        vectors.append(vector)
    return vectors


# vectorize a folder
def vectorize_folder(the_list):
    new_list = []
    for file in the_list:
        folder_vectors = vectorize_file(file)
        new_list.append(folder_vectors)

    return new_list


def vectorize(data_tokens):
    try:
        vectors = vectorize_file(data_tokens)  # tokenize file
    except:
        vectors = vectorize_folder(data_tokens)  # tokenize folder
    return vectors


# unvectorize a list
def unvectorize(the_list):
    """
    Un-vectorize's a list.
    :param the_list: list of vectors to un-vectorize
    :return: un-vectorized list of strings
    """
    try:
        text = unvectorize_file(the_list)
    except:
        text = unvectorize_folder(the_list)
    return text


# unvectorize a list
def unvectorize_file(the_list):
    """
    Un-vectorize's a list.
    :param the_list: list of vectors to un-vectorize
    :return: un-vectorized list of strings
    """
    strings = []
    for vector in the_list:
        word = get_string(vector)
        strings.append(word)
    return strings


# unvectorize a list
def unvectorize_folder(the_list):
    """
    Un-vectorize's a list.
    :param the_list: list of vectors to un-vectorize
    :return: un-vectorized list of list of strings
    """
    files = []
    for file in the_list:
        cleartext = unvectorize_file(file)
        files.append(cleartext)
    return files


# converts list dimensionality (1d-2d, 2d-1d)
def change_list_dimensions(the_list):
    islist = isinstance(the_list, list)

    if islist:
        isfilelist = isinstance(the_list[0], list)

        # checks for list in list
        if isfilelist:
            # 2d array
            join_lists(the_list[0])
        else:
            split_lists(the_list)
        #print('isfilelist: ', isfilelist)


# converts 1d list into 2d list (2d = 2 dimensional)
def split_lists(the_list):
    new_list = []
    second_dimension = []
    for entry in the_list:
        new_list.append(entry)
    second_dimension.append(new_list)
    return second_dimension


# converts 2d list into 1d list (2d = 2 dimensional)
def join_lists(the_list):
    new_list = []
    for entry in the_list:
        new_list.append(entry)
    return new_list


# writes a list to a file
def write_list_to_file(file_name, file_list=None, folder=None):
    """
    Writes a list to a file.
    :param file_name: String of file name to write to
    :param file_list: List of data to write to file
    :param folder: String of directory to write to
    :return: None as per the default behaviour of Python
    """
    if file_list is None:
        file_list = []
    if folder is None:
        folder = os.getcwd() + "/output"  # default is current working directory /output
    else:
        folder = folder  # user specified directory

        try:
            folder = os.getcwd()
        except OSError:
            print("Cannot get current working directory, this library requires read/write/execute permissions for the "
                  "file owner or group.")
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + "/" + file_name, "w") as fp:
        fp.writelines(file_list)
    fp.close()
    print('Written ' + file_name + " to " + folder)
    return None
