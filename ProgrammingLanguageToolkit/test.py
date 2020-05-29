import pltk as pl

from pltk import load_data
from pltk import get_file_contents  # Load single file
from pltk import get_folder_contents  # Load all files in folder
from pltk import tokenize  # create desired tokens
from pltk import vectorize  # convert list of string to list of vectors
from pltk import unvectorize  # convert list of vectors to list of strings
from pltk import get_string  # Un-vectorize integer
from pltk import get_vector  # Vectorize string
from pltk import tokenize_file
from pltk import tokenize_folder  # tokenize a list of files
from pltk import vectorize_file
from pltk import vectorize_folder
from pltk import unvectorize_file
from pltk import unvectorize_folder

# from pltk import list_to_vectors  # convert list of string to list of vectors
# from pltk import vectors_to_list  # convert list of vectors to list of strings
from pltk import write_list_to_file  # writes a list to a file

from pltk import change_list_dimensions
from pltk import split_lists
from pltk import join_lists


# LOAD DATA
print('Load Data: ')
test_load_file = load_data('good_strcpy.c')
test_load_file_from_directory = load_data('/home/rod/PycharmProjects/ProgrammingLanguageToolkit/input/good_strcpy.c')
test_load_folder = load_data('multifile')
test_load_my_folder = load_data('/home/rod/PycharmProjects/ProgrammingLanguageToolkit/multifile')

print('Load a file: ', test_load_file)
print('Load a file from a directory: ', test_load_file_from_directory)
print('Load a folder: ', test_load_folder)
print('Load a user defined folder: ', test_load_my_folder)

# TOKENIZE
char_tokens = tokenize(test_load_file, "chars")
word_tokens = tokenize(test_load_file, "words")
line_tokens = tokenize(test_load_file, "lines")
function_tokens = tokenize(test_load_file, "functions")
file_tokens = tokenize(test_load_file, "files")
char_folder_tokens = tokenize(test_load_folder, "chars")
word_folder_tokens = tokenize(test_load_folder, "words")
line_folder_tokens = tokenize(test_load_folder, "lines")
function_folder_tokens = tokenize(test_load_folder, "functions")
file_folder_tokens = tokenize(test_load_folder, "files")

print('\nTokenize Files: ')
print('Char Tokens: ' , char_tokens)
print('Word Tokens: ' , word_tokens)
print('Line Tokens: ' , line_tokens)
print('Function Tokens: ' , function_tokens)
print('File Tokens: ' , file_tokens)
print('\nTokenize Folders: ')
print('Char Folder Tokens: ' , char_folder_tokens)
print('Word Folder Tokens: ' , word_folder_tokens)
print('Line Folder Tokens: ' , line_folder_tokens)
print('Function Folder Tokens: ' , function_folder_tokens)
print('File Folder Tokens: ' , file_folder_tokens)

# VECTORIZE (CURREENT METHODS AND MY OWN)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()  # create the transform
vectorizer.fit(test_load_file[0])  # tokenize and build vocab
print('\nSklearn\'s Count Vectorizer used with pltk\'s load_data method')
print('Tokens: ', vectorizer.vocabulary_)  # summarize
vector = vectorizer.transform(test_load_file[0])  # encode document
print('Vector Shape (Lists,Tokens): ', vector.shape)  # summarize encoded vector
print('Vectorized Array: ', vector.toarray())
print('Count Vectorizer Features: ', vectorizer.get_feature_names())

# VECTORIZATION
char_tokens_vectorized = vectorize_file(char_tokens)
word_tokens_vectorized = vectorize_file(word_tokens)
file_tokens_vectorized = vectorize_file(file_tokens)
char_folder_tokens_vectorized = vectorize_folder(char_folder_tokens)
word_folder_tokens_vectorized = vectorize_folder(word_folder_tokens)
file_folder_tokens_vectorized = vectorize_folder(file_folder_tokens)

print('\nVectorize a File into Tokens - chars, words, lines, functions, files: ')
print('Vectorize Char Tokens: ', char_tokens_vectorized)
print('Vectorize Word Tokens: ', word_tokens_vectorized)
print('Vectorize File Tokens: ', file_tokens_vectorized)
print('\nVectorize a Folder into Tokens - chars, words, lines, functions, files: ')
print('Vectorize Char Tokens: ', char_folder_tokens_vectorized)
print('Vectorize Word Tokens: ', word_folder_tokens_vectorized)
print('Vectorize File Tokens: ', file_folder_tokens_vectorized)

char_any_tokens_vectorized = vectorize(char_tokens)
word_any_tokens_vectorized = vectorize(word_tokens)
file_any_tokens_vectorized = vectorize(file_tokens)
char_anyf_tokens_vectorized = vectorize(char_folder_tokens)
word_anyf_tokens_vectorized = vectorize(word_folder_tokens)
file_anyf_tokens_vectorized = vectorize(file_folder_tokens)

print('\nPLTK Vectorize a File into Tokens - chars, words, lines, functions, files: ')
print('PLTK Vectorize Char Tokens: ', char_any_tokens_vectorized)
print('PLTK Vectorize Word Tokens: ', word_any_tokens_vectorized)
print('PLTK Vectorize File Tokens: ', file_any_tokens_vectorized)
print('\nPLTK Vectorize a Folder into Tokens - chars, words, lines, functions, files: ')
print('PLTK Vectorize Char Tokens: ', char_anyf_tokens_vectorized)
print('PLTK Vectorize Word Tokens: ', word_anyf_tokens_vectorized)
print('PLTK Vectorize File Tokens: ', file_anyf_tokens_vectorized)

# UNVECTORIZATION
char_file_tokens_unvectorized = unvectorize(char_tokens_vectorized)
word_file_tokens_unvectorized = unvectorize(word_tokens_vectorized)
file_file_tokens_unvectorized = unvectorize(file_tokens_vectorized)
char_folder_tokens_unvectorized = unvectorize(char_folder_tokens_vectorized)
word_folder_tokens_unvectorized = unvectorize(word_folder_tokens_vectorized)
file_folder_tokens_unvectorized = unvectorize(file_folder_tokens_vectorized)

print('\nUn-Vectorizing a file: ')
print('Un-vectorized Char Tokens: ', char_file_tokens_unvectorized)
print('Un-vectorized Word Tokens: ', word_file_tokens_unvectorized)
print('Un-vectorized File Tokens: ', file_file_tokens_unvectorized)
print('Un-vectorized Char Tokens: ', char_folder_tokens_unvectorized)
print('Un-vectorized Word Tokens: ', word_folder_tokens_unvectorized)
print('Un-vectorized File Tokens: ', file_folder_tokens_unvectorized)

new_list_dimension = join_lists(word_file_tokens_unvectorized)
new_list_dimension2d = split_lists(word_file_tokens_unvectorized)
print('DIMENSION 1d file into 2d array: ', new_list_dimension)
print('DIMENSION 1d file into 2d array: ', new_list_dimension2d)

test1d = change_list_dimensions(new_list_dimension)
test2d = change_list_dimensions(new_list_dimension2d)

print('Test convert 1d: ', test1d)
print('Test convert 2d: ', test2d)

# Write to file
#write_list_to_file('test.c', test_unvectorize_list, folder=None)