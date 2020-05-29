# PLTK
PLTK is a programming Language Toolkit used for tokenization and vecorization of text/code in machine learning.</br>
Example usage of the module is below, pltk.py is required in the current working directory (pip package maybe in future)</br>
</br>
</br>
import pltk as pl</br>
</br>
from pltk import load_data</br>
from pltk import get_file_contents  # Load single file</br>
from pltk import get_folder_contents  # Load all files in folder</br>
from pltk import tokenize  # create desired tokens</br>
from pltk import vectorize  # convert list of string to list of vectors</br>
from pltk import unvectorize  # convert list of vectors to list of strings</br>
from pltk import get_string  # Un-vectorize integer</br>
from pltk import get_vector  # Vectorize string</br>
from pltk import tokenize_file</br>
from pltk import tokenize_folder  # tokenize a list of files</br>
from pltk import vectorize_file</br>
from pltk import vectorize_folder</br>
from pltk import unvectorize_file</br>
from pltk import unvectorize_folder</br>
</br>
// from pltk import list_to_vectors  # convert list of string to list of vectors</br>
// from pltk import vectors_to_list  # convert list of vectors to list of strings</br>
from pltk import write_list_to_file  # writes a list to a file</br>
</br>
from pltk import change_list_dimensions</br>
from pltk import split_lists</br>
from pltk import join_lists</br>
</br>
</br>
# LOAD DATA
print('Load Data: ')</br>
test_load_file = load_data('good_strcpy.c')</br>
test_load_file_from_directory = load_data('/home/rod/PycharmProjects/ProgrammingLanguageToolkit/input/good_strcpy.c')</br>
test_load_folder = load_data('multifile')</br>
test_load_my_folder = load_data('/home/rod/PycharmProjects/ProgrammingLanguageToolkit/multifile')</br>
</br>
print('Load a file: ', test_load_file)</br>
print('Load a file from a directory: ', test_load_file_from_directory)</br>
print('Load a folder: ', test_load_folder)</br>
print('Load a user defined folder: ', test_load_my_folder)</br>
</br>
# TOKENIZE
char_tokens = tokenize(test_load_file, "chars")</br>
word_tokens = tokenize(test_load_file, "words")</br>
line_tokens = tokenize(test_load_file, "lines")</br>
function_tokens = tokenize(test_load_file, "functions")</br>
file_tokens = tokenize(test_load_file, "files")</br>
char_folder_tokens = tokenize(test_load_folder, "chars")</br>
word_folder_tokens = tokenize(test_load_folder, "words")</br>
line_folder_tokens = tokenize(test_load_folder, "lines")</br>
function_folder_tokens = tokenize(test_load_folder, "functions")</br>
file_folder_tokens = tokenize(test_load_folder, "files")</br>
</br>
print('\nTokenize Files: ')</br>
print('Char Tokens: ' , char_tokens)</br>
print('Word Tokens: ' , word_tokens)</br>
print('Line Tokens: ' , line_tokens)</br>
print('Function Tokens: ' , function_tokens)</br>
print('File Tokens: ' , file_tokens)</br>
print('\nTokenize Folders: ')</br>
print('Char Folder Tokens: ' , char_folder_tokens)</br>
print('Word Folder Tokens: ' , word_folder_tokens)</br>
print('Line Folder Tokens: ' , line_folder_tokens)</br>
print('Function Folder Tokens: ' , function_folder_tokens)</br>
print('File Folder Tokens: ' , file_folder_tokens)</br>
</br>
# VECTORIZE (CURREENT METHODS AND MY OWN)
from sklearn.feature_extraction.text import CountVectorizer</br>
vectorizer = CountVectorizer()  # create the transform</br>
vectorizer.fit(test_load_file[0])  # tokenize and build vocab</br>
print('\nSklearn\'s Count Vectorizer used with pltk\'s load_data method')</br>
print('Tokens: ', vectorizer.vocabulary_)  # summarize</br>
vector = vectorizer.transform(test_load_file[0])  # encode document</br>
print('Vector Shape (Lists,Tokens): ', vector.shape)  # summarize encoded vector</br>
print('Vectorized Array: ', vector.toarray())</br>
print('Count Vectorizer Features: ', vectorizer.get_feature_names())</br>
</br>
# VECTORIZATION
char_tokens_vectorized = vectorize_file(char_tokens)</br>
word_tokens_vectorized = vectorize_file(word_tokens)</br>
file_tokens_vectorized = vectorize_file(file_tokens)</br>
char_folder_tokens_vectorized = vectorize_folder(char_folder_tokens)</br>
word_folder_tokens_vectorized = vectorize_folder(word_folder_tokens)</br>
file_folder_tokens_vectorized = vectorize_folder(file_folder_tokens)</br>
</br>
print('\nVectorize a File into Tokens - chars, words, lines, functions, files: ')</br>
print('Vectorize Char Tokens: ', char_tokens_vectorized)</br>
print('Vectorize Word Tokens: ', word_tokens_vectorized)</br>
print('Vectorize File Tokens: ', file_tokens_vectorized)</br>
print('\nVectorize a Folder into Tokens - chars, words, lines, functions, files: ')</br>
print('Vectorize Char Tokens: ', char_folder_tokens_vectorized)</br>
print('Vectorize Word Tokens: ', word_folder_tokens_vectorized)</br>
print('Vectorize File Tokens: ', file_folder_tokens_vectorized)</br>
</br>
char_any_tokens_vectorized = vectorize(char_tokens)</br>
word_any_tokens_vectorized = vectorize(word_tokens)</br>
file_any_tokens_vectorized = vectorize(file_tokens)</br>
char_anyf_tokens_vectorized = vectorize(char_folder_tokens)</br>
word_anyf_tokens_vectorized = vectorize(word_folder_tokens)</br>
file_anyf_tokens_vectorized = vectorize(file_folder_tokens)</br>
</br>
print('\nPLTK Vectorize a File into Tokens - chars, words, lines, functions, files: ')</br>
print('PLTK Vectorize Char Tokens: ', char_any_tokens_vectorized)</br>
print('PLTK Vectorize Word Tokens: ', word_any_tokens_vectorized)</br>
print('PLTK Vectorize File Tokens: ', file_any_tokens_vectorized)</br>
print('\nPLTK Vectorize a Folder into Tokens - chars, words, lines, functions, files: ')</br>
print('PLTK Vectorize Char Tokens: ', char_anyf_tokens_vectorized)</br>
print('PLTK Vectorize Word Tokens: ', word_anyf_tokens_vectorized)</br>
print('PLTK Vectorize File Tokens: ', file_anyf_tokens_vectorized)</br>
</br>
# UNVECTORIZATION
char_file_tokens_unvectorized = unvectorize(char_tokens_vectorized)</br>
word_file_tokens_unvectorized = unvectorize(word_tokens_vectorized)</br>
file_file_tokens_unvectorized = unvectorize(file_tokens_vectorized)</br>
char_folder_tokens_unvectorized = unvectorize(char_folder_tokens_vectorized)</br>
word_folder_tokens_unvectorized = unvectorize(word_folder_tokens_vectorized)</br>
file_folder_tokens_unvectorized = unvectorize(file_folder_tokens_vectorized)</br>
</br>
print('\nUn-Vectorizing a file: ')</br>
print('Un-vectorized Char Tokens: ', char_file_tokens_unvectorized)</br>
print('Un-vectorized Word Tokens: ', word_file_tokens_unvectorized)</br>
print('Un-vectorized File Tokens: ', file_file_tokens_unvectorized)</br>
print('Un-vectorized Char Tokens: ', char_folder_tokens_unvectorized)</br>
print('Un-vectorized Word Tokens: ', word_folder_tokens_unvectorized)</br>
print('Un-vectorized File Tokens: ', file_folder_tokens_unvectorized)</br>
</br>
new_list_dimension = join_lists(word_file_tokens_unvectorized)</br>
new_list_dimension2d = split_lists(word_file_tokens_unvectorized)</br>
print('DIMENSION 1d file into 2d array: ', new_list_dimension)</br>
print('DIMENSION 1d file into 2d array: ', new_list_dimension2d)</br>
</br>
test1d = change_list_dimensions(new_list_dimension)</br>
test2d = change_list_dimensions(new_list_dimension2d)</br>
</br>
print('Test convert 1d: ', test1d)</br>
print('Test convert 2d: ', test2d)</br>
</br>
# Write to file
#write_list_to_file('test.c', test_unvectorize_list, folder=None)</br>
