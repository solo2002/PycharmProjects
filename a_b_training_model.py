#!/apollo/bin/env python -tt


# This script (runs every month) reads book meta-data (trimDescription-books-1.txt) and converts to following format:
# (all lower case, only number and letters), for instance:
# 0001004492,1     barklem  jill  sea story  children books   0
# 0001004875,1     jets on tape  morpurgo  michael  mossop s last chance  jets on tape   fiction  children s fiction    0
# For each month, all the adult related asins and similar number of unadult related asins are selected, and their descriptions are
# converted to a list of binary numbers (for instance, [1 0 1 1...]), which would be used as input to train random forest model
#
# Please note this script is going to read raw_input_file twice: first time, calculate the ratio of adult books to unadult books;
# second time, select all the adult books and similar number of undault books based on the ratio (probability), and then output to
# updated_input_file
#
# Please note this script is going to read updated_input_file twice, too: first time, calculate all words frequency and construct
# top_words_dict; second time, convert the updated_input_file into input based on top_words_dict
#
# The trained model (random forest) and a top words dictionary that are required during applying model stage, and they are saved at:
# /sims-extract-data/adult-book-scorer-trained-model/${REGION_LOWERCASE}/latest/
#

import PyRODB
import numpy as np
import stanza
import cPickle
import random
import multiprocessing
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report # Please keep this line temporarily, which would be used when tuning the model
from sklearn import cross_validation # Please keep this line temporarily, which would be used when tuning the model
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from operator import itemgetter
from SimilaritiesPythonUtilities import Common, CommonFileSystem

STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'kindle', 'book',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'english', 'hardcover',
    'edition', 'books', 'ebook', 'paperback', 'one', 'paperback_meta_binding', 'abis_book', 'other_meta_binding',
    'hardcover_meta_binding', 'cd_rom', 'audio_cassette_meta_binding', 'audiobooks_meta_binding', 'spiral_bound',
    'kindle_edition', 'kindle_meta_binding', 'mobipocket_ebook', 'abis_ebooks',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '00', '01', '02',
    '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '19', '20',
    '21', '22', '23','24', '25', '26', '27','28', '29', '30', '31', '32', '696','1990', '1993', '1994', '1995', '1996',
    '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
    '2010', '2011', '2012', '2013', '2014', '2015', '2016', '813', '75', '873',
    '5510500', '800000', '600000', '178000', '828000', '55101500', 'gt', 'lt', 'br', 'li', 'la', 'rsquo', 'ldquo', 'rdquo']

STOP_WORDS_DICT = dict.fromkeys(STOP_WORDS, True)
STANZA_SECTION = "adult-book-scorer-training-model"

def main():
    config = Common.get_stanza_section_from_config(STANZA_SECTION)
    try:
        remoteEroticaEbookBrowseNodeFile = config['remote-erotica-ebook-browse-node-file']
        localEroticaEbookBrowseNodeFile = config['local-latest-erotica-ebook-browse-node-file']
        localItemRodb = config['local-item-rodb']
        localBookDescriptionData = config['local-trim-descriptions-books']
        localUpdatedInput = config['local-updated-input-file']
        localTrainedModel = config['trained-model-dir']
        localTopWordsDict = config['top-words-dict-dir']
    except Exception as e:
        print "[FATAL] Error retrieving configuration parameters!"
        raise

    remote_latest_browse_node_file = CommonFileSystem.get_latest_remote_filename(remoteEroticaEbookBrowseNodeFile)
    CommonFileSystem.rsync_get(remote_latest_browse_node_file, localEroticaEbookBrowseNodeFile)
    erotica_input = open(localEroticaEbookBrowseNodeFile, 'r')
    db = PyRODB.open(localItemRodb)
    trim_book_description_input = open(localBookDescriptionData, 'r')

    # construct a map for erotica ebook browse node asins
    erotica_map = {}
    for line in erotica_input:
        content = line.split('\t')
        erotica_map[content[0]] = True

    updated_input_file_for_training_model = open(localUpdatedInput, 'w+')
    update_dataset_for_training_model(trim_book_description_input, erotica_map, db, updated_input_file_for_training_model)
    updated_input_file_for_training_model.seek(0, 0)

    # a dictionary to record the frequency for each word in the file
    word_counter = {}
    construct_word_counter(word_counter, updated_input_file_for_training_model)

    # sort the dictionary, and return a list of sorted words according to the highest item frequency
    popular_words = sorted(word_counter, key=word_counter.get, reverse=True)

    print 'Size of popular_words', len(popular_words)
    # select top most frequent words as features
    top_words_set = popular_words[:2000]
    for word in top_words_set:
        print word, word_counter[word]
    top_words_dictionary = {w: True for w in top_words_set}
    word_counter = {}
    #top_words_dictionary['flute'] = True
    #top_words_dictionary['piano'] = True
    #top_words_dictionary['violin'] = True

    updated_input_file_for_training_model.seek(0, 0)
    data_matrix = []
    to_binary_data_matrix(updated_input_file_for_training_model, top_words_dictionary, data_matrix)

    # to make sure the data is randomized
    random.shuffle(data_matrix)

    input_data = []
    input_asin = []
    input_target = []
    convert_data_matrix_to_input_list(data_matrix, input_asin, input_data, input_target)
    #mnb = MultinomialNB()
    forest = RandomForestClassifier(n_jobs=multiprocessing.cpu_count() - 1,criterion='entropy', n_estimators=100,
                                    min_samples_leaf=5, min_samples_split=5)
    # n_estimators=100, criterion='entropy',

    print "Now training the random forest (this may take a while)..."
    #mnb.fit(input_data[0:], input_target[0:])
    forest.fit(input_data[0:], input_target[0:])
    #forest.fit(input_data[:593680], input_target[:593680])
    '''
    false_positive_asin = []
    false_negative_asin = []
    label_pred = forest.predict(input_data[593680:])
    true_positive = 0  # predict = '1', label (target) = '1',
    true_negative = 0  # predict = '0', label (target) = '0',
    false_positive = 0  # predict = '1', label (target) = '0',
    false_negative = 0  # predict = '0', label (target) = '1',
    for i in range(593680, len(input_data)):
        if label_pred[i - 593680] == '1' and input_target[i] == '1':
            true_positive += 1
        elif label_pred[i - 593680] == '0' and input_target[i] == '0':
            true_negative += 1
        elif label_pred[i - 593680] == '1' and input_target[i] == '0':
            false_positive += 1
            false_positive_asin.append(input_asin[i])
        elif label_pred[i - 593680] == '0' and input_target[i] == '1':
            false_negative += 1
            false_negative_asin.append(input_asin[i])
    print 'For top', len(top_words_set), 'words: '
    print 'true_positive: ', true_positive
    print 'true_negative: ', true_negative
    print 'false_positive', false_positive, 'false_positive_asin', len(false_positive_asin)
    print 'false_negative ', false_negative, 'false_negative_asin', len(false_negative_asin)
    print(classification_report(input_target[593680:], label_pred, digits=4))
    #false_negative_asin.sort()
    false_positive_asin.sort()
    print 'false_positive_asin', false_positive_asin
    print '===================================='
    #print 'false_negative_asin', false_negative_asin

    #print 'Please keep following two line temporarily, which would be used when tuning the model'
    #scores = cross_validation.cross_val_score(forest, input_data, input_target, cv=2)
    #print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    print 'Print the parameters of the trained random forest:'
    print 'the number of features (number of words selected)', len(top_words_set)
    print 'Other parameters'
    print forest.get_params()

    # use a full grid over all parameters
    param_grid = {
       "min_samples_leaf": [10, 100, 500, 1000],
        "min_samples_split": [10, 100, 500, 1000],
        "max_features": [10, 45, 100],
    }

    clf = RandomForestClassifier(n_jobs=multiprocessing.cpu_count() - 1)

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()
    grid_search.fit(input_data[:len(input_data)], input_target[:len(input_data)])

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.grid_scores_)))
    report(grid_search.grid_scores_)
    '''
    # Save 'trained_model' and 'topWordsDict' locally, which would be install to NFS
    #with open(localTrainedModel, 'wb') as f:
     #   cPickle.dump(forest, f)
    with open(localTrainedModel, 'wb') as f:
        cPickle.dump(forest, f)

    with open(localTopWordsDict, 'wb') as topWordsDict:
        cPickle.dump(top_words_dictionary, topWordsDict)

    erotica_input.close()
    trim_book_description_input.close()
    updated_input_file_for_training_model.close()




# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

# This routine determines the key exist in RODB or not
def exist_in_RODB(key, db):
    if (db.exists(key)):
        val = db.get(key)
        assert isinstance(val, object)
        val_list = val.split(',')

        # To avoid incorrect format of item.rodb, the normal format should be with length of 12
        if len(val_list) == 12:
            return True
        else:
            return False
    else:
        return False

# This routine determines the key is an adult related book or not based on item.rodb
def is_adult_book_from_RODB(key, db):
    val = db.get(key)
    val_list = val.split(',')

    # val_list[5] indicates it is adult related product or not
    # Since the raw_input_file only contains books' asins, here we just skip to
    # check it again (website_display_ID == '3' or website_display_ID == '337', books or ebooks, respectively)
    if val_list[5]:
        return True
    else:
        return False

# This routine reads book meta-data (trimDescription-books-1.txt) twice.
# First time, it counts the number of adult book asin and unadult book asin, and then calculate the ratio
# Second time, it selects all the adult book asins and some of unadult book asins based on the ratio got from first time,
# replaces comma by space, and adds label for each asin based on item.rodb and ebook erotica browse node data
def update_dataset_for_training_model(raw_input_file, erotica_map, db, updated_input_file_for_training_model):
    adult_book_asin_counter = 0
    un_adult_asin_counter = 0
    adult_to_non_adult_ratio = 0

    for line in raw_input_file:
        line_content = line.split('\t')
        current_key = line_content[0]
        if exist_in_RODB(current_key, db):
            if is_adult_book_from_RODB(current_key, db) or erotica_map.__contains__(current_key):
                adult_book_asin_counter += 1
            else:
                un_adult_asin_counter += 1

    # Calculates the ratio, i.e., 0.01683
    adult_to_non_adult_ratio = round(float(adult_book_asin_counter) / un_adult_asin_counter, 5)

    # For taking care extremely case when more adult books than unadult books
    if adult_to_non_adult_ratio >= 1:
        adult_to_non_adult_ratio = 1

    # Print out the ratio
    print 'The number of all adult book asin', adult_book_asin_counter
    print 'The number of all unadult book asin', un_adult_asin_counter
    print 'The ration of number of adult book to number of unadult book', adult_to_non_adult_ratio

    adult_book_asin_counter = 0
    output_un_adult_book_asin_counter = 0

    # Move cursor to the begining of file, since it needs to read the file one more time

    raw_input_file.seek(0, 0)
    for line in raw_input_file:
        line_content = line.split('\t')
        current_key = line_content[0]
        if exist_in_RODB(current_key, db):
            if is_adult_book_from_RODB(current_key, db) or erotica_map.__contains__(current_key):
                adult_book_asin_counter += 1
                updated_input_file_for_training_model.writelines(current_key + '\t')
                descri = line_content[1].strip().split(',')
                for word in descri:
                    if not STOP_WORDS_DICT.__contains__(word):
                        updated_input_file_for_training_model.writelines(["%s " % word])
                updated_input_file_for_training_model.writelines('\t' + '1\n')

            else:
                if random.random() < 3 * adult_to_non_adult_ratio:
                    output_un_adult_book_asin_counter += 1
                    updated_input_file_for_training_model.writelines(current_key + '\t')
                    descri = line_content[1].strip().split(',')
                    for word in descri:
                        if not STOP_WORDS_DICT.__contains__(word):
                            updated_input_file_for_training_model.writelines(["%s " % word])
                    updated_input_file_for_training_model.writelines('\t' + '0\n')
    '''
    raw_input_file.seek(0, 0)
    line_counter = 0
    first_part_line_counter = 0
    second_part_line_counter = 0
    for line in raw_input_file:
        line_content = line.split('\t')
        current_key = line_content[0]
        if exist_in_RODB(current_key, db):
            line_counter += 1;
            if is_adult_book_from_RODB(current_key, db) or erotica_map.__contains__(current_key):
                adult_book_asin_counter += 1
                updated_input_file_for_training_model.writelines(current_key + '\t')
                descri = line_content[1].strip().split(',')
                for word in descri:
                    if not STOP_WORDS_DICT.__contains__(word):
                        updated_input_file_for_training_model.writelines(["%s " % word])
                updated_input_file_for_training_model.writelines('\t' + '1\n')

            else:
                if line_counter < 5000000:      # try to select data from beginning
                    if random.random() < (9 * adult_to_non_adult_ratio):                         # double here
                        first_part_line_counter += 1
                        output_un_adult_book_asin_counter += 1
                        updated_input_file_for_training_model.writelines(current_key + '\t')
                        descri = line_content[1].strip().split(',')
                        for word in descri:
                            if not STOP_WORDS_DICT.__contains__(word):
                                updated_input_file_for_training_model.writelines(["%s " % word])
                        updated_input_file_for_training_model.writelines('\t' + '0\n')
                else:
                    if random.random() < (0.5 * adult_to_non_adult_ratio):  # double here
                        second_part_line_counter += 1
                        output_un_adult_book_asin_counter += 1
                        updated_input_file_for_training_model.writelines(current_key + '\t')
                        descri = line_content[1].strip().split(',')
                        for word in descri:
                            if not STOP_WORDS_DICT.__contains__(word):
                                updated_input_file_for_training_model.writelines(["%s " % word])
                        updated_input_file_for_training_model.writelines('\t' + '0\n')
    '''

    print 'The number of selected adult book asin', adult_book_asin_counter
    print 'The number of selected unadult book asin', output_un_adult_book_asin_counter
    #print 'From first part we select ', first_part_line_counter
    #print 'From second part we select', second_part_line_counter

# This routine converts updated input file to data matrix based on it exists in top word set or not
# Data matrix would be used for preparing the input for training
# data_matrix format:
# [['B01FY1LAKI,1', [1, 0, 1, 0, 1, 0, 0, 1, 0, 0], '0'],]
# The reason we need data_matrix instead of adding data directly to input_asin, input_data, and input_target
# is that we can randomize data_matrix without changing the relative sequence, which is necessary before
# training the model
def to_binary_data_matrix(input, top_words_dictionary, data_matrix):
    for line in input:
        # In case there is a blank line in the output file
        #if line.rstrip():
        data_vector = []
        word_dict = dict.fromkeys(top_words_dictionary.keys(), 0)
        content = line.split('\t')

        # To skip the asin that is without any word in the top words dictionary
        if not content[1] == '':
            data_vector.append(content[0])
            description_list = content[1].split()
            for word in description_list:
                if top_words_dictionary.__contains__(word):
                    word_dict[word] = 1
            data_vector.append(word_dict.values())
            data_vector.append(content[2].strip()) # To remove '\n'
            data_matrix.append(data_vector)

# This routine converts a data matrix to a list of asin, a list of input_data, and a list of input_target
# For instance, convert [['B01FY1LAKI,1', [1, 0, 1, 0, 1, 0, 0, 1, 0, 0], '0'],] to
# input_asin: [['B01FY1LAKI,1'],...]
# input_data: [[1, 0, 1, 0, 1, 0, 0, 1, 0, 0],...]
# input_target: [['0'],...]
def convert_data_matrix_to_input_list(data_matrix, input_asin, input_data,input_target):
    for list in data_matrix:
        input_asin.append(list[0])
        input_data.append(list[1])
        input_target.append(list[2])

# This routine calculates all the words frequency from input file
def construct_word_counter(word_counter, updated_input_file_for_training_model):
    counter = 0
    for line in updated_input_file_for_training_model:
        content = line.split('\t')
        if content[2].strip() == '1':
            counter += 1
            description_list = content[1].split()

            # skip the asin that is without any extracted information
            if not content[1] == '':
                for word in description_list:
                    if not len(word) == 1 and (not word.isdigit()):  # remove number and single letter
                        #if len(word_counter) < 1000000: # set a maximum size of word_counter to avoid memory errors
                        if word in word_counter:
                            word_counter[word] = word_counter[word] + 1
                        else:
                            word_counter[word] = 1
    print 'The number of asins is selected to construct topWordsList ', counter



####################### main #########################
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print "[FATAL] " + str(e)
        raise


'''
import PyRODB
import stanza
import cPickle
import random
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report # Please keep this line temporarily, which would be used when tuning the model
from sklearn import cross_validation # Please keep this line temporarily, which would be used when tuning the model
from SimilaritiesPythonUtilities import Common, CommonFileSystem

STOP_WORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'kindle', 'book',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'english', 'hardcover',
    'edition', 'books', 'ebook', 'paperback', 'one', 'paperback_meta_binding', 'abis_book', 'other_meta_binding',
    'hardcover_meta_binding', 'cd_rom', 'audio_cassette_meta_binding', 'audiobooks_meta_binding', 'spiral_bound',
    'kindle_edition', 'kindle_meta_binding', 'mobipocket_ebook', 'abis_ebooks',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '00', '01', '02',
    '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '19', '20',
    '21', '22', '23','24', '25', '26', '27','28', '29', '30', '31', '32', '696','1990', '1993', '1994', '1995', '1996',
    '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
    '2010', '2011', '2012', '2013', '2014', '2015', '2016', '813', '75', '873',
    '5510500', '800000', '600000', '178000', '828000', '55101500', 'gt', 'lt', 'br', 'li', 'la', 'rsquo', 'ldquo', 'rdquo']

STOP_WORDS_DICT = dict.fromkeys(STOP_WORDS, True)
STANZA_SECTION = "adult-book-scorer-training-model"

def main():
    config = Common.get_stanza_section_from_config(STANZA_SECTION)
    try:
        remoteEroticaEbookBrowseNodeFile = config['remote-erotica-ebook-browse-node-file']
        localEroticaEbookBrowseNodeFile = config['local-latest-erotica-ebook-browse-node-file']
        localItemRodb = config['local-item-rodb']
        localBookDescriptionData = config['local-trim-descriptions-books']
        localUpdatedInput = config['local-updated-input-file']
        localTrainedModel = config['trained-model-dir']
        localTopWordsDict = config['top-words-dict-dir']
    except Exception as e:
        print "[FATAL] Error retrieving configuration parameters!"
        raise

    remote_latest_browse_node_file = CommonFileSystem.get_latest_remote_filename(remoteEroticaEbookBrowseNodeFile)
    CommonFileSystem.rsync_get(remote_latest_browse_node_file, localEroticaEbookBrowseNodeFile)
    erotica_input = open(localEroticaEbookBrowseNodeFile, 'r')
    db = PyRODB.open(localItemRodb)
    trim_book_description_input = open(localBookDescriptionData, 'r')

    # construct a map for erotica ebook browse node asins
    erotica_map = {}
    for line in erotica_input:
        content = line.split('\t')
        erotica_map[content[0]] = True

    updated_input_file_for_training_model = open(localUpdatedInput, 'w+')
    update_dataset_for_training_model(trim_book_description_input, erotica_map, db, updated_input_file_for_training_model)
    updated_input_file_for_training_model.seek(0, 0)

    # a dictionary to record the frequency for each word in the file
    word_counter = {}
    construct_word_counter(word_counter, updated_input_file_for_training_model)

    # sort the dictionary, and return a list of sorted words according to the highest item frequency
    popular_words = sorted(word_counter, key=word_counter.get, reverse=True)

    # select top most frequent words as features
    top_words_set = popular_words[:2000]
    top_words_dictionary = {w: True for w in top_words_set}
    updated_input_file_for_training_model.seek(0, 0)
    data_matrix = []
    to_binary_data_matrix(updated_input_file_for_training_model, top_words_dictionary, data_matrix)

    # to make sure the data is randomized
    random.shuffle(data_matrix)

    input_data = []
    input_asin = []
    input_target = []
    convert_data_matrix_to_input_list(data_matrix, input_asin, input_data, input_target)

    forest = RandomForestClassifier(n_estimators=100, criterion='entropy', min_samples_split=10,
                                    n_jobs=multiprocessing.cpu_count() - 1)

    # Please keep following two line temporarily, which would be used when tuning the model
    # scores = cross_validation.cross_val_score(forest, input_data, input_target, cv=2)
    # print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    print "Now training the random forest (this may take a while)..."
    print 'len(input_data)', len(input_data), 'input_target', len(input_target)
    forest = forest.fit(input_data[:len(input_data)], input_target[:len(input_data)])

    print 'Print the parameters of the trained random forest:'
    print 'the number of features (number of words selected)', len(top_words_set)
    print 'Other parameters'
    print forest.get_params()

    # Save 'trained_model' and 'topWordsDict' locally, which would be install to NFS
    with open(localTrainedModel, 'wb') as f:
        cPickle.dump(forest, f)

    with open(localTopWordsDict, 'wb') as topWordsDict:
        cPickle.dump(top_words_dictionary, topWordsDict)

    erotica_input.close()
    trim_book_description_input.close()
    updated_input_file_for_training_model.close()

# This routine determines the key exist in RODB or not
def exist_in_RODB(key, db):
    if (db.exists(key)):
        val = db.get(key)
        assert isinstance(val, object)
        val_list = val.split(',')

        # To avoid incorrect format of item.rodb, the normal format should be with length of 12
        if len(val_list) == 12:
            return True
        else:
            return False
    else:
        return False

# This routine determines the key is an adult related book or not based on item.rodb
def is_adult_book_from_RODB(key, db):
    val = db.get(key)
    val_list = val.split(',')

    # val_list[5] indicates it is adult related product or not
    # Since the raw_input_file only contains books' asins, here we just skip to
    # check it again (website_display_ID == '3' or website_display_ID == '337', books or ebooks, respectively)
    if val_list[5]:
        return True
    else:
        return False

# This routine reads book meta-data (trimDescription-books-1.txt) twice.
# First time, it counts the number of adult book asin and unadult book asin, and then calculate the ratio
# Second time, it selects all the adult book asins and some of unadult book asins based on the ratio got from first time,
# replaces comma by space, and adds label for each asin based on item.rodb and ebook erotica browse node data
def update_dataset_for_training_model(raw_input_file, erotica_map, db, updated_input_file_for_training_model):
    adult_book_asin_counter = 0
    un_adult_asin_counter = 0
    adult_to_non_adult_ratio = 0

    for line in raw_input_file:
        line_content = line.split('\t')
        current_key = line_content[0]
        if exist_in_RODB(current_key, db):
            if is_adult_book_from_RODB(current_key, db) or erotica_map.__contains__(current_key):
                adult_book_asin_counter += 1
            else:
                un_adult_asin_counter += 1

    # Calculates the ratio, i.e., 0.01683
    adult_to_non_adult_ratio = round(float(adult_book_asin_counter) / un_adult_asin_counter, 5)

    # For taking care extremely case when more adult books than unadult books
    if adult_to_non_adult_ratio >= 1:
        adult_to_non_adult_ratio = 1

    # Print out the ratio
    print 'The number of all adult book asin', adult_book_asin_counter
    print 'The number of all unadult book asin', un_adult_asin_counter
    print 'The ration of number of adult book to number of unadult book', adult_to_non_adult_ratio

    adult_book_asin_counter = 0
    output_un_adult_book_asin_counter = 0

    # Move cursor to the begining of file, since it needs to read the file one more time
    raw_input_file.seek(0, 0)

    for line in raw_input_file:
        line_content = line.split('\t')
        current_key = line_content[0]
        if exist_in_RODB(current_key, db):
            if is_adult_book_from_RODB(current_key, db) or erotica_map.__contains__(current_key):
                adult_book_asin_counter += 1
                updated_input_file_for_training_model.writelines(current_key + '\t')
                descri = line_content[1].strip().split(',')
                for word in descri:
                    if not STOP_WORDS_DICT.__contains__(word):
                        updated_input_file_for_training_model.writelines(["%s " % word])
                updated_input_file_for_training_model.writelines('\t' + '1\n')

            else:
                if random.random() < adult_to_non_adult_ratio:
                    output_un_adult_book_asin_counter += 1
                    updated_input_file_for_training_model.writelines(current_key + '\t')
                    descri = line_content[1].strip().split(',')
                    for word in descri:
                        if not STOP_WORDS_DICT.__contains__(word):
                            updated_input_file_for_training_model.writelines(["%s " % word])
                    updated_input_file_for_training_model.writelines('\t' + '0\n')

    print 'The number of selected adult book asin', adult_book_asin_counter
    print 'The number of selected unadult book asin', output_un_adult_book_asin_counter

# This routine converts updated input file to data matrix based on it exists in top word set or not
# Data matrix would be used for preparing the input for training
# data_matrix format:
# [['B01FY1LAKI,1', [1, 0, 1, 0, 1, 0, 0, 1, 0, 0], '0'],]
# The reason we need data_matrix instead of adding data directly to input_asin, input_data, and input_target
# is that we can randomize data_matrix without changing the relative sequence, which is necessary before
# training the model
def to_binary_data_matrix(input, top_words_dictionary, data_matrix):
    for line in input:
        # In case there is a blank line in the output file
        #if line.rstrip():
        data_vector = []
        word_dict = dict.fromkeys(top_words_dictionary.keys(), 0)
        content = line.split('\t')

        # To skip the asin that is without any word in the top words dictionary
        if not content[1] == '':
            data_vector.append(content[0])
            description_list = content[1].split()
            for word in description_list:
                if top_words_dictionary.__contains__(word):
                    word_dict[word] = 1
            data_vector.append(word_dict.values())
            data_vector.append(content[2].strip()) # To remove '\n'
            data_matrix.append(data_vector)

# This routine converts a data matrix to a list of asin, a list of input_data, and a list of input_target
# For instance, convert [['B01FY1LAKI,1', [1, 0, 1, 0, 1, 0, 0, 1, 0, 0], '0'],] to
# input_asin: [['B01FY1LAKI,1'],...]
# input_data: [[1, 0, 1, 0, 1, 0, 0, 1, 0, 0],...]
# input_target: [['0'],...]
def convert_data_matrix_to_input_list(data_matrix, input_asin, input_data,input_target):
    for list in data_matrix:
        input_asin.append(list[0])
        input_data.append(list[1])
        input_target.append(list[2])

# This routine calculates all the words frequency from input file
def construct_word_counter(word_counter, updated_input_file_for_training_model):
    for line in updated_input_file_for_training_model:
        content = line.split('\t')
        description_list = content[1].split()

        # skip the asin that is without any extracted information
        if not content[1] == '':
            for word in description_list:
                if not len(word) == 1:  # remove single number of letter
                    if word in word_counter:
                        word_counter[word] = word_counter[word] + 1
                    else:
                        word_counter[word] = 1

####################### main #########################
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print "[FATAL] " + str(e)
        raise

'''


