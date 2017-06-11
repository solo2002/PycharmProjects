#!/apollo/bin/env python -tt


# This script (runs every day) reads book meta-data (trimDescription-books-1.txt) and processes it as follow:
# If the asin is labeled as adult book, then the script directly writes to 'output_data_after_applying_model.txt'
# If the asin is unlabed as adult book, then the script extracts information based on topWordsDict, and predicts it by
# applying the trained_model, and then outputs to 'output_data_after_applying_model.txt'
# Finally, it converts 'output_data_after_applying_model.txt' to 'AdultBookScore.rodb', and installs to NFS

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
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'quot',
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
    idf_counter = {}
    construct_word_counter(word_counter, idf_counter, updated_input_file_for_training_model)

    # tf * (idf + 1)
    # a = np.array([2,1,2])
    # b = np.array([2,1,2])
    # c = a * b
    # word_dict = dict.fromkeys(top_words_dict.keys(), 0)
    # idf_counter = dict.fromkeys(word_counter.keys(), 1)

    for w in idf_counter:
        idf_counter[w] = round(1.0 / idf_counter[w], 3) + 1

    tf_idf_dict = dict.fromkeys(word_counter.keys(), 0)
    for w in word_counter:
        tf_idf_dict[w] = idf_counter[w] * word_counter[w]
    # sort the dictionary, and return a list of sorted words according to the highest item frequency
    popular_words = sorted(tf_idf_dict, key=word_counter.get, reverse=True)

    # select top most frequent words as features
    top_words_set = popular_words[:2000]
    for word in top_words_set:
        print word, tf_idf_dict[word]

    top_words_dictionary = {w: True for w in top_words_set}
    #idf_counter = dict.fromkeys(word_counter.keys(), 1)

    tf_idf_dict ={}
    tf_idf_dict = {}
    word_counter = {}
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
def construct_word_counter(word_counter, idf_counter, updated_input_file_for_training_model):

    for line in updated_input_file_for_training_model:
        first_count = True
        content = line.split('\t')
        description_list = content[1].split()
        line_set = set()
        # skip the asin that is without any extracted information
        if not content[1] == '':
            for word in description_list:
                if not len(word) == 1 and (not word.isdigit()):  # remove single number of letter
                    line_set.add(word)
                    if word in word_counter:
                        word_counter[word] = word_counter[word] + 1
                    else:
                        word_counter[word] = 1
        for w in line_set:
            idf_counter[word] += 1



####################### main #########################
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print "[FATAL] " + str(e)
        raise



'''
import PyRODB
import os
import stanza
import adult_book_scorer_training_model as training_model
import cPickle
from SimilaritiesPythonUtilities import Common, CommonFileSystem

FALSE_POSITIVE_ASIN = []
TRUE_NEGATIVE = 0  # predict = '0', label (target) = '0',
FALSE_POSITIVE = 0  # predict = '1', label (target) = '0',
STANZA_SECTION = "adult-book-scorer-applying-model"

def main():
    config = Common.get_stanza_section_from_config(STANZA_SECTION)
    try:
        remoteEroticaEbookBrowseNodeFile = config['remote-erotica-ebook-browse-node-file']
        localEroticaEbookBrowseNodeFile = config['local-latest-erotica-ebook-browse-node-file']
        localTrainedModel = config['trained-model-dir']
        localTopWordsDict = config['top-words-dict-dir']
        localItemRodb = config['local-item-rodb']
        localBookDescriptionData = config['local-trim-descriptions-1-books']
        localOutputDataAfterApplyingModel = config['local-output-data-after-applying-model']
        localRODBGenerate = config['local-rodb-generate-dir']
        localAdultBookScoreRodb = config['local-adult-book-score-rodb']
    except Exception as e:
        print "[FATAL] Error retrieving configuration parameters!"
        raise

    # Sync latest erotica ebook browse node data
    remote_latest_browse_node_file = CommonFileSystem.get_latest_remote_filename(remoteEroticaEbookBrowseNodeFile)
    CommonFileSystem.rsync_get(remote_latest_browse_node_file, localEroticaEbookBrowseNodeFile)
    erotica_input = open(localEroticaEbookBrowseNodeFile, 'r')

    # construct a map for erotica ebook browse node asins
    erotica_map = {}
    for line in erotica_input:
        content = line.split('\t')
        erotica_map[content[0]] = True

    db = PyRODB.open(localItemRodb)
    input_file = open(localBookDescriptionData, 'r')
    output_data_file = open(localOutputDataAfterApplyingModel, 'w')

    # Load trained model and topWordsDict
    with open(localTrainedModel, 'rb') as f:
        forest = cPickle.load(f)
    with open(localTopWordsDict, 'rb') as topWordsDict:
        top_words_dict = cPickle.load(topWordsDict)

    apply_model_and_output(forest, db, erotica_map, top_words_dict, input_file, output_data_file)

    output_data_file.close()
    input_file.close()

    print 'Now writing to RODB file'
    os.system('LC_ALL=C sort ' + localOutputDataAfterApplyingModel + ' | ' +
                      localRODBGenerate + ' --output-file=' + localAdultBookScoreRodb)

    print 'Error Rate for unlabeled books (just for reference):', \
        round(float(FALSE_POSITIVE) / (FALSE_POSITIVE + TRUE_NEGATIVE), 5) # For analysis, will remove later
    print 'false_positive asin', FALSE_POSITIVE_ASIN # For analysis, will remove later

# In this routine, trained model predicts test_input and output to output_data_file
def predict_and_output(forest, test_input_asin, test_input_data, test_input_target, output_data_file):
    global TRUE_NEGATIVE
    global FALSE_POSITIVE
    global FALSE_POSITIVE_ASIN

    label_pred = forest.predict(test_input_data[0:])
    for i in range(0, len(test_input_data)):
        if label_pred[i] == '0' and test_input_target[i] == '0':
            TRUE_NEGATIVE += 1
            output_data_file.writelines(test_input_asin[i] + '\t' + '0\n')
        elif label_pred[i] == '1' and test_input_target[i] == '0':
            FALSE_POSITIVE += 1
            FALSE_POSITIVE_ASIN.append(test_input_asin[i]) # For analysis, it could be removed later
            output_data_file.writelines(test_input_asin[i] + '\t' + '1\n')

# This routine works as follow:
# If the asin is labeled as adult book, then the script directly writes to 'output_data_after_applying_model.txt'
# If the asin is unlabed as adult book, then the script extracts information based on topWordsDict, and predicts it by
# applying the trained_model, and then outputs to 'output_data_after_applying_model.txt'
# Due to limit size of memory, this routine processes the entire file into several batches
# For each batch, here it handles 2000000 asins
def apply_model_and_output(forest, db, erotica_map, top_words_dict, input_file, output_data_file):
    need_to_test_asin_counter = 0
    test_input_data = []
    test_input_asin = []
    test_input_target = []
    for line in input_file:
        line_content = line.split('\t')
        current_key = line_content[0]
        empty_line = True
        if training_model.exist_in_RODB(current_key, db):
            if training_model.is_adult_book_from_RODB(current_key, db) or erotica_map.__contains__(current_key):
                # if the asin is labeled as adult book, we just output it directly
                output_data_file.writelines(current_key + '\t' + '1\n')
            else:
                # if the asin is not labeled as adult book, we extract the data and test it with trained model
                need_to_test_asin_counter += 1
                if need_to_test_asin_counter % 2000000 != 0:
                    word_dict = dict.fromkeys(top_words_dict.keys(), 0)
                    descri = line_content[1].strip().split(',')

                    # Skip the asin which does not contain any words in the top_words_dict
                    for word in descri:
                        if top_words_dict.__contains__(word):
                            word_dict[word] = 1
                            empty_line = False
                    if not empty_line:
                        test_input_asin.append(current_key)
                        test_input_data.append(word_dict.values())
                        test_input_target.append('0')  # only non labeled adult books are processed here
                else:
                    # when processed_line_counter % 2000000 == 0, convert to input and applying the model
                    print 'Now process', need_to_test_asin_counter
                    predict_and_output(forest, test_input_asin, test_input_data, test_input_target, output_data_file)
                    test_input_data = []
                    test_input_asin = []
                    test_input_target = []
                    print_result()

    # Take care of whatever left (which at end of the file and the processed_line_counter < 2000000)
    if len(test_input_asin) > 0:
        print 'Now process whatever left', len(test_input_asin), ', which should less than 2000000'
        print 'For the last batch', 'size of test_input_asin:', \
            len(test_input_asin), 'size of test_input_data', len(test_input_data), \
            'size of test_input_target', len(test_input_target)
        predict_and_output(forest, test_input_asin, test_input_data, test_input_target, output_data_file)
        print_result()

    # Print final result
    print_result()

def print_result():
    global TRUE_NEGATIVE
    global FALSE_POSITIVE
    print 'The number of unadult books (according to item.rodb and, ebook erotica browse node), ' \
          'and preditced by the model as unadult book (true_negative)', TRUE_NEGATIVE
    print 'The number of unadult books (according to item.rodb and, ebook erotica browse node), ' \
          'but predicted by the model as adult books (false_positive)', FALSE_POSITIVE

####################### main #########################
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print "[FATAL] " + str(e)
        raise
'''
'''
import PyRODB
import os
import stanza
import adult_book_scorer_training_model as training_model
import cPickle
from SimilaritiesPythonUtilities import Common, CommonFileSystem

FALSE_POSITIVE_ASIN = []
TRUE_NEGATIVE = 0  # predict = '0', label (target) = '0',
FALSE_POSITIVE = 0  # predict = '1', label (target) = '0',
STANZA_SECTION = "adult-book-scorer-applying-model"

def main():
    config = Common.get_stanza_section_from_config(STANZA_SECTION)
    try:
        remoteEroticaEbookBrowseNodeFile = config['remote-erotica-ebook-browse-node-file']
        localEroticaEbookBrowseNodeFile = config['local-latest-erotica-ebook-browse-node-file']
        localTrainedModel = config['trained-model-dir']
        localTopWordsDict = config['top-words-dict-dir']
        localItemRodb = config['local-item-rodb']
        localBookDescriptionData = config['local-trim-descriptions-1-books']
        localOutputDataAfterApplyingModel = config['local-output-data-after-applying-model']
        localRODBGenerate = config['local-rodb-generate-dir']
        localAdultBookScoreRodb = config['local-adult-book-score-rodb']
    except Exception as e:
        print "[FATAL] Error retrieving configuration parameters!"
        raise

    # Sync latest erotica ebook browse node data
    remote_latest_browse_node_file = CommonFileSystem.get_latest_remote_filename(remoteEroticaEbookBrowseNodeFile)
    CommonFileSystem.rsync_get(remote_latest_browse_node_file, localEroticaEbookBrowseNodeFile)
    erotica_input = open(localEroticaEbookBrowseNodeFile, 'r')

    # construct a map for erotica ebook browse node asins
    erotica_map = {}
    for line in erotica_input:
        content = line.split('\t')
        erotica_map[content[0]] = True

    db = PyRODB.open(localItemRodb)
    input_file = open(localBookDescriptionData, 'r')
    output_data_file = open(localOutputDataAfterApplyingModel, 'w')

    # Load trained model and topWordsDict
    with open(localTrainedModel, 'rb') as f:
        forest = cPickle.load(f)
    with open(localTopWordsDict, 'rb') as topWordsDict:
        top_words_dict = cPickle.load(topWordsDict)

    apply_model_and_output(forest, db, erotica_map, top_words_dict, input_file, output_data_file)

    output_data_file.close()
    input_file.close()

    print 'Now writing to RODB file'
    os.system('LC_ALL=C sort ' + localOutputDataAfterApplyingModel + ' | ' +
                      localRODBGenerate + ' --output-file=' + localAdultBookScoreRodb)

    print 'Error Rate for unlabeled books (just for reference):', \
        round(float(FALSE_POSITIVE) / (FALSE_POSITIVE + TRUE_NEGATIVE), 5) # For analysis, will remove later
    print 'false_positive asin', FALSE_POSITIVE_ASIN # For analysis, will remove later

# In this routine, trained model predicts test_input and output to output_data_file
def predict_and_output(forest, test_input_asin, test_input_data, test_input_target, output_data_file):
    global TRUE_NEGATIVE
    global FALSE_POSITIVE
    global FALSE_POSITIVE_ASIN

    label_pred = forest.predict(test_input_data[0:])
    for i in range(0, len(test_input_data)):
        if label_pred[i] == '0' and test_input_target[i] == '0':
            TRUE_NEGATIVE += 1
            output_data_file.writelines(test_input_asin[i] + '\t' + '0\n')
        elif label_pred[i] == '1' and test_input_target[i] == '0':
            FALSE_POSITIVE += 1
            FALSE_POSITIVE_ASIN.append(test_input_asin[i]) # For analysis, it could be removed later
            output_data_file.writelines(test_input_asin[i] + '\t' + '1\n')

# This routine works as follow:
# If the asin is labeled as adult book, then the script directly writes to 'output_data_after_applying_model.txt'
# If the asin is unlabed as adult book, then the script extracts information based on topWordsDict, and predicts it by
# applying the trained_model, and then outputs to 'output_data_after_applying_model.txt'
# Due to limit size of memory, this routine processes the entire file into several batches
# For each batch, here it handles 2000000 asins
def apply_model_and_output(forest, db, erotica_map, top_words_dict, input_file, output_data_file):
    need_to_test_asin_counter = 0
    test_input_data = []
    test_input_asin = []
    test_input_target = []
    for line in input_file:
        line_content = line.split('\t')
        current_key = line_content[0]
        empty_line = True
        if training_model.exist_in_RODB(current_key, db):
            if training_model.is_adult_book_from_RODB(current_key, db) or erotica_map.__contains__(current_key):
                # if the asin is labeled as adult book, we just output it directly
                output_data_file.writelines(current_key + '\t' + '1\n')
            else:
                # if the asin is not labeled as adult book, we extract the data and test it with trained model
                need_to_test_asin_counter += 1
                if need_to_test_asin_counter % 1000000 != 0:
                    word_dict = dict.fromkeys(top_words_dict.keys(), 0)
                    descri = line_content[1].strip().split(',')

                    # Skip the asin which does not contain any words in the top_words_dict
                    for word in descri:
                        if top_words_dict.__contains__(word):
                            word_dict[word] = 1
                            empty_line = False
                    if not empty_line:
                        test_input_asin.append(current_key)
                        test_input_data.append(word_dict.values())
                        test_input_target.append('0')  # only non labeled adult books are processed here
                else:
                    # when processed_line_counter % 2000000 == 0, convert to input and applying the model
                    print 'Now process', need_to_test_asin_counter
                    predict_and_output(forest, test_input_asin, test_input_data, test_input_target, output_data_file)
                    test_input_data = []
                    test_input_asin = []
                    test_input_target = []
                    print_result()

    # Take care of whatever left (which at end of the file and the processed_line_counter < 2000000)
    if len(test_input_asin) > 0:
        print 'Now process whatever left', len(test_input_asin), ', which should less than 1000000'
        print 'For the last batch', 'size of test_input_asin:', \
            len(test_input_asin), 'size of test_input_data', len(test_input_data), \
            'size of test_input_target', len(test_input_target)
        predict_and_output(forest, test_input_asin, test_input_data, test_input_target, output_data_file)
        print_result()

    # Print final result
    print_result()

def print_result():
    global TRUE_NEGATIVE
    global FALSE_POSITIVE
    print 'The number of unadult books (according to item.rodb and, ebook erotica browse node), ' \
          'and preditced by the model as unadult book (true_negative)', TRUE_NEGATIVE
    print 'The number of unadult books (according to item.rodb and, ebook erotica browse node), ' \
          'but predicted by the model as adult books (false_positive)', FALSE_POSITIVE

####################### main #########################
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print "[FATAL] " + str(e)

'''