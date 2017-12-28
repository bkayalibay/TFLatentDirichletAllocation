import numpy as np
try:
    from observations import nips
except ImportError:
    raise ImportError('Please install the python package Observations: pip install Observations')


def subset(count_matrix, document_ids, vocabulary, year):
    doc_indices = [i for i, doc in enumerate(document_ids)
                   if doc.startswith('2011') and count_matrix[:, i].sum() > 0]
    document_ids = [document_ids[i] for i in doc_indices]
    count_matrix = count_matrix[:, doc_indices]
    take_word = np.logical_and(
        np.sum(count_matrix != 0, axis=1) >= 2,
        np.sum(count_matrix, 1) >= 10)
    words = [word for word, tw in zip(vocabulary, take_word) if tw]
    count_matrix = count_matrix[take_word, :].T
    return count_matrix, document_ids, words


def expand_docs(count_matrix):
    docs = []
    for row in count_matrix:
        doc = []
        for i, entry in enumerate(row):
            doc += [i] * entry
        docs.append(np.array(doc, dtype='int32'))
    return docs


def load_data(data_dir, year='2011', expand=False):
    x_train, metadata = nips(data_dir)
    documents = metadata['columns']
    words = metadata['rows']

    count_matrix, document_ids, vocabulary = subset(
        x_train, documents, words, year=year)
    if expand:
        documents = expand_docs(count_matrix)
    else:
        documents = count_matrix.astype('int32')

    return documents, vocabulary
