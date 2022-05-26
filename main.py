import gensim
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
   data = []

   for file in Path('data/prepared_data').glob('**/*.json'):
      with open(file, errors='ignore', encoding='utf-8') as fp:
         data.append(fp.read())
   return data


def tagged_document(list_of_list_of_words):
   for i, list_of_words in enumerate(list_of_list_of_words):
      yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


def train_model():

   print("training model...")
   data_for_training = list(tagged_document(load_data()))

   model = gensim.models.doc2vec.Doc2Vec(vector_size=30,min_count=2, epochs=20)
   model.build_vocab(data_for_training)

   model.train(data_for_training, total_examples=model.corpus_count, epochs=model.epochs)

   return model


def file(f):
   with open(f, errors='ignore', encoding='utf-8') as fp:
      data = json.load(fp)
   return data['text']


def similarity(model, file1, file2):
   print("checking similarity...")
   x = model.infer_vector(file(file1))
   y = model.infer_vector(file(file2))
   return cosine_similarity(x.reshape(1,-1), y.reshape(1,-1))


if __name__ == '__main__':
     Model = gensim.models.doc2vec.Doc2Vec.load("10-2-20")
     #Model = train_model()
     #Model.save("30-2-20")

     x = 'data\\prepared_data\\003565_Алгебра_21_5080_1-2с_content_new.json'
     y = 'data\\prepared_data\\003581_Геометрия_21_5080_2с_content_new.json'
     z = 'data\\prepared_data\\003583_Алгоритмы и структуры данных_21_5080_2с_content_new.json'
     w = 'data\\prepared_data\\003587_Алгоритмы и анализ сложности_21_5080_3с_content_new.json'
     print(similarity(Model, x, y))
     print(similarity(Model, x, z))
     print(similarity(Model, w, z))
     print(similarity(Model, w, w))




