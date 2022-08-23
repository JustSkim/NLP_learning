from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np


# 构建Doc2vec模型，获得句子向量
def get_sentence_vec(datasets):
    # gemsim里Doc2vec模型需要的输入为固定格式，输入样本为[句子，句子序号]
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(datasets)]
    # 初始化和训练模型
    model = Doc2Vec(documents, vector_size=500, dm=1, window=4, min_count=5, epochs=50)
    # model = Doc2Vec(vector_size=300, dm=1, window=4, min_count=5, epochs=50)
    # model.build_vocab(documents)
    # model.train(documents,total_examples=model.corpus_count,epochs=model.epochs)

    model.save('data/w2v/doc2vec_model.pkl')  # 将模型保存到磁盘
    # 获得数据集的句向量
    documents_vecs = np.concatenate([np.array(model.docvecs[sen.tags[0]].reshape(1, 300)) for sen in documents])
    return documents_vecs


if __name__ == '__main__':
    # 准备数据
    train_data = pd.read_csv('data/clean_data_train.csv', sep=',', names=['contents', 'labels']).astype(str)
    test_data = pd.read_csv('data/clean_data_test.csv', sep=',', names=['contents', 'labels']).astype(str)
    cw = lambda x: str(x).split()
    train_data['words'] = train_data['contents'].apply(cw)
    test_data['words'] = train_data['contents'].apply(cw)
    datasets = pd.concat([train_data, test_data])

    # doc2vec句向量训练和生成
    documents_vec = get_sentence_vec(list(datasets['words']))

    # 加载训练好的模型
    doc2vec_model = Doc2Vec.load('data/w2v/doc2vec_model.pkl')
    # 推断新文档向量
    doc2vec_model.infer_vector(['绝望', '快递', '说', '收到', '快递', '中奖', '开心'])