# !/bin/bash
# 基于源码的GloVe词向量生成，需要在Linux环境下运行，原文链接：https://asialee.blog.csdn.net/article/details/100124565
# 训练文档为counts.txt。需一并对应使用
# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

#请把make这边注释掉，这个是让你去下个demo，我们直接改成自己的数据
#make
#if [ ! -e text8 ]; then
#  if hash wget 2>/dev/null; then
#    wget http://mattmahoney.net/dc/text8.zip
#  else
#    curl -O http://mattmahoney.net/dc/text8.zip
#  fi
#  unzip text8.zip
#  rm text8.zip
#fi

CORPUS=counts.txt    #CORPUS需要对应自己的欲训练的文档
VOCAB_FILE=vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
BUILDDIR=build
SAVE_FILE=vectors
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5   #单词至少出现的次数
VECTOR_SIZE=300     #训练的词向量维度
MAX_ITER=15         #训练迭代次数
WINDOW_SIZE=15      #窗口大小
BINARY=2
NUM_THREADS=8
X_MAX=10

$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
if [[ $? -eq 0 ]]
  then
  $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
  if [[ $? -eq 0 ]]
  then
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
    if [[ $? -eq 0 ]]
    then
       $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
       if [[ $? -eq 0 ]]
       then
           if [ "$1" = 'matlab' ]; then
               matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2
           elif [ "$1" = 'octave' ]; then
               octave < ./eval/octave/read_and_evaluate_octave.m 1>&2
           else
               python eval/python/evaluate.py
           fi
       fi
    fi
  fi
fi