library(keras)
library(readr)
library(dplyr)
library(readODS)

#All major NLI corpora in a unified form of RData for a quick manipulation, ask for availability (info@martinvita.eu)
load("korpusy-SNLI-MultiNLI-SciTail-SICK.RData")

# glove embeddings, freely available
glove300en <- read_table2("glove.6B.300d.txt", progress = T, col_names = F)

# parameters
FLAGS <- flags(
  flag_integer("vocab_size", 80000),
  flag_integer("max_len_padding", 44)
)

# CST data, ask for availability (info@martinvita.eu)
enpl <- read_ods("CST2EN.ods", sheet = 1, col_names = F, strings_as_factors = F, verbose = T)

# list of all texts to be used
texty <- c(enpl$F, enpl$G, test.multinli$Premise, test.multinli$Hypothesis, dev.multinli$Premise, dev.multinli$Hypothesis, train.multinli$Premise, train.multinli$Hypothesis)

# preparing a tokenization
tokenizer <- text_tokenizer(num_words = FLAGS$vocab_size)
fit_text_tokenizer(tokenizer, texty)

#### DATA PREP ####
# using MultiNLI data - selecting only data with labels (entailment, neutral, contradiction)
# data with label "-" are ignored

relevantni.test <- test.multinli$GoldLabel!="-"
relevantni.dev <- dev.multinli$GoldLabel!="-"
relevantni.train <- train.multinli$GoldLabel!="-"

# labels preparation, i. e., from categorical labels to vector ones
make.labels <- function(clss) {
  len <- length(clss)
  res <- matrix(0, nrow = len, ncol = 3)
  res[clss=="entailment", 1] <- 1
  res[clss=="neutral", 2] <- 1
  res[clss=="contradiction", 3] <- 1
  return(res)
}

train.labels <- make.labels(train.multinli$GoldLabel)
train.labels <- as.matrix(train.labels[relevantni.train,])
dev.labels <- make.labels(dev.multinli$GoldLabel)
dev.labels <- as.matrix(dev.labels[relevantni.dev,])
test.labels <- make.labels(test.multinli$GoldLabel)
test.labels <- as.matrix(test.labels[relevantni.test,])

# preparing lookup word list, word index, obtaining vocabulary size
word.index <- tokenizer$word_index
voc.size <- min(length(word.index), FLAGS$vocab_size) 
words <- names(word.index)[1:voc.size]

# preparing a lookup table, using dplyr left_join function provide extremely fast results comparing to traditional examples of Keras.Rstudio
aux.index.table <- data.frame(Indx=1:voc.size, Wrd=words, stringsAsFactors = F)
colnames(glove300en) <- c("Wrd", paste0("S", 1:300))
aux.join <- left_join(aux.index.table, glove300en, by = "Wrd")

emb.mat <- as.matrix(aux.join[,3:ncol(aux.join)])
# words that are not recognized among precomputed GloVe are set to 0
emb.mat[is.na(emb.mat)] <- 0

# a dirty trick for R, since the indexes are shift +1 comparing to Python...
emb.mat2 <- rbind(rep(0, times = 300), emb.mat)

# tokenization of sentences and padding sequences with zeros
sent1.train <- texts_to_sequences(tokenizer, train.multinli$Premise)
sent2.train <- texts_to_sequences(tokenizer, train.multinli$Hypothesis)

sent1.dev <- texts_to_sequences(tokenizer, dev.multinli$Premise)
sent2.dev <- texts_to_sequences(tokenizer, dev.multinli$Hypothesis)

sent1.test <- texts_to_sequences(tokenizer, test.multinli$Premise)
sent2.test <- texts_to_sequences(tokenizer, test.multinli$Hypothesis)


tr.p <- pad_sequences(sent1.train, maxlen = FLAGS$max_len_padding, value = 0)
tr.h <- pad_sequences(sent2.train, maxlen = FLAGS$max_len_padding, value = 0)
tr.p <- tr.p[relevantni.train,]
tr.h <- tr.h[relevantni.train,]

de.p <- pad_sequences(sent1.dev, maxlen = FLAGS$max_len_padding, value = 0)
de.h <- pad_sequences(sent2.dev, maxlen = FLAGS$max_len_padding, value = 0)
de.p <- de.p[relevantni.dev,]
de.h <- de.h[relevantni.dev,]

te.p <- pad_sequences(sent1.test, maxlen = FLAGS$max_len_padding, value = 0)
te.h <- pad_sequences(sent2.test, maxlen = FLAGS$max_len_padding, value = 0)
te.p <- te.p[relevantni.test,]
te.h <- te.h[relevantni.test,]


#### INFERSENT MODEL ####

# embedding part
embeddovaci <- keras_model_sequential()


# simple GRU encoder
embeddovaci %>% 
 layer_embedding(
    input_dim = voc.size + 1,
    output_dim = 300,
    input_length = FLAGS$max_len_padding,
    weights = list(emb.mat2),
    trainable = F,
    mask_zero = T, name = "EmbEnc") %>%
  layer_gru(units = 256, activation = 'relu', name = "RecEnc")

# InferSent BiLSTM-max pooling
# embeddovaci %>% 
#  layer_embedding(
#    input_dim = voc.size + 1,
#    output_dim = 300,
#    input_length = FLAGS$max_len_padding,
#    weights = list(emb.mat2),
#    trainable = F,
#    mask_zero = T) %>%
#  bidirectional(layer_lstm(units = 512, activation = 'relu', return_sequences = T), merge_mode = 'concat', name = 'Bidi') %>%
#  layer_lambda(f = function(x) {k_max(x, axis = 2)}, name = 'Lam')
# SEE AXIS = 2 - for R implementation! In Python+Keras take care of axes!!


# the model with embeddings from the previous part

input1 <- layer_input(shape = c(FLAGS$max_len_padding), dtype = "int32")
input2 <- layer_input(shape = c(FLAGS$max_len_padding), dtype = "int32")


embedding1 <- input1 %>% embeddovaci
embedding2 <- input2 %>% embeddovaci


lb <- layer_concatenate(list(embedding1,
                             embedding2,
                             k_abs(layer_subtract(list(embedding1, embedding2))),
                             layer_multiply(list(embedding1, embedding2)))) %>% layer_dense(units = 512) %>% 
  layer_dense(3, activation = "softmax")

model <- keras_model(inputs = list(input1, input2), outputs = lb)

model %>% compile(
  optimizer = "adam",
  metrics = c("accuracy"),
  loss = "categorical_crossentropy"
)


summary(model)

# using sampling generator when normal fit function is not applicable
sampling_generator <- function(dt1, dt2, bs, lbls) {
  kolo <- 0
  function() {
    rows <- 1:bs+kolo*bs
    dolni <- min(rows)
    horni <- max(rows)
    mm <- nrow(lbls)
    if (horni <= mm) {
      vysledek <- list(list(dt1[rows,], dt2[rows,]), lbls[rows,])
      kolo <<- kolo + 1 
    }
    if (horni > mm && dolni > mm) {
      rows <- 1:bs
      vysledek <- list(list(dt1[rows,], dt2[rows,]), lbls[rows,])
      kolo <<- 1
    }
    if (horni > mm && dolni <= mm) {
      vrchni <- length(dolni:mm)
      spodni <- bs-vrchni
      rows <- c(dolni:mm, 1:spodni)
      vysledek <- list(list(dt1[rows,], dt2[rows,]), lbls[rows,])
      kolo <<- 0
    }
    return(vysledek)
  }
}

early_stopping <- callback_early_stopping(monitor = 'val_loss', patience = 3)
# original InferSent uses more sophisticated training, this for simplicity

model %>% fit_generator(sampling_generator(tr.p, tr.h, 64, train.labels), steps_per_epoch = nrow(tr.p) / 64, epochs = 5, validation_data = list(
  list(
    de.p,
    de.h
  ),
  dev.labels),
  callbacks = c(early_stopping))

evaluate(model, list(te.p, te.h), test.labels)

#### CST transfer ####

# preparing partition indexes
partitions <- ((1:1918)+191) %/% 192

# freezing pretrained weighths of the encoder trained over NLI
freeze_weights(embeddovaci)

#### TRANSFER ####

# tokenization and padding to length of 44

cent1 <- texts_to_sequences(tokenizer, enpl$F)
cent2 <- texts_to_sequences(tokenizer, enpl$G)

c.p <- pad_sequences(cent1, maxlen = FLAGS$max_len_padding, value = 0)
c.h <- pad_sequences(cent2, maxlen = FLAGS$max_len_padding, value = 0)

# classes labels, their number of classes
uq <- unique(enpl$E)
kolik <- length(uq)

# from categorical to vector labels
m.l <- function(clss) {
  len <- length(clss)
  res <- matrix(0, nrow = len, ncol = kolik)
  for (i in 1:len) {
    res[i,] <- as.numeric(uq == clss[i])
  }
  return(res)
}

# making labels
cst.labels <- m.l(enpl$E)
colnames(cst.labels) <- uq

# CST architecture for 10-fold cross validation

# collecting (10) accurracies
acs <- c()

for (k in 1:10) {

i1 <- layer_input(shape = c(FLAGS$max_len_padding), dtype = "int32")
i2 <- layer_input(shape = c(FLAGS$max_len_padding), dtype = "int32")

# obtaining sentence embeddings via precomputed encoders
e1 <- i1 %>% embeddovaci
e2 <- i2 %>% embeddovaci

# simple architecture dealing with (e1, e2, |e1 - e2|, e1*e2) representation of the problem:
lc <- layer_concatenate(list(e1,
                             e2,
                             k_abs(layer_subtract(list(e1, e2))),
                             layer_multiply(list(e1, e2)))) %>% layer_dense(units = 128) %>% 
  layer_dense(12, activation = "softmax")

e.model <- keras_model(inputs = list(i1, i2), outputs = lc)

e.model %>% compile(
  optimizer = "adam",
  metrics = c("accuracy"),
  loss = "categorical_crossentropy"
)


e.model %>% fit(list(c.p[partitions!=k,], c.h[partitions!=k,]), cst.labels[partitions!=k,], epochs = 10, batch_size = 64)

vlc <- evaluate(e.model, list(c.p[partitions==k,], c.h[partitions==k,]),cst.labels[partitions==k,])
acs <- c(acs, vlc$accuracy)
k_reset_uids()

}

acs
