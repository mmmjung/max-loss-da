
import numpy, cupy, re

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter, Variable

embed_init = chainer.initializers.Uniform(.25)

import ast, random, re, copy

def sequence_embed(embed, xs, dropout=0.):
    """Efficient embedding function for variable-length sequences

    This output is equally to
    "return [F.dropout(embed(x), ratio=dropout) for x in xs]".
    However, calling the functions is one-shot and faster.

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): i-th element in the list is an input variable,
            which is a :math:`(L_i, )`-shaped int array.
        dropout (float): Dropout ratio.

    Returns:
        list of ~chainer.Variable: Output variables. i-th element in the
        list is an output variable, which is a :math:`(L_i, N)`-shaped
        float array. :math:`(N)` is the number of dimensions of word embedding.

    """
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])

    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


def block_embed(embed, x, dropout=0.):
    """Embedding function followed by convolution

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which
            is a :math:`(B, L)`-shaped int array. Its first dimension
            :math:`(B)` is assumed to be the *minibatch dimension*.
            The second dimension :math:`(L)` is the length of padded
            sentences.
        dropout (float): Dropout ratio.

    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(B, N, L, 1)`. :math:`(N)` is the number of dimensions
        of word embedding.

    """
    e = embed(x)
    e = F.dropout(e, ratio=dropout)
    e = F.transpose(e, (0, 2, 1))
    e = e[:, :, :, None]
    return e


class TextClassifier(chainer.Chain):

    """A classifier using a given encoder.

     This chain encodes a sentence and classifies it into classes.

     Args:
         encoder (Link): A callable encoder, which extracts a feature.
             Input is a list of variables whose shapes are
             "(sentence_length, )".
             Output is a variable whose shape is "(batchsize, n_units)".
         n_class (int): The number of classes to be predicted.

     """

    def __init__(self, encoder, n_class, synonyms, vocab_inverse, vocab, dropout=0.0, synonym_mode=None):
        super(TextClassifier, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.output = L.Linear(encoder.out_units, n_class)
        self.dropout = dropout
        self.use_synonym_prediction_per_batch = False
        self.use_synonym_prediction_per_sentence = False
        self.counter = 0
        self.synonyms = synonyms
        self.vocab_inverse = vocab_inverse
        self.vocab = vocab
        self.synonym_mode = synonym_mode
        
    def print_sentence(self, word_ids):
        sentence_string = ''
        for w_id in word_ids:
            sentence_string += self.vocab_inverse[int(w_id)] + ' '
        print(sentence_string)

    def sentence_to_ids(self, sentence):
        words_ids = [0]
        for word in sentence.split():
            if word == '[UNK]':
                word = '<unk>'
            words_ids.append(self.vocab[word])
        words_ids.append(0)
        words_ids = cupy.array(words_ids)
        return words_ids
    
    def choose_max_sentence(self, sentence_label):

        original_sentence, label = sentence_label
        
        sentence_string = ''
        
        for w_id in original_sentence:
            word = self.vocab_inverse[int(w_id)]
            if word != '<eos>':
                if word == '<unk>':
                    return original_sentence
                sentence_string += word

        sentence_string = re.sub(r'\W+', '', sentence_string)

        transformed_sentences = self.synonyms[sentence_string]['a']

        if self.synonym_mode == 'rand':
            transformed = random.choice(transformed_sentences)
            return self.sentence_to_ids(transformed)
        
        losses_sentences = []

        for sentence in transformed_sentences:
            word_ids = self.sentence_to_ids(sentence)
            loss, _, _ = self.get_loss([word_ids], [label])
            losses_sentences.append((loss, word_ids))
        
        if self.synonym_mode == 'max':
            chosen_loss_sentence = max(losses_sentences, key = lambda i : i[0].data,
                                        default = (Variable(numpy.array(0.0, dtype=numpy.float32)),original_sentence))
        elif self.synonym_mode == 'min':
            chosen_loss_sentence = min(losses_sentences, key = lambda i : i[0].data,
                                        default = (Variable(numpy.array(0.0, dtype=numpy.float32)),original_sentence))
        else:
            raise NotImplementedError
        
        l,s = chosen_loss_sentence

        return s
        
    def __call__(self, xs, ys=None, evaluate=False):
        if ys is None:
            xs, ys = xs
        
        if self.use_synonym_prediction_per_batch and chainer.config.train:
            
            random_number = random.uniform(0.0, 1.0)
            
            if random_number <= self.synonym_probability:
                max_sentences = list(map(self.choose_max_sentence, zip(xs, ys)))
                xs = max_sentences

        elif self.use_synonym_prediction_per_sentence and chainer.config.train:
            new_xs = []
            for x,y in zip(xs, ys):
                random_number = random.uniform(0.0, 1.0)
                if random_number <= self.synonym_probability:
                    max_sentence = self.choose_max_sentence((x,y))
                    new_xs.append(max_sentence)
                else:
                    new_xs.append(x)

            xs = new_xs
 
        loss, concat_outputs, concat_truths = self.get_loss(xs, ys)

        if evaluate:
            preds = [x.data.base for x in concat_outputs]
            preds = preds[0]
            
            predicted_labels = []
            
            for p in preds:
                p = list(p)
                pp_conv = []
                for pp in p:
                    pp = float(pp)
                    pp_conv.append(pp)
                
                predicted_labels.append(pp_conv.index(max(pp_conv)))
            
            open('compare-with-TEST/predictions.txt', 'w').write(str(predicted_labels))
        
        accuracy = F.accuracy(concat_outputs, concat_truths)
        
        reporter.report({'loss': loss.data}, self)
        reporter.report({'accuracy': accuracy.data}, self)
        
        return loss
        
    def get_loss(self, xs, ys):
        
        concat_outputs = self.predict(xs, ys=ys)
        concat_truths = F.concat(ys, axis=0)

        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        
        return loss, concat_outputs, concat_truths

    def predict(self, xs, ys=None, softmax=False, argmax=False):

        concat_encodings = F.dropout(self.encoder(xs, labels=ys), ratio=self.dropout)
        concat_outputs = self.output(concat_encodings)
        
        if softmax:
            return F.softmax(concat_outputs).data
        elif argmax:
            return self.xp.argmax(concat_outputs.data, axis=1)
        else:
            return concat_outputs


class RNNEncoder(chainer.Chain):

    """A LSTM-RNN Encoder with Word Embedding.

    This model encodes a sentence sequentially using LSTM.

    Args:
        n_layers (int): The number of LSTM layers.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of a LSTM layer and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout):
        super(RNNEncoder, self).__init__(
            embed=L.EmbedID(n_vocab, n_units,
                            initialW=embed_init),
            encoder=L.NStepLSTM(n_layers, n_units, n_units, dropout),
        )
        self.n_layers = n_layers
        self.out_units = n_units
        self.dropout = dropout
        self.use_predict_embed = False

    def __call__(self, xs, labels=None):

        exs = sequence_embed(self.embed, xs, self.dropout)
    
        if self.use_predict_embed and chainer.config.train:
            exs = self.embed.embed_xs_with_prediction(xs, labels=labels, batch='list')
            
        last_h, last_c, ys = self.encoder(None, None, exs)
        assert(last_h.shape == (self.n_layers, len(xs), self.out_units))
        concat_outputs = last_h[-1]
        
        return concat_outputs


class CNNEncoder(chainer.Chain):

    """A CNN encoder with word embedding.

    This model encodes a sentence as a set of n-gram chunks
    using convolutional filters.
    Following the convolution, max-pooling is applied over time.
    Finally, the output is fed into a multilayer perceptron.

    Args:
        n_layers (int): The number of layers of MLP.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of MLP and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout):
        out_units = n_units // 3
        super(CNNEncoder, self).__init__(
            embed=L.EmbedID(n_vocab, n_units, ignore_label=-1,
                            initialW=embed_init),
            cnn_w3=L.Convolution2D(
                n_units, out_units, ksize=(3, 1), stride=1, pad=(2, 0),
                nobias=True),
            cnn_w4=L.Convolution2D(
                n_units, out_units, ksize=(4, 1), stride=1, pad=(3, 0),
                nobias=True),
            cnn_w5=L.Convolution2D(
                n_units, out_units, ksize=(5, 1), stride=1, pad=(4, 0),
                nobias=True),
            mlp=MLP(n_layers, out_units * 3, dropout)
        )
        self.out_units = out_units * 3
        self.dropout = dropout
        self.use_predict_embed = False

    def __call__(self, xs, labels=None):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block, self.dropout)
        if self.use_predict_embed and chainer.config.train:
            ex_block = self.embed.embed_xs_with_prediction(
                xs, labels=labels, batch='concat')
        h_w3 = F.max(self.cnn_w3(ex_block), axis=2)
        h_w4 = F.max(self.cnn_w4(ex_block), axis=2)
        h_w5 = F.max(self.cnn_w5(ex_block), axis=2)
        h = F.concat([h_w3, h_w4, h_w5], axis=1)
        h = F.relu(h)
        h = F.dropout(h, ratio=self.dropout)
        h = self.mlp(h)
        
        return h


class MLP(chainer.ChainList):

    """A multilayer perceptron.

    Args:
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units in a hidden or output layer.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_units, dropout=0.1):
        super(MLP, self).__init__()
        for i in range(n_layers):
            self.add_link(L.Linear(None, n_units))
        self.dropout = dropout
        self.out_units = n_units

    def __call__(self, x):
        for i, link in enumerate(self.children()):
            x = F.dropout(x, ratio=self.dropout)
            x = F.relu(link(x))
        return x


class BOWEncoder(chainer.Chain):

    """A BoW encoder with word embedding.

    This model encodes a sentence as just a set of words by averaging.

    Args:
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_vocab, n_units, dropout=0.1):
        super(BOWEncoder, self).__init__(
            embed=L.EmbedID(n_vocab, n_units, ignore_label=-1,
                            initialW=embed_init),
        )
        self.out_units = n_units
        self.dropout = dropout

    def __call__(self, xs):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block)
        x_len = self.xp.array([len(x) for x in xs], 'i')[:, None, None]
        h = F.sum(ex_block, axis=2) / x_len
        return h


class BOWMLPEncoder(chainer.Chain):

    """A BOW encoder with word embedding and MLP.

    This model encodes a sentence as just a set of words by averaging.
    Additionally, its output is fed into a multilayer perceptron.

    Args:
        n_layers (int): The number of layers of MLP.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of MLP and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        super(BOWMLPEncoder, self).__init__(
            bow_encoder=BOWEncoder(n_vocab, n_units, dropout),
            mlp_encoder=MLP(n_layers, n_units, dropout)
        )
        self.out_units = n_units

    def __call__(self, xs):
        h = self.bow_encoder(xs)
        h = self.mlp_encoder(h)
        return h
