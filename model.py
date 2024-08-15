import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np

train_embed = np.load(
    'image_to_embedding_train.npy', allow_pickle=True)
test_embed = np.load(
    'image_to_embedding_test.npy', allow_pickle=True)
train_captions = np.load(
    'image_to_index_caption_train.npy', allow_pickle=True)
test_captions = np.load(
    'image_to_index_caption_test.npy', allow_pickle=True)
vocab = np.load('vocabulary.npy', allow_pickle=True)


train_embed = dict(train_embed)
test_embed = dict(test_embed)

train_caption = dict(train_captions)
test_caption = dict(test_captions)

vocab = vocab.item()


train_embed = np.array(list(train_embed.values()))
train_caption = np.array(list(train_caption.values()))
test_embed = np.array(list(test_embed.values()))
test_caption = np.array(list(test_caption.values()))


IMG_EMBED_SIZE = train_embed.shape[1]
IMG_EMBED_BOTTLENECK = 128
WORD_EMBED_SIZE = 100
LSTM_UNITS = 256
LOGIT_BOTTLENECK = 120
pad_idx = vocab['#PAD#']
vocab_size = len(vocab)
batch_size = 64
MAX_LEN = train_caption.shape[1]


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, img_embed_size, img_embed_bottleneck_size, lstm_units, logit_bottleneck_size, word_embed_size):
        super(Decoder, self).__init__()

        self.img_embed_to_bottleneck = layers.Dense(
            img_embed_bottleneck_size, activation='elu')
        self.img_embed_bottleneck_to_h0 = layers.Dense(
            lstm_units, activation='elu')
        self.word_embed = layers.Embedding(vocab_size, word_embed_size)
        self.lstm = layers.LSTM(
            lstm_units, return_sequences=True, return_state=True)
        self.token_logits_bottleneck = layers.Dense(
            logit_bottleneck_size, activation='elu')
        self.token_logits = layers.Dense(vocab_size)

    def call(self, inputs):
        img_embeds, sentences = inputs

        img_bottleneck = self.img_embed_to_bottleneck(img_embeds)
        c0 = h0 = self.img_embed_bottleneck_to_h0(img_bottleneck)

        word_embeds = self.word_embed(sentences[:, :-1])
        hidden_states, _, _ = self.lstm(word_embeds, initial_state=[h0, c0])

        flat_hidden_states = tf.reshape(
            hidden_states, [-1, hidden_states.shape[-1]])
        flat_token_logits = self.token_logits(
            self.token_logits_bottleneck(flat_hidden_states))
        token_logits = tf.reshape(flat_token_logits, [tf.shape(
            hidden_states)[0], -1, self.token_logits.units])

        return token_logits


def masked_loss_function(y_true, y_pred, pad_idx):
    y_true = y_true[:, 1:]
    mask = tf.cast(tf.not_equal(y_true, pad_idx), dtype=tf.float32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


decoder = Decoder(vocab_size=vocab_size,
                  img_embed_size=IMG_EMBED_SIZE,
                  img_embed_bottleneck_size=IMG_EMBED_BOTTLENECK,
                  lstm_units=LSTM_UNITS,
                  logit_bottleneck_size=LOGIT_BOTTLENECK,
                  word_embed_size=WORD_EMBED_SIZE
                  )


def tf_data_generator(img_embeds, captions_indexed, batch_size):
    while True:
        indices = np.random.permutation(len(img_embeds))
        for start in range(0, len(img_embeds), batch_size):
            end = min(start + batch_size, len(img_embeds))
            batch_indices = indices[start:end]  # .tolist()
            batch_img_embeds = img_embeds[batch_indices]
            batch_captions = captions_indexed[batch_indices]

            yield (batch_img_embeds, batch_captions), batch_captions


train_dataset = tf.data.Dataset.from_generator(
    lambda: tf_data_generator(train_embed, train_caption, batch_size),
    output_signature=(
        (tf.TensorSpec(shape=(None, IMG_EMBED_SIZE), dtype=tf.float32),  # img_embeds
         tf.TensorSpec(shape=(None, MAX_LEN), dtype=tf.int32)),          # sentences
        # sentences (target)
        tf.TensorSpec(shape=(None, MAX_LEN), dtype=tf.int32)
    )
).prefetch(tf.data.AUTOTUNE)


val_dataset = tf.data.Dataset.from_generator(
    lambda: tf_data_generator(test_embed, test_caption, batch_size),
    output_signature=(
        (tf.TensorSpec(shape=(None, IMG_EMBED_SIZE), dtype=tf.float32),  # img_embeds
         tf.TensorSpec(shape=(None, MAX_LEN), dtype=tf.int32)),           # sentences
        # sentences (target)
        tf.TensorSpec(shape=(None, MAX_LEN), dtype=tf.int32)
    )
).prefetch(tf.data.AUTOTUNE)


steps_per_epoch = len(train_embed) // batch_size
validation_steps = len(test_embed) // batch_size

steps_per_epoch = len(train_embed) // batch_size
validation_steps = len(test_embed) // batch_size

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "weights_{epoch:02d}.keras", save_best_only=True, monitor="val_loss", mode="min"),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True)
]

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, name='Adam')

decoder.compile(optimizer=optimizer, loss=lambda y_true,
                y_pred: masked_loss_function(y_true, y_pred, pad_idx))

history = decoder.fit(
    train_dataset,
    validation_data=val_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

decoder.save('/content/sample_data/data/decoder_model.keras')

# INFERENCE CODE: Generates captions for single picture at a time. It will generate #END# token is encoutered or max_length is reached.
# NOTE: max_length here can be different.


def generate_caption(decoder, img_embed, max_length=20):

    input_seq = np.array([[vocab['#START#']]])

    img_bottleneck = decoder.img_embed_to_bottleneck(img_embed)
    c0 = h0 = decoder.img_embed_bottleneck_to_h0(img_bottleneck)

    generated_caption = []

    for i in range(max_length):

        word_embeds = decoder.word_embed(input_seq)

        hidden_states, h0, c0 = decoder.lstm(
            word_embeds, initial_state=[h0, c0])

        flat_hidden_states = tf.reshape(
            hidden_states, [-1, hidden_states.shape[-1]])
        token_logits = decoder.token_logits(
            decoder.token_logits_bottleneck(flat_hidden_states))

        predicted_id = tf.argmax(token_logits, axis=-1).numpy()[0]
        predicted_value = token_logits.numpy()[0][predicted_id]
        print(predicted_value)

        if predicted_id == vocab['#END#']:
            break

        generated_caption.append(
            list(vocab.keys())[list(vocab.values()).index(predicted_id)])
        input_seq = np.array([[predicted_id]])
        print(predicted_id)

    return ' '.join(generated_caption)


img_embed = np.reshape(test_embed[0], (1, 512))
max_length = 20
caption = generate_caption(decoder, img_embed, max_length)
print(f"Generated caption: {caption}")
