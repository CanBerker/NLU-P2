from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import string

from strategies import Strategy


def strip_data(s):
    """Strip the story text from special characters, only allowing letters
    and whitespace, and convert the story to lowercase. """
    stripped = filter(lambda x: x in string.ascii_letters + string.whitespace,
                      s)
    return ''.join(stripped).lower()


class CharacterBasedLstmStrategy(Strategy):
    """Simple LSTM model, having individual characters as inputs to the LSTM
    cell instead of words.

    Draws from https://blog.keras.io/a-ten-minute-introduction-to-sequence-to
    -sequence-learning-in-keras.html """
    def __init__(self):
        self.batch_size = 64
        self.epochs = 100
        self.latent_dim = 256
        self.num_samples = 10000

        self.input_texts = []
        self.target_texts = []
        self.input_characters = set(string.ascii_letters + string.whitespace)
        self.target_characters = set(string.ascii_letters + string.whitespace)

        self.encoder_model = None
        self.decoder_model = None
        self.num_encoder_tokens = None
        self.num_decoder_tokens = None
        self.input_token_index = None
        self.target_token_index = None
        self.reverse_input_char_index = None
        self.reverse_target_char_index = None

        self.max_encoder_seq_length = 280  # Assuming max 70 per sentence
        self.max_decoder_seq_length = 100

    def fit(self, data: np.ndarray) -> None:
        # Restrict the number of samples, if necessary.
        if self.num_samples < len(data):
            data = data[:self.num_samples]
        for story in data:
            # Story excluding the fifth sentence
            input_text = strip_data(' '.join(story[2:6]))
            self.input_texts.append(input_text)
            # Use tab character as start and newline as end
            target_text = strip_data('\t' + story[6] + '\n')
            self.target_texts.append(target_text)

        self.input_characters = sorted(list(self.input_characters))
        self.target_characters = sorted(list(self.target_characters))
        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)

        print('Number of samples:', len(self.input_texts))
        print('Number of unique input tokens:', self.num_encoder_tokens)
        print('Number of unique output tokens:', self.num_decoder_tokens)
        print('Max sequence length for inputs:', self.max_encoder_seq_length)
        print('Max sequence length for outputs:', self.max_decoder_seq_length)

        self.input_token_index = dict(
            [(char, i) for i, char in enumerate(self.input_characters)])
        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(self.target_characters)])

        encoder_input_data = np.zeros(
            (len(self.input_texts), self.max_encoder_seq_length,
             self.num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (len(self.input_texts), self.max_decoder_seq_length,
             self.num_decoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (len(self.input_texts), self.max_decoder_seq_length,
             self.num_decoder_tokens),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(
                zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one
                # timestep
                decoder_input_data[i, t, self.target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[
                        i, t - 1, self.target_token_index[char]] = 1.

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        # We set up our decoder to return full output sequences, and to
        # return internal states as well. We don't use the return states in
        # the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True,
                            return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Run training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  validation_split=0.2)
        # Save model
        model.save('lstm.h5')

        # Define sampling models
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())

    def predict(self, data: np.ndarray) -> str:
        input_text = strip_data(' '.join(data[1:5]))
        input_data = np.zeros(
            (1, self.max_encoder_seq_length, self.num_encoder_tokens))
        for t, char in enumerate(input_text):
            input_data[0, t, self.input_token_index[char]] = 1.

        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_data)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start
        # character.
        target_seq[0, 0, self.target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        print(' '.join(data[1:5]) + decoded_sentence)
        return decoded_sentence
