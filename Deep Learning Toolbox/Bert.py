# Load pre-trained model (weights)
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

def lstm_model():
    """
    Define the model
    """
    model = Sequential()

    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1,768], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.summary()

    return model

total_loss = []
results = []
prediction_list = []

cv = KFold(n_splits=5, shuffle=True)
cv_data = cv.split(X)

fold_cnt = 1
cuda = torch.device('cuda')

with torch.cuda.device(cuda):
    for traincv, testcv in cv_data:
        torch.cuda.empty_cache()
        print("Fold {}".format(fold_cnt))

        # get the train and test from the dataset
        X_train, X_test, y_train, y_test = X.iloc[traincv], X.iloc[testcv], y.iloc[traincv], y.iloc[testcv]
        train_essays = X_train['essay']
        test_essays = X_test['essay']

        tokenized_train = train_essays.apply((lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=200)))
        tokenized_test = test_essays.apply((lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=200)))

        # train dataset
        max_len = 0
        for i in tokenized_train.values:
            if len(i) > max_len:
                max_len = len(i)
        padded_train = np.array([i + [0] * (max_len - len(i)) for i in tokenized_train.values])

        attention_mask_train = np.where(padded_train != 0, 1, 0)
        train_input_ids = torch.tensor(padded_train)
        train_attention_mask = torch.tensor(attention_mask_train)

        with torch.no_grad():
            outputs_train = model(train_input_ids, attention_mask=train_attention_mask)
            last_hidden_states_train = outputs_train[0]  # get the last hidden state
        train_features = last_hidden_states_train[:, 0, :].numpy()

        ## test dataset
        max_len = 0
        for i in tokenized_test.values:
            if len(i) > max_len:
                max_len = len(i)
        padded_test = np.array([i + [0] * (max_len - len(i)) for i in tokenized_test.values])

        attention_mask_test = np.where(padded_test != 0, 1, 0)
        test_input_ids = torch.tensor(padded_test)
        test_attention_mask = torch.tensor(attention_mask_test)

        with torch.no_grad():
            outputs_test = model(test_input_ids, attention_mask=test_attention_mask)
            last_hidden_states_test = outputs_test[0]  # get the last hidden state
        test_features = last_hidden_states_test[:, 0, :].numpy()
        train_x, train_y = train_features.shape
        test_x, test_y = test_features.shape

        x_train_reshaped = np.reshape(train_features, (train_x, 1, train_y))
        x_test_reshaped = np.reshape(test_features, (test_x, 1, test_y))

        lstm = lstm_model()
        lstm.fit(x_train_reshaped, y_train, batch_size=128, epochs=70)

        y_pred = lstm.predict(x_test_reshaped)

        # evaluate the model
        result = mean_squared_error(y_test.values, y_pred)
        print("MSE: {}".format(result))
        results.append(result)
        fold_cnt += 1

        tf.keras.backend.clear_session()