from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
from textblob import TextBlob
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, SpatialDropout1D, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__, template_folder='template')

UPLOAD_FOLDER = r'C:\Users\mishr\Desktop\PREDICTING STOCK MARKET MOVEMENT BY ANALYZING SENTIMENT IN NEWS HEADLINES\Project'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_sentiment(description):
    blob = TextBlob(description)
    sentiment_score = blob.sentiment.polarity
    return 1 if sentiment_score >= 0 else 0


def add_sentiment_label(df):
    df['Label'] = df['Description'].apply(analyze_sentiment)
    return df[['Date', 'Label', 'Description']]


def preprocess_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return ' '.join(stemmed_tokens)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'csvFile' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['csvFile']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Read the CSV file
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

        # Add sentiment label to the dataframe
        df_with_label = add_sentiment_label(df)

        # Preprocess text
        df_with_label['Processed'] = df_with_label['Description'].apply(preprocess_text)

        # Tokenize and pad sequences
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df_with_label['Processed'])
        X_sequences = tokenizer.texts_to_sequences(df_with_label['Processed'])
        X_padded = pad_sequences(X_sequences)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_padded, df_with_label['Label'], test_size=0.2, random_state=42)

        # Build and compile the LSTM model
        model = Sequential()
        model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=X_padded.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(32, activation='relu', kernel_regularizer='l2'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Fit the model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=12, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

        # Predict
        y_pred_probs = model.predict(X_test)
        y_pred = (y_pred_probs > 0.5).astype(int)

        # Calculate accuracy and confusion matrix
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Extract confusion matrix values
        tn, fp, fn, tp = conf_matrix.ravel()

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        conf_matrix_path = os.path.join(app.config['UPLOAD_FOLDER'], 'confusion_matrix.png')
        plt.savefig(conf_matrix_path)
        plt.close()

        return jsonify({
            'accuracy': accuracy,
            'confusion_matrix': 'confusion_matrix.png',
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
