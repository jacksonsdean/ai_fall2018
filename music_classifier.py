import numpy as np
import tensorflow as tf
import librosa as lb
import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox


N_SAMPLES = 1000
TRAIN_SPLIT = 0.8
LEARN_RATE = 0.003
N_CATEGORIES = 10

train_size = int(N_SAMPLES * TRAIN_SPLIT)
test_size = N_SAMPLES - train_size


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()

        self.lineCount = 0
        self.lineMax = 3

        self. input_shape = [96, 1366]
        self.model_type = tk.IntVar()
        self.model_built = False

        self.batch_size = 0
        self.n_epoch = 0

        self.create_widgets()

    def create_widgets(self):
        self.winfo_toplevel().title("COMP 484 Music Classifier")
        self.typeFrame = tk.LabelFrame(self, text="Model Type", padx=5, pady=5)
        self.typeFrame.grid(row=1, column=0,rowspan=2)
        self.typeFrame.config(bd=1)
        rb1 = tk.Radiobutton(self.typeFrame, text="CNN", variable=self.model_type, value=1, command=lambda: self.changeType(1))
        # rb1.config(bg="#272323", fg="#ffffff")
        rb1.pack(anchor="n")
        rb2 = tk.Radiobutton(self.typeFrame, text="LSTM", variable=self.model_type, value=2, command=lambda: self.changeType(2))
        # rb2.config(bg="#272323", fg="#ffffff")
        rb2.pack(anchor="n")



        self.model_type.set(1)

        self.btnFrame = tk.LabelFrame(self, text="", padx=5, pady=5)

        self.btnFrame.grid(row=1, column=2, rowspan=2)
        self.btnFrame.config(bd=1)

        self.train_btn = tk.Button(self.btnFrame)
        self.train_btn["text"] = " Train "
        self.train_btn["command"] = self.train
        self.train_btn.config(width = 10)
        self.train_btn.grid(row=1, column=2,pady=0, padx=5)

        self.predict_btn = tk.Button(self.btnFrame)
        self.predict_btn["text"] = "Predict"
        self.predict_btn["command"] = self.predict
        self.predict_btn.config(width = 10)

        self.predict_btn.grid(row=2, column=2,pady=0, padx=5)

        self.out = tk.Text(self, state='disabled',height=4, width = 30, background="#272323", fg="#ffffff")
        self.out.tag_config("right", justify=tk.RIGHT)
        self.out.grid(row = 4, columnspan=3, pady=20, padx=20)

        self.out.delete(1.0, tk.END)
        self.printLine("Started")

    def printLine(self, text):
        self.out.configure(state='normal')
        if(self.lineCount> self.lineMax):
            self.out.delete(1.0, tk.END)
            self.lineCount = 0
        else:
            self.out.insert(tk.END, "\n", "right")
        self.out.insert(tk.END, text, "right")
        self.out.configure(state='disabled')
        self.lineCount += 1


    def changeType(self, type):
        self.model_built = False
        self.model_type.set(type)

    def buildLSTMModel(self):
        print("Building LSTM...")
        self.printLine('Building LSTM....')

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True,
                                       input_shape=self.input_shape))
        self.model.add(tf.keras.layers.LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
        self.model.add(tf.keras.layers.Dense(units=N_CATEGORIES, activation='softmax'))

        adam = tf.keras.optimizers.Adam()
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        self.model_built = True


    def buildCNNModel(self):
        print("Building CNN...")
        self.printLine('Building CNN....')

        self.input_shape = self.input_shape + [1]

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(4, 4),
                                   strides=(4, 4),  # X and Y to move the window by
                                   activation=tf.nn.sigmoid,
                                   input_shape=self.input_shape),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.Conv2D(64, (4, 4), activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1000, activation=tf.nn.relu),
            tf.keras.layers.Dense(N_CATEGORIES, activation='softmax')
        ])
        adam = tf.keras.optimizers.Adam(lr=LEARN_RATE)
        self.model.compile(optimizer=adam,
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
        self.model_built = True

    def train(self):
        self.printLine("Model built: " + str(self.model_built))
        self.valid = False
        window = tk.Toplevel(self)
        window.grab_set()
        tk.Label(window, text="Batch Size: ", anchor="e").grid(row=0, column=0)
        tk.Label(window, text="Epochs: ", anchor="e").grid(row=1, column=0)


        batch_entry = tk.Entry(window)
        batch_entry.grid(row=0, column=1)

        epoch_entry = tk.Entry(window)
        epoch_entry.grid(row=1, column=1)

        def go(obj):
            obj.valid = False

            try:
                obj.n_epoch = int(epoch_entry.get())
                obj.batch_size = int(batch_entry.get())
                if obj.n_epoch == 0 or obj.batch_size == 0:
                    raise ValueError
                else:
                    obj.valid = True
                    obj.printLine("Preparing...")
                    window.destroy()
            except ValueError:
                return

        go_btn = tk.Button(window, command=lambda: go(self))
        go_btn["text"] = "Go"
        go_btn.grid(row=2, column=1)

        self.wait_window(window)

        if not self.valid:
            self.printLine("Cancelling..")
            return

        x_train, x_test, y_train, y_test = self.getData()

        if not self.model_built or self.model == None:
            if (self.model_type == tk.IntVar(value=1)):
                self.buildCNNModel()
                x_train = x_train.reshape([-1, 96, 1366, 1])
                x_test = x_test.reshape([-1, 96, 1366, 1])

            elif (self.model_type == tk.IntVar(value=2)):
                self.buildLSTMModel()

        history = AccuracyHistory()


        self.printLine("Training...")

        self.model.fit(x_train, y_train,
                  batch_size=self.batch_size,
                  epochs=self.n_epoch,
                  verbose=1,
                  validation_data=(x_test, y_test),
                  callbacks=[history])

        score = self.model.evaluate(x_test, y_test)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        print("Saving...")

        string = 'Loss:' +  str(score[0]) + "\nAcc:" + str(score[1])
        messagebox.showinfo("Test", string)




        self.printLine('Saving...')

        if self.model_type == tk.IntVar(value=2):
            self.model.save_weights('./LSTMweights')
            print("Saved to: ./LSTMweights")
            self.printLine('Saved to: ./LSTMweights')

        elif self.model_type == tk.IntVar(value=1):
            self.model.save_weights('./CNNweights')
            self.printLine('Saved to: ./CNNweights')


        plt.plot(range(1, self.n_epoch + 1), history.acc)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()

        plt.plot(range(1, self.n_epoch + 1), history.loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

        return score

    def getData(self):
        labels = np.load("labels.npy")
        data = np.load("data.npy")
        data = np.ndarray.astype(data, "float32")
        labels = np.ndarray.astype(labels, "int")
        # randomly shuffle data
        rng_state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        x_train = data[0:train_size]
        x_test = data[train_size:]

        x_train = x_train.reshape([-1, 96, 1366])
        x_test = x_test.reshape([-1, 96, 1366])

        y_train = labels[0:train_size]
        y_test = labels[train_size:]

        return x_train, x_test, y_train, y_test

    def predict(self):
        classes = ["jazz", "blues", "reggae", "pop", "disco", "country", "metal", "hiphop", "rock", "classical"]
        # classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

        # self.grab_set()
        if (not self.model_built) or self.model == None:
            if (self.model_type == tk.IntVar(value=1)):
                self.buildCNNModel()

                self.model.load_weights("./CNNweights")
            elif (self.model_type == tk.IntVar(value=2)):
                self.buildLSTMModel()
                with open("weights") as f:
                    for i in range(1):
                        self.printLine(f.readline())
                self.model.load_weights("./LSTMweights")

        self.printLine("Model built, choose file...")


        Tk().withdraw()
        path = askopenfilename(initialdir="./data")
        if path == "":
            self.printLine("please choose a file")
            return
        self.printLine("Predicting...")

        test = False
        if(test):
            score = self.model.evaluate(self.getData()[0], self.getData()[2])

            string = 'Loss:\n\t' +  str(score[0]) + "\nAcc:\n\t" + str(score[1])
            messagebox.showinfo("Test", string)

        y, sr = lb.load(path, mono=True)
        spectogram = lb.feature.melspectrogram(y=y, sr=sr, n_mels=96, n_fft=2048, hop_length=256)
        spectogram = lb.power_to_db(spectogram, ref=np.max)

        predict_data = np.empty([1, 1366, 96])
        np.append(predict_data, spectogram)

        prediction = self.model.predict(predict_data.reshape(1, 96, 1366))
        pred_class = prediction.argmax()
        self.printLine("Built spectrogram...")
        string = "Computer thinks the genre of this song is:\n" + classes[pred_class]
        string += "\n\nwith:\n" + str(prediction[0][pred_class]*100) + "% certainty"
        messagebox.showinfo("Prediction", string)

        self.printLine("Done")





class AccuracyHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.loss = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
    quit()







