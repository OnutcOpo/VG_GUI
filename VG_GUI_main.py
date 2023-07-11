import tkinter as tk 
from tkinter import *
from tkinter import messagebox
import torch
from pprint import pprint
from omegaconf import OmegaConf
from IPython.display import Audio, display

local_file = 'model.pt'
language = 'ru'
model_id = 'v3_1_ru'
device = torch.device('cpu')

model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)
model.to(device)  # gpu or cpu

model.speakers

sample_rate = 48000
speaker = 'xenia'
put_accent=True
put_yo=True

def waving_bmi():
    audio = model.apply_tts(text=text_tf.get(),
                        speaker=speaker,
                        sample_rate=sample_rate,
                        put_accent=put_accent,
                        put_yo=put_yo)
    display(Audio(audio, rate=sample_rate))

    audio_paths = model.save_wav(text=text_tf.get(),
                             speaker=speaker,
                             sample_rate=sample_rate,
                             put_accent=put_accent,
                             put_yo=put_yo)




window = Tk()
window.title("Генератор голосов")
window.geometry('800x600+10+10')

frame = Frame(
   window, #Обязательный параметр, который указывает окно для размещения Frame.
   padx = 10, #Задаём отступ по горизонтали.
   pady = 10 #Задаём отступ по вертикали.
)
frame.pack(expand=True) #Не забываем позиционировать виджет в окне. Здесь используется метод pack. С помощью свойства expand=True указываем, что Frame заполняет весь контейнер, созданный для него.
text_lb = Label(
   frame,
   text="Текст для озвучивания  "
)
text_lb.grid(row=1, column=1)

text_tf = Entry(
   frame, #Используем нашу заготовку с настроенными отступами.
)
text_tf.grid(row=1, column=2)

wav_btn = Button(
   frame, #Заготовка с настроенными отступами.
   text='Озвучить нахуй!', #Надпись на кнопке.
   command=waving_bmi #Позволяет запустить событие с функцией при нажатии на кнопку.
)
wav_btn.grid(row=5, column=1) #Размещаем кнопку в ячейке, расположенной ниже, чем наши надписи, но во втором столбце, то есть под ячейками для ввода информации.



window.mainloop()