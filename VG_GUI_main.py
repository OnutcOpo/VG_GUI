import tkinter as tk 
from tkinter import *
from tkinter import messagebox
import torch
import os
from pprint import pprint
from omegaconf import OmegaConf
from IPython.display import Audio, display


def select():
    header.config(text=f"Выбран {voice.get()}")
    balabol=voice.get() 
    select_lb.config(text=f"Выбран {balabol}")

def waving_bmi():
   ssml_1 = """
              <speak>
              <p>
                  <break time="500ms"/>

            """
   ssml_2 = """

              </p>
              </speak>
            """
   ssml = ssml_1 + text_tf.get() + ssml_2
   audio = model.apply_tts(ssml_text = ssml,
                        speaker=voice.get(),
                        sample_rate=sample_rate,
                        put_accent=put_accent,
                        put_yo=put_yo)
   display(Audio(audio, rate=sample_rate))
    

   audio_paths = model.save_wav(ssml_text = ssml,
                             speaker=voice.get(),
                             sample_rate=sample_rate,
                             put_accent=put_accent,
                             put_yo=put_yo)
   count = 0
   f_exist = True
   while f_exist:
      count = count + 1
      file_name = f"wav/{voice.get()}_{str(count)}.wav"
      if not os.path.exists(file_name):
           f_exist = False
                 

   os.rename("test.wav", file_name)
   select_lb.config(text=f"Готово! Выбран {balabol}")






window = Tk()
window.title("Голосилка")
window.geometry('1024x400+10+10')
#выбор голоса
frame = LabelFrame(
   window, #Обязательный параметр, который указывает окно для размещения Frame.
   width=500,
   text="Озвучивает...",
   padx = 10, #Задаём отступ по горизонтали.
   pady = 10, #Задаём отступ по вертикали.
   borderwidth=1
)
frame.place(x=10, y=50) #Не забываем позиционировать виджет в окне. Здесь используется метод pack. С помощью свойства expand=True указываем, что Frame заполняет весь контейнер, созданный для него.

position = {"padx":6, "pady":6, "anchor":NW}
 
voice_1 = "aidar"
voice_2 = "baya"
voice_3 = "kseniya"
voice_4 = "xenia"
voice_5 = "eugene"

voice = StringVar(value=voice_1)    # по умолчанию будет выбран элемент с value=voice_1
 
header = Label(frame, textvariable=voice)
header.pack(**position)


voice_1_btn = Radiobutton(frame, text=voice_1, value=voice_1, variable=voice, command=select)
voice_1_btn.pack(**position)

voice_2_btn = Radiobutton(frame, text=voice_2, value=voice_2, variable=voice, command=select)
voice_2_btn.pack(**position)

voice_3_btn = Radiobutton(frame, text=voice_3, value=voice_3, variable=voice, command=select)
voice_3_btn.pack(**position)

voice_4_btn = Radiobutton(frame, text=voice_4, value=voice_4, variable=voice, command=select)
voice_4_btn.pack(**position)

voice_5_btn = Radiobutton(frame, text=voice_5, value=voice_5, variable=voice, command=select)
voice_5_btn.pack(**position)

balabol = voice.get()

text_lb = Label(
   window,
   text="Текст для озвучивания  "
)
text_lb.place(x=10, y=10)

text_tf = Entry(
   window, #Используем нашу заготовку с настроенными отступами.
   width=120
)
text_tf.insert(0, "Привет, народ!")
text_tf.place(x=150, y=10)

wav_btn = Button(
   window, #Заготовка с настроенными отступами.
   text='Озвучить быстро!', #Надпись на кнопке.
   command=waving_bmi #Позволяет запустить событие с функцией при нажатии на кнопку.
)
wav_btn.place(x=10, y=350)

select_lb = Label(
   window,
   text=f"Выбран {voice.get()}"
)
select_lb.place(x=150, y=350)


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
#balabol = voice.get()
put_accent=True
put_yo=True


window.mainloop()
