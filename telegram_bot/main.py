from scipy import misc
from torchvision import transforms

from PIL import Image
from io import BytesIO
from multiprocessing import Queue, Process
from time import sleep

import os
import sys
import time
import re
import torch
import numpy as np

import utils
from vgg import Vgg16
from model import StyleTransferModel
from telegram_token import token, token_dialog

import requests
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove, ReplyKeyboardMarkup, KeyboardButton, InputTextMessageContent, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler, InlineQueryHandler
import apiai, json

model = StyleTransferModel()
job_queue = Queue()

def worker(bot, queue):
    while True:
        query = queue.get()
        s_model = query.callback_query.data
        save_model = ('saved_models/' + s_model + '.pth')
        print(save_model)
        message = queue.get()
        print(message)
        # Получаем сообщение с картинкой из очереди и обрабатываем ее
        chat_id = message.chat_id
        print("Got image from {}".format(chat_id))

        # получаем информацию о картинке
        image_info = message.photo[-1]
        image_file = bot.get_file(image_info)
        content_image_stream = BytesIO()
        image_file.download(out=content_image_stream)
        output = transfer_style(content_image_stream, save_model)
        
        # теперь отправим назад фото
        output_stream = BytesIO()
        output.save(output_stream, format='PNG')
        output_stream.seek(0)
        bot.send_photo(chat_id, photo=output_stream)
        print("Sent Photo to user")

def photo(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Ваше фото помещено в очередь')
    sleep(5)
    bot.send_message(chat_id=update.message.chat_id, text='Пожалуста ждите...')
    job_queue.put(update.message)


def transfer_style(content_image, model):
    content_image = load_image(content_image)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)

    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        output = style_model(content_image)
    return misc.toimage(output[0])

updater = Updater(token=token) # Токен API к Telegram
dispatcher = updater.dispatcher
# Обработка команд
def startCommand(bot, update):
    keyboard = [[InlineKeyboardButton("/style", callback_data='/style')]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    update.message.reply_text('Привет, давай пообщаемся? Или нажми на кнопку ниже для переноса стиля', reply_markup=reply_markup)
def textMessage(bot, update):
    request = apiai.ApiAI(token_dialog).text_request() # Токен API к Dialogflow
    request.lang = 'ru' # На каком языке будет послан запрос
    request.session_id = 'BatlabAIBot' # ID Сессии диалога (нужно, чтобы потом учить бота)
    request.query = update.message.text # Посылаем запрос к ИИ с сообщением от юзера
    responseJson = json.loads(request.getresponse().read().decode('utf-8'))
    response = responseJson['result']['fulfillment']['speech'] # Разбираем JSON и вытаскиваем ответ
    # Если есть ответ от бота - присылаем юзеру, если нет - бот его не понял
    if response == 'Выберите пожалуйста стиль для переноса':
        keyboard = [[InlineKeyboardButton("candy", callback_data='candy')],
                    [InlineKeyboardButton("mosaic", callback_data='mosaic')],
                    [InlineKeyboardButton("rain_princess", callback_data='rain_princess')],
                    [InlineKeyboardButton("udnie", callback_data='udnie')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text('Выберите пожалуйста стиль для переноса:', reply_markup=reply_markup)    
    elif response:
        bot.send_message(chat_id=update.message.chat_id, text=response)
    else:
        bot.send_message(chat_id=update.message.chat_id, text='Я Вас не совсем понял!')
  
def button(bot, update):
    query = update.callback_query
    query.edit_message_text(text="Выбран стиль: {}".format(query.data))
    sleep(5)
    query.edit_message_text(text="Пожалуйста загрузите фотографию")
    s_model = query.data
    sv_model = ('saved_models/' + s_model + '.pth')
    job_queue.put(update)
def styleCommand(bot, update):
    keyboard = [[InlineKeyboardButton("candy", callback_data='candy')],
                [InlineKeyboardButton("mosaic", callback_data='mosaic')],
                [InlineKeyboardButton("rain_princess", callback_data='rain_princess')],
                [InlineKeyboardButton("udnie", callback_data='udnie')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Выберите пожалуйста стиль для переноса:', reply_markup=reply_markup)
# Хендлеры
start_command_handler = CommandHandler('start', startCommand)
style_command_handler = CommandHandler('style', styleCommand)
text_message_handler = MessageHandler(Filters.text, textMessage)
photo_message_handler = MessageHandler(Filters.photo, photo)
button_call_handler = CallbackQueryHandler(button)
# Добавляем хендлеры в диспетчер
dispatcher.add_handler(start_command_handler)
dispatcher.add_handler(style_command_handler)
dispatcher.add_handler(text_message_handler)
dispatcher.add_handler(button_call_handler)
dispatcher.add_handler(photo_message_handler)
# Сделаем отдельный процесс для того, чтобы обрабатывать картинки
worker_args = (updater.bot, job_queue)
worker_process = Process(target=worker, args=worker_args)
worker_process.start()
# Начинаем поиск обновлений
updater.start_polling(clean=True)
# Останавливаем бота, если были нажаты Ctrl + C
updater.idle()
