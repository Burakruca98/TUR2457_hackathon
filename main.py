import discord
from discord.ext import commands
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='$', intents=intents)

model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def get_class(image_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open(image_path).convert("RGB")
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    #Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    # Print prediction and confidence score
    return class_name,confidence_score



@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send(f'Hi! I am a bot {bot.user}!')

@bot.command()
async def heh(ctx, count_heh = 5):
    await ctx.send("he" * count_heh)

@bot.command()
async def image_classify(ctx):
    if not ctx.message.attachments:
        await ctx.send("sadece görsel gönderiniz ( jpeg jpg ve png formatını destekler)")
        return
    
    image = ctx.message.attachments[0]
    if not image.filename.lower().endswith((".jpg",".jpeg",".png")):
        await ctx.send("sadece geçerli formatta resim yükleyiniz")
    
    image_path = f"temp_{ctx.author.id}.jpg"
    await image.save(image_path)
    try:
        result ,confidence_score = get_class(image_path)
        if result is None:
            await ctx.send("bu resimi seçemedim")
        else:
            await ctx.send(f"model tahmini: {result} güven skoru: {confidence_score}")

    except:
        await ctx.send("bir hata oluştu")



bot.run("")