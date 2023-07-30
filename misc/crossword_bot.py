"""
Created on Saturday July 29th, 2023

@author: David J. Gayowsky

Simple crossword bot for uOttawa Trivia Discord server.

"""

#######################################################################

import discord
import os 
import asyncio

#######################################################################

#Define a function to allow a user to ask for instructions:

#Define a function to allow a user to set a crossword clue:

#Define a function to allow a user to answer the crossword:

#Define a function to evaluate crossword answer:

#######################################################################

def get_chelp():
    msg = 'Hello! Looks like you need some help with Crossword Bot. \n \
                To start a crossword puzzle, type $crossword followed by \
                <clue> <number of letters> <answer>. \n \
                To answer the current crossword puzzle, type $cans followed by <answer>. \n \
                To cancel your current crossword puzzle, type $ccancel, which will reveal the answer. \n \
                To check points, type $cpoints, which will display the top 5 goals and assists points. \n \
                Happy crosswording!'
    return(msg)

class MyClient(discord.Client):

    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        #Don't reply to ourselves...
        if message.author.id == self.user.id:
            return
        
        if message.content.startswith('$chelp'):
            helpmessage = get_chelp()
            await message.reply(helpmessage, mention_author=True)

intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)

client.run(os.getenv('TOKEN'))