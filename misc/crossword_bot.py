"""
Created on Saturday July 29th, 2023

@author: David J. Gayowsky

Simple crossword bot for uOttawa Trivia Discord server.

"""

#######################################################################

import discord
import os 
import asyncio
from discord.ext import commands

#######################################################################

intents = discord.Intents.default()
intents.members = True
intents.message_content = True

#######################################################################

cdescription = '''A crossword bot to allow you to play crosswords with your friends.'''

bot = commands.Bot(command_prefix='$c', description=cdescription, intents=intents)

#Bot login.
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')

#Define a bot command to set a new crossword puzzle.
@bot.command(description='Set a crossword puzzle.')
async def crossword(ctx, clue, numletters: int, answer):
    await ctx.message.delete()
    await ctx.send(f"{ctx.author.mention} has declared a new crossword challenge! Your challenge is: \n {clue}, {numletters} letters.")

#Define a bot command to try and answer the crossword puzzle.
@bot.command(description='Answer the current crossword puzzle.')
async def ans(ctx, userans):
    if userans == answer:
        return


#######################################################################

#client = MyClient(intents=intents)

bot.run(os.getenv('TOKEN'))