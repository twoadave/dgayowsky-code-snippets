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

bot = commands.Bot(command_prefix='&', description=cdescription, intents=intents)

#Bot login.
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')

#Check function to make sure we're only in the allowed channel.
def in_channel(*channels):
    def predicate(ctx):
        return ctx.channel.id in channels
    return commands.check(predicate)

#Define a bot command to set a new crossword puzzle.
#Check to make sure we're only in the allowed channel.
@in_channel(1135273922649133176)
@bot.command(description='Set a crossword puzzle.')
async def crossword(ctx, clue, numletters: int, answer):
    #Delete the message so nobody can see the answer.
    await ctx.message.delete()
    #Declare the challenge to the channel.
    init_message = await ctx.send(f"{ctx.author.mention} has declared a new crossword challenge! Your challenge is: \n {clue}, {numletters} letters.")

    #Check that our answer is a reply to the bot.
    def check(m):
        if m.reference is not None:
            if m.reference.message_id == init_message.id:
                return True
        return False
    
    def check_assists(m):
        if m.reference is not None:
            if m.reference.message_id == assists_msg.id and m.author == user_ans.author and m.mentions:
                return True
        return False
    
    answer_guessed = 0
    
    while answer_guessed == 0:

        #Wait for a reply, make sure that the reply we're getting is a reply to the bot's message:
        user_ans = await bot.wait_for('message', check=check)

        #If they've guessed the answer correctly:
        if user_ans.content == answer:
            answer_guessed = 1
            #Congrats!
            await user_ans.add_reaction("✅")
            await ctx.send(f":tada: Congratulations {user_ans.author.mention}! :tada: You solved the crossword clue! The answer was {answer}.")
            #Ask for assists:
            assists_msg = await ctx.send(f"Did anyone help you guess? Please reply to this message with your assist. :hockey:")

            try:
                user_assists_msg = await bot.wait_for('message', timeout=120.0, check=check_assists)
            except asyncio.TimeoutError:
                await ctx.send(f"Timeout! You didn't declare any assists. Thanks for playing!")
            else:
                assist_member = user_assists_msg.mentions
                await ctx.send(f"Assist accepted! Thanks {assist_member.mention} for the flow, the snow, and the apple :sunglasses: :hockey:")

        else:
            await user_ans.add_reaction("❌")

#######################################################################

#client = MyClient(intents=intents)

bot.run('MTEzNTAwMjY1Nzk5NjY3MzA4NQ.GIPmPX.eCduy3bUyodCRHIH0XETh8_3UOMIxEUG7qG1T0')