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
async def crossword(self, ctx, clue, numletters: int, answer):
    #Delete the message so nobody can see the answer.
    await ctx.message.delete()
    #Declare the challenge to the channel.
    init_message = await ctx.send(f"{ctx.author.mention} has declared a new crossword challenge! Your challenge is: \n {clue}, {numletters} letters.")

    #Check that our answer is correct and in our channel, and is a reply to the bot.
    def check(m):
        if m.reference is not None:
            if m.reference.message_id == init_message.id and m.content == answer:
                return True
        return False
    
    #Wait for a reply, make sure that the reply we're getting is a reply to the bot's message and contains the answer:
    user_ans = await self.bot.wait_for('message', check=check)

    def check_assists(m):
        if m.reference is not None:
            if m.reference.message_id == assists_msg.id and m.author == user_ans.author and m.mentions:
                return True
        return False

    #If they've guessed the answer correctly:
    if user_ans == answer:
        #Congrats!
        await ctx.send(f":tada: Congratulations {user_ans.author.mention}! :tada: You solved the crossword clue! The answer was {answer}.")
        #Ask for assists:
        assists_msg = await ctx.send(f"Did anyone help you guess? Please reply to this message with your assist. :hockey:")

        try:
            user_assists_msg = await self.bot.wait_for('message', timeout=120.0, check=check_assists)
        except asyncio.TimeoutError:
            await ctx.send(f"Timeout! You didn't declare any assists. Thanks for playing!")
        else:
            assist_member = user_assists_msg.mentions.members.first()
            await ctx.send(f"Assist accepted! Thanks {assist_member.mention} for the flow, the snow, and the apple :sunglasses: :hockey:")


#Define a bot command to try and answer the crossword puzzle.
'''@bot.command(description='Answer the current crossword puzzle.')
async def ans(ctx, userans):
    if userans == answer:
        return'''


#######################################################################

#client = MyClient(intents=intents)

bot.run(os.getenv('TOKEN'))