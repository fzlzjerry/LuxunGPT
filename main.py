import os
import random
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2');
tokenizer = GPT2Tokenizer.from_pretrained(('gpt2'));

