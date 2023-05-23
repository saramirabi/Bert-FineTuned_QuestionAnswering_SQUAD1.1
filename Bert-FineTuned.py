import torch


#####Load Fine-Tuned BERT-large
from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

####Load Tokenizer
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

####Ask a Question
question = "How many parameters does BERT-large have?"
answer_text = "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."

# Apply the tokenizer to the input text, treating them as a text-pair.
input_ids = tokenizer.encode(question,answer_text)


####Print the Number of Generated Tokens from Questions and Answers
print('The input has a total of {:} tokens.'.format(len(input_ids)))


#####Print the Tokens with Token_IDs


####To Get the Actual Token for an ID: From Transformers Import BertTokenizer Tokenizer
tokens = tokenizer.convert_ids_to_tokens(input_ids)
for token, id in zip(tokens, input_ids):

    # If this is the [SEP] token, add some space around it to make it stand out.
    if id == tokenizer.sep_token_id:
        print('')

    # Print the token string and its ID in two columns.
    print('{:<12} {:>6,}'.format(token, id))

    if id == tokenizer.sep_token_id:
        print('')

#### Search the input_ids for the first instance of the [SEP] Tokesn
sep_index = input_ids.index(tokenizer.sep_token_id)

####The number of segment A tokens includes the [SEP] token istelf.
num_seg_a= sep_index+1


#####The remainder are segment B
num_seg_b = len(input_ids) - num_seg_a


####Construct the List of 0s an 1s.
segment_ids = [0]*num_seg_a + [1]*num_seg_b
assert len(segment_ids) == len(input_ids)


outputs = model(torch.tensor([input_ids]), token_type_ids= torch.tensor([segment_ids]),return_dict= True)
start_scores = outputs.start_logits
end_scores = outputs.end_logits


####Find the tokens with the highest 'start' and 'end' scores. 
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)


answer = ' '.join(tokens[answer_start:answer_end])
print('Answer: "' +answer+ '"')




import matplotlib.pyplot as plt
import seaborn as sns 

sns.set(style='darkgrid')
plt.rcParams["figure.figsize"] = (16,8)


s_score = start_scores.detach().numpy().flatten()
e_score= end_scores.detach().numpy().flatten()


token_labels=[]
for (i,token) in enumerate(tokens):
    token_labels.append('{:} - {:>2}'.format(token,i))


ax= sns.barplot(x=token_labels, y= s_score , ci= None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
ax.grid(True)
plt.title('Start Word Scores')
plt.show()


ax= sns.barplot(x=token_labels, y= e_score , ci= None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
ax.grid(True)
plt.title('End Word Scores')
plt.show()


import pandas as pd

scores=[]
for(i,token_label) in enumerate(token_labels):
    scores.append({'token_label':token_label, 'score':s_score[i],'marker':'start'})
    scores.append({'token_label':token_label, 'score':e_score[i],'marker':'end'})    

df = pd.DataFrame(scores)

g= sns.catplot(x='token_label',y='score',hue='marker',data=df,kind='bar',height=6, aspect=4)
g.set_xticklabels(g.ax.get_xticklabels(),rotation=90,ha='center')
g.ax.grid(True)









