#!/usr/bin/env python
# coding: utf-8

# # Colab Pro benefits

# In[1]:


gpu_info = get_ipython().getoutput('nvidia-smi')
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)


# In[2]:


from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')


# # Importer les données WMT14

# In[3]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


get_ipython().system('cp -r /content/drive/MyDrive/Projet2_NLP/data /content/')


# # Joey NMT Demo - Guide
# Source: https://github.com/joeynmt/joeynmt/blob/master/joey_demo.ipynb
# 
# Nous allons entraîner un modèle de Transformer NMT de base de l'anglais vers le français. 
# 
# **Important:** Avant de commencer, il est indispensable de choisir le type d'exécution de GPU. 
# 
# 

# ## Installation

# Installez la bonne version de PyTorch pour Joey NMT. Il faudra peut-être redémarrer le colab après avoir installé Joey NMT.

# In[5]:


get_ipython().system('pip install torch==1.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html')


# In[6]:


get_ipython().system('pip install joeynmt')


# ## Continuing with the extracted data..

# ### Word count

# In[1]:


get_ipython().system('wc -l /content/data/en-fr/*')


# Nous n'utiliserons qu'un sous-ensemble de données pour développement et test.

# In[2]:


get_ipython().system('head -2007723  /content/data/en-fr/train_full.en > /content/data/en-fr/train.en')
get_ipython().system('head -2007723  /content/data/en-fr/train_full.fr > /content/data/en-fr/train.fr')


# Let's have a look at our data

# In[3]:


get_ipython().system('head -2 data/en-fr/train.* data/en-fr/dev.* data/en-fr/test.*')


# ## Entraînement d'un modèle de Subword

# Nous utiliserons la bibliothèque `subword_nmt` pour diviser les mots en sous-mots (BPE) en fonction de leur fréquence dans le corpus d'apprentissage.
# 
# 

# In[4]:


import os


# In[5]:


src_lang = 'en'
trg_lang = 'fr'
bpe_size = 20000    # taille du vocabulaire à définir
datadir = '/content/data/en-fr/'
name = f'{src_lang}_{trg_lang}_bpe{bpe_size}'


train_src_file = os.path.join(datadir, f'train.{src_lang}')
train_trg_file = os.path.join(datadir, f'train.{trg_lang}')
train_joint_file = os.path.join(datadir, f'train.{src_lang}-{trg_lang}')
dev_src_file = os.path.join(datadir, f'dev.{src_lang}')
dev_trg_file = os.path.join(datadir, f'dev.{trg_lang}')
test_src_file = os.path.join(datadir, f'test.{src_lang}')
test_trg_file = os.path.join(datadir, f'test.{trg_lang}')
src_files = {'train': train_src_file, 'dev': dev_src_file, 'test': test_src_file}
trg_files = {'train': train_trg_file, 'dev': dev_trg_file, 'test': test_trg_file}


vocab_src_file = os.path.join(datadir, f'vocab.{bpe_size}.{src_lang}')
vocab_trg_file = os.path.join(datadir, f'vocab.{bpe_size}.{trg_lang}')
bpe_file = os.path.join(datadir, f'bpe.codes.{bpe_size}')


# Entraîner un modèle BPE avec 20000 symboles pour les deux langues conjointement.
# 
# 

# In[6]:


get_ipython().system(' cat $train_src_file $train_trg_file > $train_joint_file')

get_ipython().system(' subword-nmt learn-bpe   --input $train_joint_file   -s $bpe_size   -o $bpe_file')


# Ce fichier contient les fusions de séquences de caractères qui composent les sous-mots.

# In[7]:


get_ipython().system(' head $bpe_file')


# Nous appliquons les fusions BPE apprises aux données de formation, de développement et de test.
# 

# In[8]:


src_bpe_files = {}
trg_bpe_files = {}
for split in ['train', 'dev', 'test']:
  src_input_file = src_files[split]
  trg_input_file = trg_files[split]
  src_output_file = src_input_file.replace(split, f'{split}.{bpe_size}.bpe')
  trg_output_file = trg_input_file.replace(split, f'{split}.{bpe_size}.bpe')
  src_bpe_files[split] = src_output_file
  trg_bpe_files[split] = trg_output_file

  get_ipython().system(' subword-nmt apply-bpe     -c $bpe_file     < $src_input_file > $src_output_file')

  get_ipython().system(' subword-nmt apply-bpe     -c $bpe_file     < $trg_input_file > $trg_output_file')


# Les données subword-split contiennent `@@ ` pour indiquer où les mots ont été divisés en sous-mots.

# In[9]:


get_ipython().system(' head data/en-fr/dev.20000.bpe.en')


# In[10]:


get_ipython().system(' head data/en-fr/dev.20000.bpe.fr')


# ## Préparer le vocabulaire

# À partir des données d'entraînement prétraitées, nous extrayons le vocabulaire final du modèle de traduction. Il doit contenir tous les sous-mots nécessaires pour représenter les données d'entraînement source et cible.

# In[11]:


get_ipython().system(' wget https://raw.githubusercontent.com/joeynmt/joeynmt/master/scripts/build_vocab.py')


# In[12]:


vocab_src_file = src_bpe_files['train']
vocab_trg_file = trg_bpe_files['train']
bpe_vocab_file = os.path.join(datadir, f'joint.{bpe_size}bpe.vocab')

get_ipython().system(' python build_vocab.py    $vocab_src_file $vocab_trg_file   --output_path $bpe_vocab_file')


# # Configuration du modèle

# Joey NMT lit les hyperparamètres de modèle et d'entraînement à partir d'un fichier de configuration. Nous générons ceci maintenant pour configurer les chemins aux endroits appropriés.
# 
# La configuration ci-dessous crée un petit modèle Transformer avec des intégrations partagées entre la langue source et la langue cible sur la base des vocabulaires de sous-mots créés ci-dessus.

# In[13]:


# Create the config
config = """
name: "{name}_transformer"

data:
    src: "{source_language}"
    trg: "{target_language}"
    train: "{datadir}/train.{bpe_size}.bpe"
    dev:   "{datadir}/dev.{bpe_size}.bpe"
    test:  "{datadir}/test.{bpe_size}.bpe"
    level: "bpe"
    lowercase: False                
    max_sent_length: 100             # Extend to longer sentences. **
    src_vocab: "{vocab_src_file}"
    trg_vocab: "{vocab_trg_file}"

testing:
    beam_size: 5
    alpha: 1.0
    sacrebleu:                      # sacrebleu options
        remove_whitespace: True     # `remove_whitespace` option in sacrebleu.corpus_chrf() function (defalut: True)
        tokenize: "none"            # `tokenize` option in sacrebleu.corpus_bleu() function (options include: "none" (use for already tokenized test data), "13a" (default minimal tokenizer), "intl" which mostly does punctuation and unicode, etc) 

training:
    #load_model: "models/{name}_transformer/1.ckpt" # if uncommented, load a pre-trained model from this checkpoint
    random_seed: 42
    optimizer: "adam"
    normalization: "tokens"
    adam_betas: [0.9, 0.999] 
    scheduling: "plateau"           # Alternative: try switching from plateau to Noam scheduling
    patience: 5                     # For plateau: decrease learning rate by decrease_factor if validation score has not improved for this many validation rounds.
    learning_rate_factor: 0.5       # factor for Noam scheduler (used with Transformer)
    learning_rate_warmup: 1000      # warmup steps for Noam scheduler (used with Transformer)
    decrease_factor: 0.7
    loss: "crossentropy"
    learning_rate: 0.0003
    learning_rate_min: 0.00000001
    weight_decay: 0.0
    label_smoothing: 0.1
    batch_size: 4096
    batch_type: "token"
    #eval_batch_size: 3600
    #eval_batch_type: "token"
    #batch_multiplier: 1
    early_stopping_metric: "loss"
    epochs: 20                     # Decrease for when playing around and checking of working. Around 30 is sufficient to check if its working at all
    validation_freq: 2000          # Set to at least once per epoch.
    logging_freq: 200
    eval_metric: "bleu"
    model_dir: "models/{name}_transformer"
    overwrite: True               # Set to True if you want to overwrite possibly existing models. 
    shuffle: True
    use_cuda: True
    max_output_length: 100
    print_valid_sents: [0, 1, 2, 3]
    keep_last_ckpts: 3    

model:
    initializer: "xavier"
    bias_initializer: "zeros"
    init_gain: 1.0
    embed_initializer: "xavier"
    embed_init_gain: 1.0
    tied_embeddings: True       # Requires joint vocabulary.
    tied_softmax: True
    encoder:
        type: "transformer"
        num_layers: 6
        num_heads: 8             # Increase 4 to 8 for larger data.
        embeddings:
            embedding_dim: 512   # Increase 256 to 512 for larger data.
            scale: True
            dropout: 0.2
        # typically ff_size = 4 x hidden_size
        hidden_size: 512         # Increase 256 to 512 for larger data.
        ff_size: 2048            # Increase 1024 to 2048 for larger data.
        dropout: 0.3
    decoder:
        type: "transformer"
        num_layers: 6
        num_heads: 8              # Increase 4 to 8 for larger data.
        embeddings:
            embedding_dim: 512    # Increase 256 to 512 for larger data.
            scale: True
            dropout: 0.2
        # typically ff_size = 4 x hidden_size
        hidden_size: 512         # TODO: Increase 256 to 512 for larger data.
        ff_size: 2048            # TODO: Increase 1024 to 2048 for larger data.
        dropout: 0.3
""".format(name=name, source_language=src_lang, target_language=trg_lang,
           datadir=datadir, vocab_src_file=bpe_vocab_file, 
           vocab_trg_file=bpe_vocab_file, bpe_size=bpe_size)
with open("transformer_{name}.yaml".format(name=name),'w') as f:
    f.write(config)


# In[13]:





# # Entraînement

# Cela va prendre du temps. Le journal rapporte le processus de formation, recherchez les impressions d'exemples de traductions et les notes d'évaluation BLEU pour avoir une idée de la qualité actuelle.
# 
# Le journal est également stocké dans le répertoire du modèle au sein de ce runtime (inspectez les fichiers dans le menu de gauche). Vous y trouverez également un rapport récapitulatif de toutes les validations. Nous utiliserons également TensorBoard pour visualiser la progression de l'entraînement lors de vos déplacements. Cela nécessite l'activation des cookies dans le navigateur.
# 
# Après 12h au plus tard, Colab se déconnectera, donc pour vous assurer que votre progression n'est pas perdue, téléchargez les points de contrôle depuis le répertoire des modèles de temps en temps. Vous pourrez plus tard les recharger si les hyperparamètres du modèle correspondent.

# In[14]:


get_ipython().system(' pip install --upgrade sacrebleu==1.5.1')
# conflit de version sacreBleu==2.0 => donc réinstaller l'ancienne version


# In[33]:


# Chargez l'extension de bloc-notes TensorBoard. Il sera vide au début.
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[32]:


get_ipython().system('kill 529')


# In[34]:


get_ipython().run_line_magic('tensorboard', '--logdir models/en_fr_bpe20000_transformer/tensorboard')


# In[18]:


get_ipython().system('python -m joeynmt train transformer_en_fr_bpe20000.yaml')


# ## Continuer l'entraînement après interruption

# Pour continuer après une interruption, la configuration doit être modifiée à 2 endroits :
# 1. `load_model` pour pointer vers le point de contrôle à charger.
# 2. `model_dir` pour créer un nouveau répertoire.
# 

# In[27]:


ckpt_number = 2000
reload_config = config.replace(
    f'#load_model: "models/{name}_transformer/1.ckpt"', f'load_model: "models/{name}_transformer/{ckpt_number}.ckpt"').replace(
        f'model_dir: "models/{name}_transformer"', f'model_dir: "models/{name}_transformer_continued"')
with open("transformer_{name}_reload.yaml".format(name=name),'w') as f:
    f.write(reload_config)


# Joey NMT reprend alors l'entraînement à partir de là.
# 
# 

# In[28]:


get_ipython().system('python -m joeynmt train transformer_epo_eng_bpe20000_reload.yaml')


# Sinon, supprimer l'ancien dossier models, puis réfaire l'entraînement. 

# In[ ]:


# !rm -fr models  # pour supprimer un dossier de models , par exemple pour pouvoir ré-entraîner un autre modèle


# # Évaluation et Prédiction

# Le mode `test` peut être utilisé pour traduire (et évaluer sur) l'ensemble de test spécifié dans la configuration. Nous ne le faisons généralement qu'une seule fois après avoir réglé les hyperparamètres sur l'ensemble de développement.

# In[19]:


get_ipython().system('python -m joeynmt test models/en_fr_bpe20000_transformer/config.yaml')

# pour 10 itérations sur (train, dev, test) = (181k, 1k, 1k)
# dev bleu[none]:   0.32 
# test bleu[none]:   0.25


# 
# 
# Le mode `translate` est plus interactif et prend des invites pour traduire de manière interactive. Attention : cela nécessite d'appliquer les mêmes étapes de pré-traitement à la nouvelle entrée que celles que vous avez appliquées avant l'entraînement du modèle (c'est-à-dire la segmentation en sous-mots).

# In[24]:


from subword_nmt import apply_bpe

with open(bpe_file, "r") as merge_file:
  bpe = apply_bpe.BPE(codes=merge_file)

preprocess = lambda x: bpe.process_line(x.strip())


# In[21]:


my_sentence = "I don't know whether machine translation will eventually get good enough to allow us to browse people's websites in different languages so you can see how they live in different countries."
#my_sentence = 'Please enter a source sentence.'  

# référence : Toutes les parties prenantes auraient intérêt à lancer un autre moteur.


# In[22]:


preprocess(my_sentence)


# Copiez la phrase prétraitée ci-dessus dans le champ lorsque vous y êtes invité ci-dessous. Arrêtez la cellule pour quitter le mode interactif.

# In[23]:


get_ipython().system('python -m joeynmt translate models/en_fr_bpe20000_transformer/config.yaml')


# Vous pouvez également obtenir les n meilleures hypothèses (jusqu'à la taille du faisceau, dans notre exemple 5), pas seulement la plus élevée. Plus votre modèle s'améliore, plus les alternatives devraient être intéressantes.
# 

# In[ ]:


get_ipython().system('python -m joeynmt translate models/en_fr_bpe20000_transformer/config.yaml -n 5')


# In[ ]:





# ## Récupérer le test de prédiction

# In[ ]:


get_ipython().system('python -m joeynmt test models/en_fr_bpe20000_transformer/config.yaml --output_path data/en-fr/out')


# In[ ]:


get_ipython().system('cp -r models/ /content/drive/MyDrive/Projet2_NLP')


# Fin du guide !
# 
