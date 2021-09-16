# Dependency-based Mixture Language Models

| Table of Contents |
|-|
| [Setup](#setup)|
| [Training](#training)|
| [Generation](#generation)|
| [Evaluation](#evaluation)|



## **Setup**
------
### **Dependencies**

Install `fairseq` , and place the code file in `custom` into the corresponding path of fairseq's source code.

Download the [GPT2-base](https://huggingface.co/gpt2), and place the files in `~/pretrained_model/gpt2-base`

Then to run Transformer-based models' scripts, enter `DPLM-Transformer` folder:
```bash
cd DPLM-Transformer/
```

### **Data preprocess**

First, you can download the processed data from [here](https://drive.google.com/file/d/1Z5K_T0-CKg3E_ksSOi2wS_2hYmI6CLR3/view?usp=sharing)

Then, extract and place them it in the parent directory `data/` 

For custom dataset, you can use the [HPSG-Neural-Parser](https://github.com/DoodleJZ/HPSG-Neural-Parser) to get the dependency parse tree for each sentence in the datasets. For train/valid/test data, rename the dependency head file as `train/valid/test.head`, and place it in `data/YOUR_DATASET/dependency` . Then, to preprocess the data for Transformer by fairseq:
```bash
sh scripts/preprocess_data.sh
```
To preprocess the data for GPT-2 by fairseq:
```bash
sh scripts/encode_data_gpt2.sh
sh scripts/preprocess_gpt2_data.sh
```
Please specify `TEXT` in scripts according to your data's path.

## **Training**
------
\*We tested these scripts on a machine with NVIDIA GTX 1080Ti 11GB gpus 
If you get OOM errors, try decreasing ```MAX_TOKEN``` of the training scripts. 

To train base Transformer LM:
```bash
sh scripts/train_base_transformer_lm.sh
```

To train base GPT-2:
```bash
sh scripts/train_gpt2_base.sh
```

To train DM-Transformer, first train Transformer by Dependency Modeling:
```bash
sh scripts/train_dependency_decoder.sh
```
Then finetuning by MLE:
```bash
sh scripts/train_dp_transformer_lm.sh
```

To train DM-GPT-2, 
```bash
sh scripts/train_dpgpt2.sh
```
Please specify `TEXT` in scripts according to your data's path.

## **Generation**
------
To sample from Transformer or DM-Transformer:
```bash
sh scripts/sampling.sh
```

To sample from GPT-2 or DM-GPT-2:
```bash
sh scripts/gpt2_sampling.sh
```

To extract generated text from the sample output:
```bash
sh scripts/cut_samples.sh
```

If you want to decode text generated by GPT-2 or DM-GPT-2,  run:
```bash
sh scripts/decode_gpt2_txt.sh
```

Please specify `Model` in scripts according to the model checkpoint path, and specify `TEXT` according to your data's path.

## **Evaluation**
------
Code (`eval_ppl_by_gpt2.py`) is used to calculated the GPT-2 Perplexity of generated sentences, and it can evaluate all the sentences files in one folder:
```bash
python eval_ppl_by_gpt2.py --folderpath FOLDER_FOR_EVALUATION --model_file PATH_TO_GPT2CHECKPOINT
```
Code (`eval_sentences.py`) is used to evaulate the automatic metrics for unconditional text generation task:
```bash
python eval_sentences.py --folderpath FOLDER_FOR_EVALUATION 
```
Code (`eval_stories.py`) is used to evaulate the automatic metrics of generated story endings:
```bash
python eval_stories.py --folderpath FOLDER_FOR_EVALUATION 
```

Besides, we also calculate the [UNION](https://github.com/thu-coai/UNION) and [BERTScore](https://github.com/Tiiiger/bert_score) and for the story ending generation task.