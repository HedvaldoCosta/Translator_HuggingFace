# Tradução de textos utilizando transformers

## Sobre tradução
Em processamento de linguagem natural, a tarefa de traduzir é realizada ao converter uma sequência de texto de um idioma para outro. Tradução é um modelo sequence-to-sequence que utiliza as partes de codificação e decodificação dentro da arquitetura do Transformer. As camadas de atenção do codificador podem acessar todas as palavras da frase inicial, enquanto as camadas de atenção do decodificador podem acessar apenas as palavras posicionadas antes de uma determinada palavra na entrada.

## Sobre transformers
As arquiteturas de transformadores facilitaram a construção de modelos de maior capacidade e o pré-treinamento tornou possível utilizar efetivamente essa capacidade para uma ampla variedade de tarefas. Transformers é uma biblioteca de código aberto com o objetivo de abrir esses avanços para a comunidade de aprendizado de máquina mais ampla. A biblioteca consiste em arquiteturas Transformer de última geração cuidadosamente projetadas em uma API unificada. A arquitetura do transformers é dimensionada com dados de treinamento e tamanho do modelo que facilitam o treinamento paralelo eficiente, junto ao seu pré-treinamento que permite o treino com longos corpus para facilitar a adaptação em tarefas com forte desempenho.Corpus é ‘Um conjunto de dados lingüísticos (pertencentes ao uso oral ou escrito da língua, ou a ambos), sistematizados segundo determinados critérios, suficientemente extensos em amplitude e profundidade, de maneira que sejam representativos da totalidade do uso lingüístico ou de algum de seus âmbitos, dispostos de tal modo que possam ser processados por computador, com a finalidade de propiciar resultados vários e úteis para a descrição e análise’ (Sanchez, 1995, pp. 8-9).

## Sobre pipeline
Ter que fazer o pré-processamento, passar as entradas pelo modelo e o pós-processamento é algo que demanda muito tempo. No entanto, o pipeline é uma função do Transformer que já engloba todos esses passos. 
<p align="center">
  <img width="700" height="300" src="https://user-images.githubusercontent.com/67663958/181862944-29f31644-a0d4-4f0b-a74f-c31320528493.png" >
</p>

### Pré-processamento com tokenizer
Os modelos Transformer não podem processar texto bruto diretamente, portanto, a primeira etapa do nosso pipeline é converter as entradas de texto em números que o modelo possa entender utilizando o tokenizer. O tokenizer irá dividir a entrada em tokens e os mapear para inteiros. Assim que a tokenização esteja completa, receberemos um dicionário pronto para alimentarmos o modelo.
```python
from transformers import AutoTokenizer 

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english" 
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
Os modelos Transformer só aceitam tensores como entrada, então é necessário converter a lista de tokens em tensores.
```python
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

### Modelo
Para cada entrada de modelo, recuperaremos um vetor de alta dimensão representando o entendimento contextual dessa entrada pelo modelo Transformer. Se diz um vetor de alta dimensão pois a saída vetorial do módulo Transformer geralmente é grande.
```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
```
As cabeças do modelo pegam o vetor de alta dimensão de estados ocultos como entrada e os projetam em uma dimensão diferente. A saída do modelo Transformer é enviada diretamente para o cabeçote do modelo para ser processado.
<p align="center">
  <img width="700" height="300" src="https://user-images.githubusercontent.com/67663958/181863238-52063f79-c361-4adc-a883-c6316b977d4a.png" >
</p>

A camada de agrupamento do modelo converte cada token de entrada em um vetor que representa o token associado. As camadas subsequentes manipulam esses vetores usando o mecanismo de atenção para produzir a representação final das frases.
```python
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
```

### Pós-processamento
Normalização das pontuações brutas, emitidas pela última camada do modelo, para a verificação de probabilidades.
```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```
## Código principal

### Sobre o modelo utilizado:
Originalmente os modelos opus-mt-tc-big-en-pt foram treinados utilizando o framework Marian NMT. Os modelos foram convertidos para pyTorch usando a biblioteca de transformadores do huggingface. Os dados de treinamento são obtidos do OPUS e os pipelines de treinamento usam os procedimentos do OPUS-MT-train.
```python
#Importando a função "pipeline" da biblioteca python que servirá para
#construir o pré-processamento, alimentar o modelo e
#finalizar o pós-processamento
from transformers import pipeline
```
```python
# Traduzindo do inglês para o português
def translator(frase, modelo):
    translate = pipeline("translation", model=modelo)
    return translate(frase)


model_checkpoint = "Helsinki-NLP/opus-mt-tc-big-en-pt"
raw_frase = str(input("Entre com uma frase: "))
translator(frase=raw_frase, modelo=model_checkpoint)
```

## Referências
https://huggingface.co/docs/transformers/tasks/translation

https://huggingface.co/course/chapter1/7?fw=pt

https://aclanthology.org/2020.emnlp-demos.6

https://huggingface.co/docs/transformers/tasks/translation

https://huggingface.co/course/chapter1/7?fw=pt

https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-pt
